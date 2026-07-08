"""HookedModel — the architecture-agnostic base class.

``HookedModel`` is the boundary between experiment-side knowledge
(training, variants, families, tasks) and analysis-side knowledge
(canonical hook names + tensors). Concrete architecture subclasses
(``HookedTransformer``, ``HookedOneHotMLP``, ``HookedEmbeddingMLP`` —
landing under REQ_112 / REQ_113) implement ``setup_hooks()``,
``forward()``, ``weight_names()``, and ``get_weight()``. Analyzers
receive a ``HookedModel`` and an :class:`ActivationCache`; they read by
canonical name and never import a concrete subclass type.

Subclass contract
-----------------
Subclasses must:

1. Build their submodules in ``__init__``.
2. Call ``self.setup_hooks()`` at the **end** of ``__init__`` — after all
   submodules are constructed. This ordering is the subclass's
   responsibility; the base class does not auto-invoke ``setup_hooks()``
   because that would run before the subclass has finished building.
3. Override ``forward(inputs) -> Tensor`` (the standard ``nn.Module``
   contract). The forward method must route activations through
   ``self._hook_points[canonical_name]`` at every capture point so the
   default ``run_with_cache`` implementation can collect them.
4. Override ``weight_names()`` and ``get_weight(name)`` to publish
   learned parameters under canonical paths.
5. Override ``setup_hooks()`` to populate ``self._hook_points`` with the
   canonical-name → ``HookPoint`` mapping. Each ``HookPoint`` must be
   registered as a child module so PyTorch's hook machinery can find it
   (the helper ``_register_hook_point`` does both at once).

Why ``_hook_points`` is private
-------------------------------
The dict storing the canonical-name → ``HookPoint`` map is a private
implementation detail. The public surface is ``hook_names()``. This
matters specifically because :class:`~miscope.architectures.HookedTransformer`
multi-inherits from ``transformer_lens.HookedRootModule``, which already
defines a ``hook_points()`` method (returning a list, not a dict). A
public dict attribute on the same name would shadow the inherited
method and break TL's internal cleanup. Keeping the dict private avoids
the collision.

This module has no imports from the underlying TransformerLens
library. The grep quarantine test (REQ_112) depends on this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from miscope.architectures.activation_cache import ActivationCache
from miscope.architectures.hooks import HookPoint

# Hook function: (tensor, *, hook=hook_point) -> tensor or None.
# A ``None`` return leaves the tensor unchanged; returning a tensor
# replaces it for downstream consumers.
HookFunction = Callable[..., Any]
HookSpec = tuple[str, HookFunction]


class HookedModel(nn.Module, ABC):
    """Abstract base class for miscope-trained models.

    Provides the canonical-name surface (``hook_names``, ``weight_names``,
    ``get_weight``, ``run_with_cache``, ``run_with_hooks``) that
    analyzers consume. Concrete subclasses implement the architecture.
    """

    _hook_points: dict[str, HookPoint]

    def __init__(self) -> None:
        super().__init__()
        self._hook_points = {}

    # ------------------------------------------------------------------
    # Subclass responsibilities
    # ------------------------------------------------------------------

    @abstractmethod
    def setup_hooks(self) -> None:
        """Register canonical hook points on this model.

        Subclasses populate ``self._hook_points`` with the canonical-name
        → :class:`HookPoint` mapping appropriate to the architecture.
        Each ``HookPoint`` should also be registered as a child module
        (use :meth:`_register_hook_point`).

        Called by the subclass at the end of ``__init__``, after
        submodules are constructed. The base class does not auto-invoke
        — see module docstring for rationale.
        """

    @abstractmethod
    def weight_names(self) -> list[str]:
        """Return all canonical weight paths this model exposes."""

    @abstractmethod
    def get_weight(self, canonical_name: str) -> torch.Tensor:
        """Return a learned parameter by canonical name.

        Raises:
            KeyError: if ``canonical_name`` is not published by this
                model. The error message includes the names this model
                does publish.
        """

    # ------------------------------------------------------------------
    # Default surface
    # ------------------------------------------------------------------

    def hook_names(self) -> list[str]:
        """Return all canonical hook paths this model publishes."""
        return list(self._hook_points)

    def run_with_cache(
        self,
        inputs: torch.Tensor,
        fwd_hooks: list[HookSpec] | None = None,
    ) -> tuple[torch.Tensor, ActivationCache]:
        """Forward pass with full canonical-name activation capture.

        Returns ``(logits, cache)`` where ``cache`` is keyed by every
        canonical hook name in ``self._hook_points``. Caller-supplied
        ``fwd_hooks`` (canonical-name → callable) fire in addition to
        the capture hooks; their semantics match
        :meth:`run_with_hooks`.

        Subclasses may override (e.g., ``HookedTransformer`` delegates
        to TL's optimized ``run_with_cache``); the default implementation
        works for any subclass whose ``forward`` routes activations
        through ``self._hook_points[name]``.
        """
        cache = ActivationCache()
        handles = [_register_capture(hp, name, cache) for name, hp in self._hook_points.items()]
        if fwd_hooks:
            handles.extend(self._register_user_hooks(fwd_hooks))
        try:
            logits = self(inputs)
        finally:
            for h in handles:
                h.remove()
        return logits, cache

    def run_with_hooks(
        self,
        inputs: torch.Tensor,
        fwd_hooks: list[HookSpec],
    ) -> torch.Tensor:
        """Forward pass with caller-supplied canonical-name hooks.

        ``fwd_hooks`` is a list of ``(canonical_name, callable)`` pairs.
        Each callable receives ``(tensor, *, hook=hook_point)`` and may
        return a modified tensor or ``None`` to leave the activation
        unchanged.
        """
        handles = self._register_user_hooks(fwd_hooks)
        try:
            return self(inputs)
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------
    # Subclass helpers
    # ------------------------------------------------------------------

    def _register_hook_point(self, canonical_name: str) -> HookPoint:
        """Create a ``HookPoint``, register it, and return it.

        The point is registered both in ``self._hook_points`` (the
        canonical-name lookup) and as a child ``nn.Module`` (so
        PyTorch's hook plumbing can find it). Returns the ``HookPoint``
        for the subclass to wire into its forward path.
        """
        if canonical_name in self._hook_points:
            raise ValueError(f"Hook point {canonical_name!r} is already registered.")
        point = HookPoint()
        point.name = canonical_name
        self._hook_points[canonical_name] = point
        # nn.Module child names cannot contain dots; use underscores.
        attr_name = "_hp__" + canonical_name.replace(".", "__")
        self.add_module(attr_name, point)
        return point

    def _register_user_hooks(
        self,
        fwd_hooks: list[HookSpec],
    ) -> list[Any]:
        """Register caller-supplied hooks, returning the removal handles.

        The user-supplied callable has the TL-style signature
        ``fn(tensor, *, hook=hook_point) -> tensor or None``. We adapt
        it to PyTorch's ``(module, inputs, output) -> output | None``
        contract so we own the ``RemovableHandle`` and can clean up
        deterministically without disturbing any other hooks the user
        may have registered.
        """
        handles: list[Any] = []
        for name, fn in fwd_hooks:
            point = self._hook_points.get(name)
            if point is None:
                for h in handles:
                    h.remove()
                raise KeyError(
                    f"Cannot register hook on {name!r}: not published by "
                    f"this model. Available: {sorted(self._hook_points)}"
                )
            handles.append(_register_user_forward_hook(point, fn))
        return handles


def _register_capture(
    hook_point: HookPoint,
    canonical_name: str,
    cache: ActivationCache,
) -> Any:
    """Register a forward hook that captures activations into ``cache``.

    Returns a ``RemovableHandle`` so the caller controls lifetime.
    """

    def _capture(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        cache[canonical_name] = output

    return hook_point.register_forward_hook(_capture)


def _register_user_forward_hook(
    hook_point: HookPoint,
    fn: HookFunction,
) -> Any:
    """Adapt a TL-style hook function to PyTorch's forward-hook signature."""

    def _adapter(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> Any:
        return fn(output, hook=hook_point)

    return hook_point.register_forward_hook(_adapter)
