"""Smoke tests for the HookedModel abstract surface (REQ_105).

Verifies that the boundary is well-formed:

- Construction succeeds; ``setup_hooks`` populates ``hook_points``.
- ``hook_names()`` returns the published names.
- ``get_weight()`` returns published weights and raises ``KeyError`` with
  a helpful message for unknown names.
- ``run_with_cache()`` returns ``(logits, cache)`` populated by canonical
  name; ``cache[unknown_name]`` raises ``KeyError`` with available keys
  listed.
- ``run_with_hooks()`` invokes caller-supplied hooks at the named hook
  point.
- Substitution: a downstream consumer that reads the cache by canonical
  name works against any ``HookedModel`` subclass without architecture
  branching.

This module imports nothing from ``transformer_lens`` — the analyzer-side
surface (cache + base class) must stay TL-free.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from miscope.architectures import ActivationCache, HookedModel, HookPoint
from miscope.core import architecture as arch_names

# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------

CANARY_HOOK = arch_names.hook(arch_names.BLOCKS, 0, arch_names.MLP, arch_names.HOOK_OUT)
CANARY_WEIGHT = "blocks.0.mlp.in.W"


class StubHookedModel(HookedModel):
    """Minimal HookedModel: one linear layer, one canonical hook point."""

    def __init__(self, d_in: int = 4, d_out: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        self.setup_hooks()

    def setup_hooks(self) -> None:
        self._mlp_out = self._register_hook_point(CANARY_HOOK)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        return self._mlp_out(x)

    def weight_names(self) -> list[str]:
        return [CANARY_WEIGHT]

    def get_weight(self, canonical_name: str) -> torch.Tensor:
        if canonical_name == CANARY_WEIGHT:
            return self.linear.weight
        raise KeyError(
            f"Unknown canonical weight name {canonical_name!r}. Available: {self.weight_names()}"
        )


# ---------------------------------------------------------------------------
# Construction + introspection
# ---------------------------------------------------------------------------


def test_construction_publishes_hook():
    model = StubHookedModel()
    assert CANARY_HOOK in model.hook_names()
    assert isinstance(model._hook_points[CANARY_HOOK], HookPoint)


def test_hook_names_returns_published_list():
    model = StubHookedModel()
    assert model.hook_names() == [CANARY_HOOK]


def test_canonical_hook_name_passes_grammar():
    assert arch_names.is_canonical_hook_name(CANARY_HOOK)


def test_weight_names_returns_published_list():
    model = StubHookedModel()
    assert model.weight_names() == [CANARY_WEIGHT]


# ---------------------------------------------------------------------------
# get_weight
# ---------------------------------------------------------------------------


def test_get_weight_returns_tensor():
    model = StubHookedModel(d_in=3, d_out=5)
    w = model.get_weight(CANARY_WEIGHT)
    assert isinstance(w, torch.Tensor)
    assert w.shape == (5, 3)
    assert w.data_ptr() == model.linear.weight.data_ptr()  # zero-copy


def test_get_weight_unknown_name_raises_keyerror_with_available():
    model = StubHookedModel()
    with pytest.raises(KeyError) as excinfo:
        model.get_weight("embed.W_E")
    msg = str(excinfo.value)
    assert "embed.W_E" in msg
    assert CANARY_WEIGHT in msg


# ---------------------------------------------------------------------------
# run_with_cache
# ---------------------------------------------------------------------------


def test_run_with_cache_returns_logits_and_cache():
    model = StubHookedModel(d_in=4, d_out=4)
    inputs = torch.randn(2, 4)
    logits, cache = model.run_with_cache(inputs)
    assert logits.shape == (2, 4)
    assert isinstance(cache, ActivationCache)
    assert CANARY_HOOK in cache
    assert torch.allclose(cache[CANARY_HOOK], logits)


def test_run_with_cache_unknown_key_raises_keyerror():
    model = StubHookedModel()
    _, cache = model.run_with_cache(torch.randn(1, 4))
    with pytest.raises(KeyError) as excinfo:
        cache["embed.hook_out"]
    assert "embed.hook_out" in str(excinfo.value)
    assert CANARY_HOOK in str(excinfo.value)


def test_run_with_cache_cleans_up_capture_hooks():
    model = StubHookedModel()
    model.run_with_cache(torch.randn(1, 4))
    # The capture hooks registered via register_forward_hook live on the
    # nn.Module's _forward_hooks dict; they must be removed after the
    # call so subsequent forwards don't accumulate captures.
    point = model._hook_points[CANARY_HOOK]
    assert len(point._forward_hooks) == 0


# ---------------------------------------------------------------------------
# run_with_hooks
# ---------------------------------------------------------------------------


def test_run_with_hooks_invokes_user_hook():
    model = StubHookedModel()
    seen: list[torch.Tensor] = []

    def my_hook(tensor: torch.Tensor, *, hook: HookPoint) -> None:
        seen.append(tensor.clone())

    model.run_with_hooks(torch.randn(2, 4), [(CANARY_HOOK, my_hook)])
    assert len(seen) == 1
    assert seen[0].shape == (2, 4)


def test_run_with_hooks_can_replace_activation():
    model = StubHookedModel()
    replacement = torch.zeros(2, 4)

    def replace(tensor: torch.Tensor, *, hook: HookPoint) -> torch.Tensor:
        return replacement

    out = model.run_with_hooks(torch.randn(2, 4), [(CANARY_HOOK, replace)])
    assert torch.equal(out, replacement)


def test_run_with_hooks_unknown_name_raises_keyerror():
    model = StubHookedModel()
    with pytest.raises(KeyError) as excinfo:
        model.run_with_hooks(
            torch.randn(1, 4),
            [("embed.hook_out", lambda t, *, hook: None)],
        )
    assert "embed.hook_out" in str(excinfo.value)


def test_run_with_hooks_cleans_up_after_call():
    model = StubHookedModel()
    model.run_with_hooks(
        torch.randn(1, 4),
        [(CANARY_HOOK, lambda t, *, hook: None)],
    )
    point = model._hook_points[CANARY_HOOK]
    assert len(point._forward_hooks) == 0


# ---------------------------------------------------------------------------
# Architecture-agnostic substitution
# ---------------------------------------------------------------------------


def architecture_agnostic_consumer(model: HookedModel) -> torch.Tensor:
    """A toy 'analyzer' that reads by canonical name only."""
    in_features = 4
    if hasattr(model, "linear"):
        linear: nn.Linear = model.linear  # type: ignore[attr-defined,assignment]
        in_features = linear.in_features
    inputs = torch.randn(3, in_features)
    _, cache = model.run_with_cache(inputs)
    return cache[CANARY_HOOK]


def test_consumer_works_against_stub():
    """A consumer that reads cache[canonical_name] runs against any
    HookedModel — no architecture branching needed.
    """
    model = StubHookedModel()
    out = architecture_agnostic_consumer(model)
    assert out.shape == (3, 4)


def test_subclass_must_implement_abstract_methods():
    """Instantiating a HookedModel without overriding abstract methods fails."""

    class Incomplete(HookedModel):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_duplicate_hook_registration_raises():
    """A subclass that tries to register the same name twice fails fast."""

    class Duplicate(HookedModel):
        def __init__(self) -> None:
            super().__init__()
            self.setup_hooks()

        def setup_hooks(self) -> None:
            self._register_hook_point(CANARY_HOOK)
            self._register_hook_point(CANARY_HOOK)

        def weight_names(self) -> list[str]:
            return []

        def get_weight(self, canonical_name: str) -> torch.Tensor:
            raise KeyError(canonical_name)

    with pytest.raises(ValueError, match="already registered"):
        Duplicate()
