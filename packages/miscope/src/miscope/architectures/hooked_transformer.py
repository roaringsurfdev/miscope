"""miscope.HookedTransformer — TransformerLens quarantine.

This is the **only** module in ``src/miscope/`` allowed to import from
``transformer_lens`` (excluding the deliberate alias in
``architectures/hooks.py``). Everything else in the platform — analyzers,
families, pipeline, dashboard — sees the canonical
:class:`~miscope.architectures.HookedModel` interface and never the
underlying TL types.

Implementation strategy
-----------------------
Multiple inheritance: ``class HookedTransformer(MiHookedModel,
TLHookedTransformer)``. The platform-side interface (``MiHookedModel``)
is structurally a refinement of the TL interface; both inherit from
``nn.Module``. This lets ``HookedTransformer`` satisfy
``isinstance(model, HookedModel)`` checks while reusing TL's training-
ready architecture (parameters, state dicts, optimizer integration,
checkpoint loading) unchanged.

``__init__`` is **not** cooperative — ``MiHookedModel.__init__`` takes
no positional args, ``TLHookedTransformer.__init__`` takes a config, so
they cannot share a ``super()`` chain. Instead we call TL's init
directly, then manually establish the MiHookedModel invariants
(``self._hook_points = {}``; ``setup_hooks()``).

Translation table
-----------------
:data:`_canonical_to_tl_hook` and :data:`_canonical_to_tl_weight`
encapsulate **every** point where TL's spelling is allowed to leak.
They are the load-bearing audit surface for the quarantine. Everything
outside these tables is canonical.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from transformer_lens import HookedTransformer as TLHookedTransformer
from transformer_lens import HookedTransformerConfig as TLHookedTransformerConfig

from miscope.architectures.activation_cache import ActivationCache
from miscope.architectures.hooked_model import HookedModel
from miscope.architectures.hooks import HookPoint
from miscope.core import architecture as canonical_hooks
from miscope.core import weights as canonical_weights

__all__ = ["HookedTransformer", "HookedTransformerConfig"]


# Re-export TL's config dataclass under our canonical surface so callers
# (e.g., :class:`ModuloAddition1LayerFamily`) construct configs without
# reaching into ``transformer_lens`` directly. Keeping the alias here
# means the quarantine module stays the single TL entry point.
HookedTransformerConfig = TLHookedTransformerConfig


def _canonical_to_tl_hook_table(cfg: TLHookedTransformerConfig) -> dict[str, str]:
    """Build the canonical → TL hook-name translation table for ``cfg``.

    The table covers every canonical hook the model publishes for this
    config. TL hooks that don't fit cleanly into a canonical name (e.g.,
    ``hook_resid_mid`` in attn-only configs, the ``_input`` hooks gated
    on ``use_split_qkv_input``) are intentionally omitted — adding them
    requires a canonical name to point them at.
    """
    table: dict[str, str] = {
        # Top-level activations
        f"{canonical_hooks.EMBED}.{canonical_hooks.HOOK_OUT}": "hook_embed",
        f"{canonical_hooks.POS_EMBED}.{canonical_hooks.HOOK_OUT}": "hook_pos_embed",
    }
    for layer in range(cfg.n_layers):
        block = canonical_hooks.block(layer)
        # Block-level residual stream: hook_in (entering block) ≡ hook_resid_pre;
        # hook_out (leaving block) ≡ hook_resid_post.
        table[f"{block}.{canonical_hooks.HOOK_IN}"] = f"{block}.hook_resid_pre"
        table[f"{block}.{canonical_hooks.HOOK_OUT}"] = f"{block}.hook_resid_post"

        # Attention component
        if not cfg.attn_only or cfg.attn_only:  # always present in HookedTransformer
            attn = canonical_hooks.block_component(layer, canonical_hooks.ATTN)
            # The block's hook_attn_out is the residual contribution from
            # attention — semantically "what attention emits into the
            # residual stream", which matches both attn.hook_out (component
            # output) and attn.o.hook_out (output of the O projection).
            attn_out_tl = f"{block}.hook_attn_out"
            table[f"{attn}.{canonical_hooks.HOOK_OUT}"] = attn_out_tl
            table[f"{attn}.{canonical_hooks.ATTN_O}.{canonical_hooks.HOOK_OUT}"] = attn_out_tl
            # Per-subcomponent activations
            table[f"{attn}.{canonical_hooks.ATTN_Q}.{canonical_hooks.HOOK_OUT}"] = (
                f"{block}.attn.hook_q"
            )
            table[f"{attn}.{canonical_hooks.ATTN_K}.{canonical_hooks.HOOK_OUT}"] = (
                f"{block}.attn.hook_k"
            )
            table[f"{attn}.{canonical_hooks.ATTN_V}.{canonical_hooks.HOOK_OUT}"] = (
                f"{block}.attn.hook_v"
            )
            # Attention pattern + raw scores
            table[f"{attn}.{canonical_hooks.HOOK_PATTERN}"] = f"{block}.attn.hook_pattern"
            table[f"{attn}.{canonical_hooks.HOOK_ATTN_SCORES}"] = f"{block}.attn.hook_attn_scores"

        # MLP component
        if not cfg.attn_only:
            mlp = canonical_hooks.block_component(layer, canonical_hooks.MLP)
            table[f"{mlp}.{canonical_hooks.HOOK_PRE}"] = f"{block}.mlp.hook_pre"
            # mlp.hook_out (post-activation neurons) ≡ TL's mlp.hook_post
            table[f"{mlp}.{canonical_hooks.HOOK_OUT}"] = f"{block}.mlp.hook_post"

    return table


def _canonical_to_tl_weight_table(cfg: TLHookedTransformerConfig) -> dict[str, str]:
    """Build the canonical → TL attribute-path translation table for weights.

    Each value is a dotted attribute path from the model root, suitable
    for resolution via :func:`_resolve_attr`. Bias paths are included
    even for configs that don't use biases — :meth:`HookedTransformer.get_weight`
    raises ``KeyError`` cleanly when the underlying tensor is None or
    absent.
    """
    table: dict[str, str] = {
        canonical_weights.EMBED_WEIGHT: "embed.W_E",
        canonical_weights.EMBED_BIAS: "embed.b_E",
        canonical_weights.POS_EMBED_WEIGHT: "pos_embed.W_pos",
        canonical_weights.POS_EMBED_BIAS: "pos_embed.b_pos",
        canonical_weights.UNEMBED_WEIGHT: "unembed.W_U",
        canonical_weights.UNEMBED_BIAS: "unembed.b_U",
    }
    for layer in range(cfg.n_layers):
        # Attention weights — TL uses uppercase Q/K/V/O; canonical uses lowercase q/k/v/o.
        for canonical_sub, tl_sub in (
            (canonical_hooks.ATTN_Q, "Q"),
            (canonical_hooks.ATTN_K, "K"),
            (canonical_hooks.ATTN_V, "V"),
            (canonical_hooks.ATTN_O, "O"),
        ):
            table[canonical_weights.block_attn_weight(layer, canonical_sub)] = (
                f"blocks.{layer}.attn.W_{tl_sub}"
            )
            table[canonical_weights.block_attn_bias(layer, canonical_sub)] = (
                f"blocks.{layer}.attn.b_{tl_sub}"
            )
        # MLP weights — TL uses W_in / W_out / b_in / b_out.
        table[canonical_weights.block_mlp_weight(layer, canonical_hooks.MLP_IN)] = (
            f"blocks.{layer}.mlp.W_in"
        )
        table[canonical_weights.block_mlp_bias(layer, canonical_hooks.MLP_IN)] = (
            f"blocks.{layer}.mlp.b_in"
        )
        table[canonical_weights.block_mlp_weight(layer, canonical_hooks.MLP_OUT)] = (
            f"blocks.{layer}.mlp.W_out"
        )
        table[canonical_weights.block_mlp_bias(layer, canonical_hooks.MLP_OUT)] = (
            f"blocks.{layer}.mlp.b_out"
        )
    return table


def _resolve_attr(root: Any, dotted_path: str) -> Any:
    """Resolve ``"a.b.0.c"`` against ``root`` via getattr / __getitem__."""
    obj = root
    for part in dotted_path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class HookedTransformer(HookedModel, TLHookedTransformer):
    """Canonical-name surface over a TransformerLens HookedTransformer.

    Subclasses both :class:`miscope.architectures.HookedModel` (the
    platform interface) and :class:`transformer_lens.HookedTransformer`
    (the underlying training-ready architecture). Reuses TL's parameter
    layout, state-dict format, and forward pass while exposing the
    canonical names the rest of miscope reads from.
    """

    def __init__(
        self,
        cfg: TLHookedTransformerConfig,
        tokenizer: Any | None = None,
        move_to_device: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
    ) -> None:
        # Non-cooperative init: MiHookedModel takes no args, TL takes cfg —
        # they cannot share a super() chain. Run TL's init explicitly,
        # then establish MiHookedModel invariants manually.
        TLHookedTransformer.__init__(
            self,
            cfg,
            tokenizer=tokenizer,
            move_to_device=move_to_device,
            default_padding_side=default_padding_side,
        )
        # Lookup tables. Built per-config because the table depends on
        # n_layers and cfg.attn_only.
        self._canonical_to_tl_hook = _canonical_to_tl_hook_table(cfg)
        self._tl_to_canonical_hook = {tl: c for c, tl in self._canonical_to_tl_hook.items()}
        self._canonical_to_tl_weight = _canonical_to_tl_weight_table(cfg)

        self._hook_points: dict[str, HookPoint] = {}
        self.setup_hooks()

    # ------------------------------------------------------------------
    # MiHookedModel surface
    # ------------------------------------------------------------------

    def setup_hooks(self) -> None:
        """Bind canonical names to the TL HookPoints already in the model.

        TL's ``HookedTransformer`` constructs its own ``HookPoint``
        instances during its ``__init__``. We don't create new ones —
        we look up the existing ones by TL's name and register them
        under canonical names in ``self._hook_points``.
        """
        # ``hook_dict`` is populated by TL's ``setup()``, which is called
        # at the end of TL's ``__init__``.
        for canonical_name, tl_name in self._canonical_to_tl_hook.items():
            tl_hook = self.hook_dict.get(tl_name)
            if tl_hook is None:
                # Skip canonical names whose underlying TL hook isn't
                # published by this config (e.g., MLP hooks on attn_only).
                continue
            self._hook_points[canonical_name] = tl_hook

    def weight_names(self) -> list[str]:
        """Return canonical paths for weights this model exposes.

        Filters the static table by checking which underlying tensors
        actually exist (some bias tensors are absent when the config
        disables biases or layernorm).
        """
        names = []
        for canonical, tl_path in self._canonical_to_tl_weight.items():
            try:
                tensor = _resolve_attr(self, tl_path)
            except AttributeError:
                continue
            if isinstance(tensor, torch.Tensor):
                names.append(canonical)
        return names

    def get_weight(self, canonical_name: str) -> torch.Tensor:
        """Return a learned parameter by canonical name."""
        tl_path = self._canonical_to_tl_weight.get(canonical_name)
        if tl_path is None:
            raise KeyError(
                f"Unknown canonical weight name {canonical_name!r}. "
                f"Available: {sorted(self.weight_names())}"
            )
        try:
            tensor = _resolve_attr(self, tl_path)
        except AttributeError:
            raise KeyError(
                f"Canonical weight {canonical_name!r} maps to {tl_path!r} but "
                f"that attribute is absent on this model. "
                f"Available: {sorted(self.weight_names())}"
            ) from None
        if not isinstance(tensor, torch.Tensor):
            raise KeyError(
                f"Canonical weight {canonical_name!r} resolved to a non-tensor "
                f"({type(tensor).__name__}). "
                f"Available: {sorted(self.weight_names())}"
            )
        return tensor

    def run_with_cache(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        fwd_hooks: list[tuple[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, ActivationCache]:
        """Forward pass with canonical-name activation capture.

        Delegates to TL's ``run_with_cache`` (single forward pass, full
        activation capture via TL's optimized machinery), then translates
        the resulting cache keys to canonical names. Tensors are passed
        by reference — no copies.

        Caller-supplied ``fwd_hooks`` use canonical hook names; they are
        translated to TL names for delegation.
        """
        if fwd_hooks:
            tl_fwd_hooks = [(self._canonical_to_tl_hook[name], fn) for name, fn in fwd_hooks]
            logits, tl_cache = TLHookedTransformer.run_with_cache(
                self,
                inputs,
                fwd_hooks=tl_fwd_hooks,  # type: ignore[arg-type]
            )
        else:
            logits, tl_cache = TLHookedTransformer.run_with_cache(self, inputs)

        canonical_cache = ActivationCache()
        for canonical, tl_name in self._canonical_to_tl_hook.items():
            if tl_name in tl_cache:
                canonical_cache[canonical] = tl_cache[tl_name]
        return logits, canonical_cache  # type: ignore[return-value]

    def run_with_hooks(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        fwd_hooks: list[tuple[str, Any]],
    ) -> torch.Tensor:
        """Forward pass with caller-supplied canonical-name hooks.

        Translates canonical hook names to TL names and delegates to
        TL's ``run_with_hooks``. The translation is the only point where
        the caller's canonical-name spelling meets TL's legacy spelling.
        """
        tl_fwd_hooks = [(self._canonical_to_tl_hook[name], fn) for name, fn in fwd_hooks]
        return TLHookedTransformer.run_with_hooks(
            self,
            inputs,
            fwd_hooks=tl_fwd_hooks,  # type: ignore[arg-type]
        )
