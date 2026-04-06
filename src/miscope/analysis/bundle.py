"""TransformerLensBundle — ActivationBundle implementation for TransformerLens models.

Wraps a (HookedTransformer, ActivationCache, logits) triple into the
ActivationBundle protocol so analyzers can access activations and weights
without importing TransformerLens types directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer
    from transformer_lens.ActivationCache import ActivationCache

_WEIGHT_LOOKUP: dict[str, Any] = {
    "W_E":   lambda m: m.embed.W_E,
    "W_pos": lambda m: m.pos_embed.W_pos,
    "W_Q":   lambda m: m.blocks[0].attn.W_Q,
    "W_K":   lambda m: m.blocks[0].attn.W_K,
    "W_V":   lambda m: m.blocks[0].attn.W_V,
    "W_O":   lambda m: m.blocks[0].attn.W_O,
    "W_in":  lambda m: m.blocks[0].mlp.W_in,
    "W_out": lambda m: m.blocks[0].mlp.W_out,
    "W_U":   lambda m: m.unembed.W_U,
}


class TransformerLensBundle:
    """Zero-cost wrapper over (HookedTransformer, ActivationCache, logits).

    Delegates all calls directly to the underlying TL objects — no tensor
    copies, no re-computation.

    Args:
        model: HookedTransformer with checkpoint weights loaded.
        cache: ActivationCache from model.run_with_cache(probe).
        logits: Raw logits tensor (batch, seq_len, vocab) from the same pass.
    """

    def __init__(
        self,
        model: HookedTransformer,
        cache: ActivationCache,
        logits: torch.Tensor,
    ) -> None:
        self._model = model
        self._cache = cache
        self._logits = logits

    # ------------------------------------------------------------------
    # ActivationBundle protocol
    # ------------------------------------------------------------------

    def mlp_post(self, layer: int, position: int) -> torch.Tensor:
        """Post-activation MLP neurons. Returns (batch, d_mlp)."""
        acts = self._cache["post", layer, "mlp"]  # (batch, seq, d_mlp)
        return acts[:, position, :]

    def residual_stream(self, layer: int, position: int, location: str) -> torch.Tensor:
        """Residual stream at a site. Returns (batch, d_model)."""
        acts = self._cache[location, layer]  # (batch, seq, d_model)
        return acts[:, position, :]

    def attention_pattern(self, layer: int) -> torch.Tensor:
        """Attention weights (post-softmax). Returns (batch, n_heads, seq_to, seq_from)."""
        return self._cache["pattern", layer]

    def weight(self, name: str) -> torch.Tensor:
        """Named weight matrix. Raises KeyError for unknown names."""
        lookup = _WEIGHT_LOOKUP.get(name)
        if lookup is None:
            raise KeyError(f"Unknown weight name: {name!r}. Expected one of {list(_WEIGHT_LOOKUP)}")
        return lookup(self._model)

    def logits(self, position: int) -> torch.Tensor:
        """Logits at a token position. Returns (batch, vocab_size)."""
        return self._logits[:, position, :]

    # ------------------------------------------------------------------
    # Escape hatch for analyzers requiring direct model access
    # (e.g. LandscapeFlatnessAnalyzer which perturbs model parameters)
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> HookedTransformer:
        """Direct access to the underlying model. Use only when the bundle
        protocol is insufficient (e.g., parameter perturbation)."""
        return self._model
