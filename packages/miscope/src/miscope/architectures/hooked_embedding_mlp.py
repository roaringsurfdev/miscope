"""HookedEmbeddingMLP — two-layer MLP with separate learned embeddings.

Architecture (matches the original ``ModuloAdditionEmbedMLP``):

    a, b → embed_a(a) + embed_b(b) → Linear(d_embed, d_hidden) → ReLU
        → Linear(d_hidden, p) → logits

State-dict compatible with checkpoints saved by ``ModuloAdditionEmbedMLP``;
submodule names are preserved (``embed_a``, ``embed_b``, ``W_in``,
``W_out``).

Embedding-identity invariant
----------------------------
The two per-input embedding matrices are published as
``embed.embed_a`` and ``embed.embed_b`` — *not* as ``embed.W_E``. The
distinction is load-bearing: an analyzer that asks for ``embed.W_E``
expects a transformer's shared embedding matrix, and silently aliasing
to one of the per-input matrices (or their concatenation, or their sum)
would produce wrong output without surfacing the category error. The
boundary is enforced via ``KeyError`` in ``get_weight``.

This subclass has no TransformerLens involvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from miscope.architectures.hooked_model import HookedModel
from miscope.core import architecture as canonical_hooks
from miscope.core import weights as canonical_weights

__all__ = ["HookedEmbeddingMLP", "HookedEmbeddingMLPConfig"]


@dataclass
class HookedEmbeddingMLPConfig:
    """Construction config for ``HookedEmbeddingMLP``.

    Attributes:
        vocab_size: Number of input classes ``p`` (also output dim).
        d_embed: Embedding dimension.
        d_hidden: Hidden-layer width.
        seed: Optional torch RNG seed for reproducible init.
    """

    vocab_size: int
    d_embed: int
    d_hidden: int
    seed: int | None = None


# Canonical hook paths.
HOOK_EMBED_OUT = f"{canonical_hooks.EMBED}.{canonical_hooks.HOOK_OUT}"
HOOK_BLOCK_IN = canonical_hooks.hook(canonical_hooks.BLOCKS, 0, canonical_hooks.HOOK_IN)
HOOK_BLOCK_OUT = canonical_hooks.hook(canonical_hooks.BLOCKS, 0, canonical_hooks.HOOK_OUT)
HOOK_MLP_PRE = canonical_hooks.hook(
    canonical_hooks.BLOCKS, 0, canonical_hooks.MLP, canonical_hooks.HOOK_PRE
)
HOOK_MLP_OUT = canonical_hooks.hook(
    canonical_hooks.BLOCKS, 0, canonical_hooks.MLP, canonical_hooks.HOOK_OUT
)
HOOK_UNEMBED_IN = f"{canonical_hooks.UNEMBED}.{canonical_hooks.HOOK_IN}"
HOOK_UNEMBED_OUT = f"{canonical_hooks.UNEMBED}.{canonical_hooks.HOOK_OUT}"

# Canonical weight paths. Note: ``embed.W_E`` is intentionally absent —
# see module docstring for the embedding-identity invariant.
WEIGHT_EMBED_A = canonical_weights.EMBED_A_PATH
WEIGHT_EMBED_B = canonical_weights.EMBED_B_PATH
WEIGHT_MLP_IN = canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_IN)
WEIGHT_MLP_OUT = canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_OUT)


class HookedEmbeddingMLP(HookedModel):
    """Canonical-name surface over the learned-embedding 2L MLP architecture."""

    def __init__(self, cfg: HookedEmbeddingMLPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        self.embed_a = nn.Embedding(cfg.vocab_size, cfg.d_embed)
        self.embed_b = nn.Embedding(cfg.vocab_size, cfg.d_embed)
        self.W_in = nn.Linear(cfg.d_embed, cfg.d_hidden, bias=False)
        self.relu = nn.ReLU()
        self.W_out = nn.Linear(cfg.d_hidden, cfg.vocab_size, bias=False)

        self.setup_hooks()

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size

    @property
    def d_embed(self) -> int:
        return self.cfg.d_embed

    @property
    def d_hidden(self) -> int:
        return self.cfg.d_hidden

    # ------------------------------------------------------------------
    # HookedModel surface
    # ------------------------------------------------------------------

    def setup_hooks(self) -> None:
        self._hp_embed_out = self._register_hook_point(HOOK_EMBED_OUT)
        self._hp_block_in = self._register_hook_point(HOOK_BLOCK_IN)
        self._hp_mlp_pre = self._register_hook_point(HOOK_MLP_PRE)
        self._hp_mlp_out = self._register_hook_point(HOOK_MLP_OUT)
        self._hp_block_out = self._register_hook_point(HOOK_BLOCK_OUT)
        self._hp_unembed_in = self._register_hook_point(HOOK_UNEMBED_IN)
        self._hp_unembed_out = self._register_hook_point(HOOK_UNEMBED_OUT)

    def weight_names(self) -> list[str]:
        return [WEIGHT_EMBED_A, WEIGHT_EMBED_B, WEIGHT_MLP_IN, WEIGHT_MLP_OUT]

    def get_weight(self, canonical_name: str) -> torch.Tensor:
        if canonical_name == WEIGHT_EMBED_A:
            return self.embed_a.weight
        if canonical_name == WEIGHT_EMBED_B:
            return self.embed_b.weight
        if canonical_name == WEIGHT_MLP_IN:
            return self.W_in.weight
        if canonical_name == WEIGHT_MLP_OUT:
            return self.W_out.weight
        raise KeyError(
            f"Unknown canonical weight name {canonical_name!r}. "
            f"Available: {self.weight_names()}. "
            f"Note: this MLP architecture publishes per-input embeddings as "
            f"'embed.embed_a' and 'embed.embed_b' — never 'embed.W_E', "
            f"which is reserved for transformer-class architectures with a "
            f"single shared embedding."
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            inputs: Long tensor of shape ``(batch, 2)`` where columns are
                ``a`` and ``b`` indices, or two columns of any tensor that
                supports ``.unbind(1)``.

        Returns:
            Logits of shape ``(batch, vocab_size)``.
        """
        a, b = inputs.unbind(1)
        embed_sum = self._hp_embed_out(self.embed_a(a) + self.embed_b(b))
        block_in = self._hp_block_in(embed_sum)
        pre = self._hp_mlp_pre(self.W_in(block_in))
        post = self._hp_mlp_out(self.relu(pre))
        block_out = self._hp_block_out(post)
        unembed_in = self._hp_unembed_in(block_out)
        logits = self.W_out(unembed_in)
        return self._hp_unembed_out(logits)
