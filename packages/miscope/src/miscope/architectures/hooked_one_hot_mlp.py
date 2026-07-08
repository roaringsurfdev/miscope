"""HookedOneHotMLP — two-layer MLP with one-hot input (no embedding, no attention).

Architecture (matches the original ``ModuloAddition2LMLP``):

    one-hot(a, b) → Linear(2p, d_hidden) → ReLU → Linear(d_hidden, p) → logits

State-dict compatible with checkpoints saved by ``ModuloAddition2LMLP``;
submodule names are preserved (``W_in``, ``W_out``).

This subclass has no TransformerLens involvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from miscope.architectures.hooked_model import HookedModel
from miscope.core import architecture as canonical_hooks
from miscope.core import weights as canonical_weights

__all__ = ["HookedOneHotMLP", "HookedOneHotMLPConfig"]


@dataclass
class HookedOneHotMLPConfig:
    """Construction config for ``HookedOneHotMLP``.

    Attributes:
        vocab_size: Number of input classes ``p``. Input width is ``2 * p``.
        d_hidden: Hidden-layer width.
        seed: Optional torch RNG seed for reproducible init. ``None``
            inherits the global RNG state.
    """

    vocab_size: int
    d_hidden: int
    seed: int | None = None


# Canonical hook paths this architecture publishes. Module-level constants
# keep the names auditable and let analyzer ``required_hooks`` declarations
# reference them programmatically.
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

# Canonical weight paths this architecture publishes.
WEIGHT_MLP_IN = canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_IN)
WEIGHT_MLP_OUT = canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_OUT)


class HookedOneHotMLP(HookedModel):
    """Canonical-name surface over the one-hot 2L MLP architecture.

    Forbidden under the embedding-identity invariant (REQ_113):
    ``embed.W_E`` and ``embed.hook_out`` are *not* published. An
    analyzer that asks for them gets a ``KeyError`` listing the model's
    actual surface.
    """

    def __init__(self, cfg: HookedOneHotMLPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        self.W_in = nn.Linear(2 * cfg.vocab_size, cfg.d_hidden, bias=False)
        self.relu = nn.ReLU()
        self.W_out = nn.Linear(cfg.d_hidden, cfg.vocab_size, bias=False)

        self.setup_hooks()

    # Convenience accessors so external callers (e.g., training scripts
    # that read model.vocab_size) keep working without reaching into cfg.
    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size

    @property
    def d_hidden(self) -> int:
        return self.cfg.d_hidden

    # ------------------------------------------------------------------
    # HookedModel surface
    # ------------------------------------------------------------------

    def setup_hooks(self) -> None:
        self._hp_block_in = self._register_hook_point(HOOK_BLOCK_IN)
        self._hp_mlp_pre = self._register_hook_point(HOOK_MLP_PRE)
        self._hp_mlp_out = self._register_hook_point(HOOK_MLP_OUT)
        # block_out and unembed_in coincide with mlp_out for this no-residual
        # architecture, but distinct hook points let analyzers declare the
        # site they care about (block-level vs. component-level).
        self._hp_block_out = self._register_hook_point(HOOK_BLOCK_OUT)
        self._hp_unembed_in = self._register_hook_point(HOOK_UNEMBED_IN)
        self._hp_unembed_out = self._register_hook_point(HOOK_UNEMBED_OUT)

    def weight_names(self) -> list[str]:
        return [WEIGHT_MLP_IN, WEIGHT_MLP_OUT]

    def get_weight(self, canonical_name: str) -> torch.Tensor:
        if canonical_name == WEIGHT_MLP_IN:
            return self.W_in.weight
        if canonical_name == WEIGHT_MLP_OUT:
            return self.W_out.weight
        raise KeyError(
            f"Unknown canonical weight name {canonical_name!r}. Available: {self.weight_names()}"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            inputs: One-hot encoded input of shape ``(batch, 2 * vocab_size)``.

        Returns:
            Logits of shape ``(batch, vocab_size)``.
        """
        x = self._hp_block_in(inputs)
        pre = self._hp_mlp_pre(self.W_in(x))
        post = self._hp_mlp_out(self.relu(pre))
        block_out = self._hp_block_out(post)
        unembed_in = self._hp_unembed_in(block_out)
        logits = self.W_out(unembed_in)
        return self._hp_unembed_out(logits)
