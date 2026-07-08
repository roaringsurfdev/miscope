"""Activation extraction and manipulation utilities.

Helpers for reading common activation sites from a canonical-name
:class:`miscope.architectures.ActivationCache`. Architecture-agnostic
where possible: ``extract_mlp_activations`` works against both
transformer caches (with a sequence dimension) and MLP caches (without).

This module has no imports from the underlying TransformerLens
library. The grep quarantine test depends on this.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import einops
import torch

from miscope.core import architecture as canonical_hooks
from miscope.core import weights as canonical_weights

if TYPE_CHECKING:
    from miscope.architectures import ActivationCache, HookedModel


# Map legacy residual-stream location names → canonical hook paths.
# Used by ``extract_residual_stream`` so callers don't need to know the
# canonical-name vocabulary.
_RESID_LOCATION_TO_CANONICAL_SUFFIX: dict[str, str] = {
    "resid_pre": canonical_hooks.HOOK_IN,
    "resid_post": canonical_hooks.HOOK_OUT,
    "attn_out": f"{canonical_hooks.ATTN}.{canonical_hooks.HOOK_OUT}",
}


def extract_mlp_activations(
    cache: ActivationCache,
    layer: int = 0,
    position: int = -1,
) -> torch.Tensor:
    """Extract MLP post-activation neurons from a canonical cache.

    Returns the activations at ``blocks.{layer}.mlp.hook_out``, sliced
    to the requested position when the cache tensor carries a sequence
    dimension. MLP-class architectures (one-hot, embedding) produce
    seq-less activations; ``position`` is ignored for those.

    Args:
        cache: Canonical-name-keyed activation cache.
        layer: Layer index (default 0).
        position: Token position (default -1, last).

    Returns:
        Tensor of shape ``(batch, d_mlp)``.
    """
    canonical = canonical_hooks.hook(
        canonical_hooks.BLOCKS, layer, canonical_hooks.MLP, canonical_hooks.HOOK_OUT
    )
    acts = cache[canonical]
    if acts.ndim == 3:
        return acts[:, position, :]
    return acts


def extract_attention_patterns(
    cache: ActivationCache,
    layer: int = 0,
) -> torch.Tensor:
    """Extract attention patterns from a canonical cache.

    Args:
        cache: Canonical-name-keyed activation cache.
        layer: Layer index (default 0).

    Returns:
        Tensor of shape ``(batch, n_heads, seq_to, seq_from)``.
    """
    canonical = canonical_hooks.hook(
        canonical_hooks.BLOCKS,
        layer,
        canonical_hooks.ATTN,
        canonical_hooks.HOOK_PATTERN,
    )
    return cache[canonical]


def extract_residual_stream(
    cache: ActivationCache,
    layer: int = 0,
    position: int = -1,
    location: str = "resid_post",
) -> torch.Tensor:
    """Extract residual-stream activations at a layer site.

    Args:
        cache: Canonical-name-keyed activation cache.
        layer: Layer index (default 0).
        position: Token position (default -1).
        location: One of ``"resid_pre"`` (block input),
            ``"resid_post"`` (block output), ``"attn_out"`` (attention
            component output).

    Returns:
        Tensor of shape ``(batch, d_model)``.

    Raises:
        KeyError: If ``location`` is not recognized.
    """
    suffix = _RESID_LOCATION_TO_CANONICAL_SUFFIX.get(location)
    if suffix is None:
        raise KeyError(
            f"Unknown residual-stream location {location!r}. "
            f"Expected one of {sorted(_RESID_LOCATION_TO_CANONICAL_SUFFIX)}."
        )
    canonical = f"{canonical_hooks.BLOCKS}.{layer}.{suffix}"
    acts = cache[canonical]
    return acts[:, position, :]


def reshape_to_grid(activations: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Reshape flat batch of activations to 2D grid.

    For modular arithmetic tasks where inputs are (a, b) pairs,
    reshapes activations from (p^2, d) to (d, p, p).

    Args:
        activations: Tensor of shape (p^2, d) where p^2 = grid_size^2
        grid_size: Size of each grid dimension (p)

    Returns:
        Tensor of shape (d, p, p) where d is the feature dimension
    """
    return einops.rearrange(
        activations,
        "(a b) d -> d a b",
        a=grid_size,
        b=grid_size,
    )


def get_embedding_weights(
    model: HookedModel,
    exclude_special_tokens: int = 1,
) -> torch.Tensor:
    """Get embedding weights, optionally excluding special tokens.

    Reads ``embed.W_E`` via ``model.get_weight``. Architectures that do
    not publish a shared embedding matrix (e.g.,
    ``HookedEmbeddingMLP``, which exposes per-input ``embed.embed_a``
    and ``embed.embed_b``) raise ``KeyError`` from ``get_weight`` —
    callers expecting a transformer-class shared embedding must handle
    that case explicitly.

    Args:
        model: ``HookedModel`` providing canonical weight access.
        exclude_special_tokens: Number of trailing rows to exclude
            (default 1, for the equals token).

    Returns:
        Embedding weight tensor of shape ``(vocab_size - exclude, d_model)``.
    """
    W_E = model.get_weight(canonical_weights.EMBED_WEIGHT)
    if exclude_special_tokens > 0:
        return W_E[:-exclude_special_tokens]
    return W_E


def compute_grid_size_from_dataset(dataset: torch.Tensor) -> int:
    """Infer grid size (p) from dataset shape.

    For modular arithmetic datasets of shape (p^2, ...), returns p.

    Args:
        dataset: Dataset tensor whose first dimension is p^2.

    Returns:
        Grid size p where p^2 = n_samples
    """
    n_samples = dataset.shape[0]
    p = int(math.sqrt(n_samples))
    if p * p != n_samples:
        raise ValueError(f"Dataset size {n_samples} is not a perfect square")
    return p
