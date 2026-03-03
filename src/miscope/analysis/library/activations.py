"""Activation extraction and manipulation utilities.

This module provides functions for extracting and reshaping activations
from transformer models for analysis purposes.
"""

import einops
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache


def extract_mlp_activations(
    cache: ActivationCache,
    layer: int = 0,
    position: int = -1,
) -> torch.Tensor:
    """Extract MLP neuron activations from cache.

    Args:
        cache: Activation cache from model.run_with_cache()
        layer: Which transformer layer to extract from (default: 0)
        position: Token position to extract (default: -1, last position)

    Returns:
        Tensor of shape (batch, d_mlp) containing neuron activations
    """
    # Shape: (batch, seq_len, d_mlp)
    mlp_acts = cache["post", layer, "mlp"]
    # Extract specified position
    return mlp_acts[:, position, :]


def extract_attention_patterns(
    cache: ActivationCache,
    layer: int = 0,
) -> torch.Tensor:
    """Extract attention patterns from cache.

    Args:
        cache: Activation cache from model.run_with_cache()
        layer: Which transformer layer to extract from (default: 0)

    Returns:
        Tensor of shape (batch, n_heads, seq_to, seq_from)
        containing attention weights (softmax outputs).
    """
    return cache["pattern", layer]


def extract_residual_stream(
    cache: ActivationCache,
    layer: int = 0,
    position: int = -1,
    location: str = "resid_post",
) -> torch.Tensor:
    """Extract residual stream activations from cache.

    Args:
        cache: Activation cache from model.run_with_cache()
        layer: Which transformer layer (default: 0)
        position: Token position to extract (default: -1)
        location: One of "resid_pre", "resid_post", "attn_out"

    Returns:
        Tensor of shape (batch, d_model)
    """
    acts = cache[location, layer]  # (batch, seq_len, d_model)
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
    model: HookedTransformer,
    exclude_special_tokens: int = 1,
) -> torch.Tensor:
    """Get embedding weights from model, optionally excluding special tokens.

    Args:
        model: HookedTransformer model
        exclude_special_tokens: Number of tokens to exclude from end
                               (default: 1, for equals token)

    Returns:
        Embedding weight tensor of shape (vocab_size - exclude, d_model)
    """
    W_E = model.embed.W_E
    if exclude_special_tokens > 0:
        return W_E[:-exclude_special_tokens]
    return W_E


def run_with_cache(
    model: HookedTransformer,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, ActivationCache]:
    """Run model forward pass and return logits and activation cache.

    Convenience wrapper around model.run_with_cache() with inference mode.

    Args:
        model: HookedTransformer model
        inputs: Input tensor

    Returns:
        Tuple of (logits, cache)
    """
    with torch.inference_mode():
        logits, cache = model.run_with_cache(inputs)
    return logits, cache  # type: ignore[return-value]


def compute_grid_size_from_dataset(dataset: torch.Tensor) -> int:
    """Infer grid size (p) from dataset shape.

    For modular arithmetic datasets of shape (p^2, 3), returns p.

    Args:
        dataset: Dataset tensor of shape (n_samples, seq_len)

    Returns:
        Grid size p where p^2 = n_samples
    """
    import math

    n_samples = dataset.shape[0]
    p = int(math.sqrt(n_samples))
    if p * p != n_samples:
        raise ValueError(f"Dataset size {n_samples} is not a perfect square")
    return p
