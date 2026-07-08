"""Loss landscape flatness analysis utilities.

Provides functions for measuring local loss landscape flatness via
random perturbation of model parameters. Used by
LandscapeFlatnessAnalyzer (REQ_031).

The compute function takes any ``nn.Module`` — only ``state_dict``,
``parameters``, and ``load_state_dict`` are used, all of which are
generic PyTorch surface. Works against ``HookedTransformer``,
``HookedOneHotMLP``, or ``HookedEmbeddingMLP`` uniformly.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


def compute_landscape_flatness(
    model: nn.Module,
    probe: torch.Tensor,
    loss_fn: Callable[[nn.Module, torch.Tensor], float],
    n_directions: int = 50,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Measure local loss landscape flatness via random perturbation.

    Samples random unit directions in parameter space, perturbs the
    model weights by epsilon along each direction, and measures the
    resulting loss change.

    Args:
        model: Model at a specific checkpoint.
        probe: Input tensor for loss computation.
        loss_fn: Function(model, probe) -> scalar loss.
        n_directions: Number of random perturbation directions.
        epsilon: Perturbation magnitude (L2 norm of perturbation vector).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Dict with:
          "baseline_loss": scalar array
          "delta_losses": array of shape (n_directions,)
          "epsilon": scalar array
    """
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    baseline_loss = loss_fn(model, probe)

    param_list = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in param_list)
    device = param_list[0].device

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.seed()

    delta_losses = np.empty(n_directions)

    for i in range(n_directions):
        direction = torch.randn(total_params, generator=rng, device=device)
        direction = direction / direction.norm()

        offset = 0
        for p in param_list:
            n = p.numel()
            chunk = direction[offset : offset + n].view_as(p.data)
            p.data.add_(chunk, alpha=epsilon)
            offset += n

        perturbed_loss = loss_fn(model, probe)
        delta_losses[i] = perturbed_loss - baseline_loss

        model.load_state_dict(original_state)

    return {
        "baseline_loss": np.array(baseline_loss, dtype=np.float32),
        "delta_losses": delta_losses.astype(np.float32),
        "epsilon": np.array(epsilon, dtype=np.float32),
    }
