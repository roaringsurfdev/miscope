"""Frequency-selective attention gain hook for intervention training.

Implements REQ_068: applies a per-frequency gain function to hook_attn_out
during a configurable intervention window, using W_E-based Fourier directions
to project and rescale the attention signal.

Usage:
    # Load the model at plateau onset to capture W_E at first descent
    model_at_plateau = thrasher_variant.load_model_at_checkpoint(epoch=1500)
    W_E = model_at_plateau.embed.W_E.detach()

    # Build frequency directions from the family's Fourier basis
    ctx = thrasher_variant.analysis_context()
    D_sin, D_cos = build_frequency_directions(ctx["fourier_basis"], W_E, prime=59)

    # Construct the hook with the intervention config and directions
    hook = FrequencyGainHook(intervention_config, D_sin, D_cos)

    # Create an intervention variant and train with the hook
    family = registry.get_family("modadd_intervention")
    iv_variant = family.create_intervention_variant(
        prime=59, seed=485, data_seed=598,
        intervention_config=intervention_config,
        results_dir=results_dir,
    )
    iv_variant.train(training_hook=hook)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

# Minimum gain floor — prevents hard zeroing (REQ_068 constraint)
MIN_GAIN: float = 0.1
DEFAULT_RAMP_EPOCHS: int = 50

# Hook registration point for attention output in a 1-layer HookedTransformer
ATTN_OUT_HOOK: str = "blocks.0.hook_attn_out"


def build_frequency_directions(
    fourier_basis: torch.Tensor,
    W_E: torch.Tensor,
    prime: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute frequency-aligned directions in d_model space.

    For each frequency k (1-based, 1..prime//2), projects the sin_k and cos_k
    Fourier basis vectors through W_E to obtain directions in residual stream
    space. These directions capture how frequency k is represented in the model's
    embedding space.

    D_sin[k], D_cos[k] = normalize(sin_k @ W_E[:prime]),  normalize(cos_k @ W_E[:prime])

    Args:
        fourier_basis: (p, p) Fourier basis from get_fourier_basis(prime).
            Row layout: row 0 = constant, row 2k-1 = sin k, row 2k = cos k.
        W_E: (vocab_size, d_model) embedding weight matrix from model.embed.W_E.
            Rows [:prime] correspond to input tokens (0..prime-1).
        prime: Modulus p. Only rows [:prime] of W_E are used.

    Returns:
        D_sin: (n_freqs, d_model) — normalized sin-direction per frequency
        D_cos: (n_freqs, d_model) — normalized cos-direction per frequency
        where n_freqs = prime // 2
    """
    n_freqs = prime // 2
    W_input = W_E[:prime]  # (prime, d_model)

    sin_rows = fourier_basis[1::2][:n_freqs]  # sin k=1..n_freqs, shape (n_freqs, prime)
    cos_rows = fourier_basis[2::2][:n_freqs]  # cos k=1..n_freqs, shape (n_freqs, prime)

    D_sin = sin_rows @ W_input  # (n_freqs, d_model)
    D_cos = cos_rows @ W_input  # (n_freqs, d_model)

    D_sin = D_sin / D_sin.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    D_cos = D_cos / D_cos.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return D_sin, D_cos


def _build_gain_vector(config: dict[str, Any], n_freqs: int) -> torch.Tensor:
    """Build a per-frequency gain tensor from the intervention config.

    Frequencies not listed in target_frequencies are left at 1.0 (no change).
    Gains below MIN_GAIN are clamped to prevent hard zeroing.

    Args:
        config: Intervention config with 'gain' dict and 'target_frequencies' list.
            Frequency labels are 1-based integers (matching QK^T fraction output).
        n_freqs: Total number of frequencies (prime // 2).

    Returns:
        gain: (n_freqs,) tensor of per-frequency scalars
    """
    raw_gain: dict[str | int, Any] = config.get("gain", {})
    target_freqs: list[int] = config.get("target_frequencies", [])

    gain = torch.ones(n_freqs)
    for freq_label in target_freqs:
        freq_idx = int(freq_label) - 1  # 1-based label → 0-based index
        if 0 <= freq_idx < n_freqs:
            # Config gain dict may have int or string keys (JSON round-trip)
            raw = raw_gain.get(freq_label, raw_gain.get(str(freq_label), 1.0))
            gain[freq_idx] = max(float(raw), MIN_GAIN)

    return gain


def _compute_ramp_factor(epoch: int, config: dict[str, Any]) -> float:
    """Compute the gain interpolation factor for the current epoch.

    Outside [epoch_start, epoch_end): returns 0.0 (hook inactive).
    At window start: ramps linearly from 0.0 to 1.0 over ramp_epochs.
    At window end:   ramps linearly from 1.0 back to 0.0 over ramp_epochs.
    In the middle:   returns 1.0 (full gain applied).

    Args:
        epoch: Current training epoch.
        config: Intervention config with epoch_start, epoch_end, ramp_epochs.

    Returns:
        Factor in [0.0, 1.0].
    """
    epoch_start: int = config["epoch_start"]
    epoch_end: int = config["epoch_end"]
    ramp_epochs: int = config.get("ramp_epochs", DEFAULT_RAMP_EPOCHS)

    if epoch < epoch_start or epoch >= epoch_end:
        return 0.0

    elapsed = epoch - epoch_start
    remaining = epoch_end - epoch

    if elapsed < ramp_epochs:
        return elapsed / ramp_epochs
    if remaining <= ramp_epochs:
        return remaining / ramp_epochs
    return 1.0


def _apply_frequency_gain(
    value: torch.Tensor,
    D_sin: torch.Tensor,
    D_cos: torch.Tensor,
    gain_vector: torch.Tensor,
    ramp_factor: float,
) -> torch.Tensor:
    """Apply ramped frequency-selective gain to an attention output tensor.

    For each frequency k, projects value onto the sin_k and cos_k directions,
    scales those projections by the effective gain, and adds the delta back.
    The component of value orthogonal to all frequency directions is preserved.

    effective_gain[k] = 1.0 + (gain[k] - 1.0) * ramp_factor

    Args:
        value: (batch, seq_len, d_model) — attention output from hook_attn_out
        D_sin: (n_freqs, d_model) — normalized sin directions per frequency
        D_cos: (n_freqs, d_model) — normalized cos directions per frequency
        gain_vector: (n_freqs,) — per-frequency gain scalars
        ramp_factor: float in (0.0, 1.0] — interpolation toward target gain

    Returns:
        Modified attention output, same shape as value.
    """
    effective_gain = 1.0 + (gain_vector - 1.0) * ramp_factor  # (n_freqs,)

    amp_sin = value @ D_sin.T  # (batch, seq_len, n_freqs)
    amp_cos = value @ D_cos.T  # (batch, seq_len, n_freqs)

    sin_delta = ((amp_sin * effective_gain) - amp_sin) @ D_sin  # (batch, seq_len, d_model)
    cos_delta = ((amp_cos * effective_gain) - amp_cos) @ D_cos  # (batch, seq_len, d_model)

    return value + sin_delta + cos_delta


def compute_hook_verification(
    variant: Any,
    epoch: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute per-frequency amplitude of hook_attn_out: baseline vs. hook-modified.

    Reconstructs D_sin/D_cos from the plateau checkpoint's W_E (matching what
    was used during training), then runs the full analysis dataset at the
    selected epoch to measure the frequency content of hook_attn_out before
    and after the gain function is applied.

    Args:
        variant: Intervention Variant with a valid intervention config in config.json
        epoch: Checkpoint epoch to analyze
        device: Device for model and tensor operations

    Returns:
        dict with:
            baseline_power:     (n_freqs,) ndarray — RMS amplitude per frequency
            modified_power:     (n_freqs,) ndarray — RMS amplitude after hook
            freq_labels:        list[int] — 1-based frequency labels
            ramp_factor:        float — gain interpolation factor for this epoch
            target_frequencies: list[int] — 1-based target freq labels from config
            gain:               dict[int, float] — per-frequency gain values
            epoch_start:        int
            epoch_end:          int
            ramp_epochs:        int
            plateau_epoch:      int — epoch whose W_E was used for D_sin/D_cos
            epoch:              int — the analyzed epoch
            prime:              int
    """
    import torch

    config = variant.model_config
    intervention_config = config["intervention"]
    prime = int(config["prime"])
    plateau_epoch = int(intervention_config["epoch_start"])

    # Build frequency directions from W_E at the plateau checkpoint —
    # this matches exactly what FrequencyGainHook used during training.
    model_plateau = variant.load_model_at_checkpoint(plateau_epoch)
    model_plateau.to(device)
    W_E = model_plateau.embed.W_E.detach()
    ctx = variant.analysis_context(device=device)
    D_sin, D_cos = build_frequency_directions(ctx["fourier_basis"], W_E, prime=prime)
    del model_plateau

    # Load model at the selected epoch and run the analysis probe with cache.
    model = variant.load_model_at_checkpoint(epoch)
    model.to(device)
    probe = variant.analysis_dataset(device=device)

    with torch.inference_mode():
        _, cache = model.run_with_cache(probe)
    del model

    attn_out = cache[ATTN_OUT_HOOK]  # (batch, seq_len, d_model)

    # Project onto frequency directions: (batch, seq_len, n_freqs)
    amp_sin = attn_out @ D_sin.T
    amp_cos = attn_out @ D_cos.T
    # RMS amplitude averaged across all batch items and sequence positions
    baseline_power = (amp_sin**2 + amp_cos**2).mean(dim=(0, 1)).sqrt()

    # Apply the gain function at this epoch's ramp factor
    ramp_factor = _compute_ramp_factor(epoch, intervention_config)
    n_freqs = D_sin.shape[0]
    gain_vector = _build_gain_vector(intervention_config, n_freqs).to(device)
    modified = _apply_frequency_gain(attn_out, D_sin, D_cos, gain_vector, ramp_factor)

    mod_sin = modified @ D_sin.T
    mod_cos = modified @ D_cos.T
    modified_power = (mod_sin**2 + mod_cos**2).mean(dim=(0, 1)).sqrt()

    # Normalize gain dict to int keys (JSON round-trip may produce string keys)
    raw_gain: dict[str, Any] = intervention_config.get("gain", {})
    gain_normalized = {int(k): float(v) for k, v in raw_gain.items()}

    return {
        "baseline_power": baseline_power.cpu().numpy(),
        "modified_power": modified_power.cpu().numpy(),
        "freq_labels": list(range(1, n_freqs + 1)),
        "ramp_factor": ramp_factor,
        "target_frequencies": [int(f) for f in intervention_config.get("target_frequencies", [])],
        "gain": gain_normalized,
        "epoch_start": int(intervention_config["epoch_start"]),
        "epoch_end": int(intervention_config["epoch_end"]),
        "ramp_epochs": int(intervention_config.get("ramp_epochs", DEFAULT_RAMP_EPOCHS)),
        "plateau_epoch": plateau_epoch,
        "epoch": epoch,
        "prime": prime,
    }


class FrequencyGainHook:
    """Training hook applying frequency-selective gain to hook_attn_out.

    Implements the "EQ" metaphor from REQ_068: adjusts the balance of existing
    frequency signals in the attention output without introducing foreign structure.
    The gain ramps linearly in at window start and back out at window end to avoid
    discontinuous jumps.

    The frequency directions (D_sin, D_cos) must be pre-computed from the model's
    W_E at the intervention epoch — typically the plateau onset checkpoint. These
    directions are fixed for the duration of the window (stable, not reactive).

    See module docstring for usage example.
    """

    def __init__(
        self,
        config: dict[str, Any],
        D_sin: torch.Tensor,
        D_cos: torch.Tensor,
    ):
        """Initialize the hook.

        Args:
            config: Intervention config dict. Must include: type, target_frequencies,
                gain, epoch_start, epoch_end. Optional: ramp_epochs (default 50).
            D_sin: (n_freqs, d_model) — normalized sin directions from
                build_frequency_directions().
            D_cos: (n_freqs, d_model) — normalized cos directions from
                build_frequency_directions().
        """
        self._config = config
        self._D_sin = D_sin
        self._D_cos = D_cos
        n_freqs = D_sin.shape[0]
        self._gain_vector = _build_gain_vector(config, n_freqs).to(D_sin.device)

    def __call__(self, epoch: int) -> list[tuple[str, Callable[..., Any]]]:
        """Return hook tuples for this epoch, or [] if outside the window.

        Called by Variant.train() once per epoch. Returns an empty list outside
        [epoch_start, epoch_end), which keeps the standard forward pass active.

        Args:
            epoch: Current training epoch.

        Returns:
            [(ATTN_OUT_HOOK, hook_fn)] if inside window, [] otherwise.
        """
        ramp_factor = _compute_ramp_factor(epoch, self._config)
        if ramp_factor == 0.0:
            return []

        D_sin = self._D_sin
        D_cos = self._D_cos
        gain_vector = self._gain_vector

        def hook_fn(value: torch.Tensor, hook: Any) -> torch.Tensor:
            return _apply_frequency_gain(value, D_sin, D_cos, gain_vector, ramp_factor)

        return [(ATTN_OUT_HOOK, hook_fn)]
