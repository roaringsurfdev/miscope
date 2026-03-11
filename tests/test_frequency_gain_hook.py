"""Tests for REQ_068: FrequencyGainHook and frequency direction utilities.

Covers:
- build_frequency_directions: shape, normalization, device
- _compute_ramp_factor: boundary behavior, linear ramps
- _build_gain_vector: freq label mapping, gain floor, default=1.0
- _apply_frequency_gain: identity at gain=1.0, orthogonal preservation,
  correct scaling, ramp interpolation
- FrequencyGainHook: inactive outside window, active inside, no-op at gain=1.0
"""

from __future__ import annotations

import torch
import pytest

from miscope.families.implementations.frequency_gain_hook import (
    ATTN_OUT_HOOK,
    MIN_GAIN,
    FrequencyGainHook,
    _apply_frequency_gain,
    _build_gain_vector,
    _compute_ramp_factor,
    build_frequency_directions,
)
from miscope.analysis.library import get_fourier_basis

PRIME = 59
N_FREQS = PRIME // 2  # 29
D_MODEL = 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fourier_basis() -> torch.Tensor:
    basis, _ = get_fourier_basis(PRIME, "cpu")
    return basis


@pytest.fixture
def W_E() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(PRIME + 1, D_MODEL)  # vocab = p+1 (includes equals token)


@pytest.fixture
def directions(fourier_basis: torch.Tensor, W_E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return build_frequency_directions(fourier_basis, W_E, prime=PRIME)


@pytest.fixture
def base_config() -> dict:
    return {
        "type": "frequency_gain",
        "target_heads": "all",
        "target_frequencies": [4, 10, 12],
        "gain": {4: 0.3, 10: 0.3, 12: 0.3},
        "epoch_start": 1500,
        "epoch_end": 6500,
        "ramp_epochs": 200,
    }


@pytest.fixture
def noop_config() -> dict:
    return {
        "type": "frequency_gain",
        "target_heads": "all",
        "target_frequencies": [],
        "gain": {},
        "epoch_start": 1500,
        "epoch_end": 6500,
        "ramp_epochs": 200,
    }


# ---------------------------------------------------------------------------
# build_frequency_directions
# ---------------------------------------------------------------------------


def test_directions_shape(directions: tuple[torch.Tensor, torch.Tensor]) -> None:
    D_sin, D_cos = directions
    assert D_sin.shape == (N_FREQS, D_MODEL)
    assert D_cos.shape == (N_FREQS, D_MODEL)


def test_directions_are_normalized(directions: tuple[torch.Tensor, torch.Tensor]) -> None:
    D_sin, D_cos = directions
    norms_sin = D_sin.norm(dim=-1)
    norms_cos = D_cos.norm(dim=-1)
    assert torch.allclose(norms_sin, torch.ones(N_FREQS), atol=1e-5)
    assert torch.allclose(norms_cos, torch.ones(N_FREQS), atol=1e-5)


def test_directions_sin_cos_not_identical(directions: tuple[torch.Tensor, torch.Tensor]) -> None:
    D_sin, D_cos = directions
    # sin and cos directions should differ for each frequency
    assert not torch.allclose(D_sin, D_cos)


def test_directions_different_per_frequency(directions: tuple[torch.Tensor, torch.Tensor]) -> None:
    D_sin, _ = directions
    # Adjacent frequency directions should not be identical
    for i in range(N_FREQS - 1):
        assert not torch.allclose(D_sin[i], D_sin[i + 1])


# ---------------------------------------------------------------------------
# _compute_ramp_factor
# ---------------------------------------------------------------------------


def test_ramp_zero_before_window() -> None:
    config = {"epoch_start": 1500, "epoch_end": 6500, "ramp_epochs": 200}
    assert _compute_ramp_factor(0, config) == 0.0
    assert _compute_ramp_factor(1499, config) == 0.0


def test_ramp_zero_at_and_after_epoch_end() -> None:
    config = {"epoch_start": 1500, "epoch_end": 6500, "ramp_epochs": 200}
    assert _compute_ramp_factor(6500, config) == 0.0
    assert _compute_ramp_factor(7000, config) == 0.0


def test_ramp_full_in_middle() -> None:
    config = {"epoch_start": 1500, "epoch_end": 6500, "ramp_epochs": 200}
    assert _compute_ramp_factor(4000, config) == 1.0


def test_ramp_linear_at_start() -> None:
    config = {"epoch_start": 1500, "epoch_end": 6500, "ramp_epochs": 200}
    # At epoch_start + ramp_epochs/2, ramp should be 0.5
    assert _compute_ramp_factor(1600, config) == pytest.approx(0.5)
    assert _compute_ramp_factor(1500, config) == pytest.approx(0.0)
    assert _compute_ramp_factor(1700, config) == pytest.approx(1.0)


def test_ramp_linear_at_end() -> None:
    config = {"epoch_start": 1500, "epoch_end": 6500, "ramp_epochs": 200}
    # 200 epochs before end: ramp starts declining
    assert _compute_ramp_factor(6300, config) == pytest.approx(1.0)
    assert _compute_ramp_factor(6400, config) == pytest.approx(0.5)
    assert _compute_ramp_factor(6499, config) == pytest.approx(pytest.approx(1 / 200))


def test_ramp_defaults_ramp_epochs() -> None:
    # Without ramp_epochs, default should be used (not crash)
    config = {"epoch_start": 0, "epoch_end": 1000}
    assert _compute_ramp_factor(500, config) == 1.0


# ---------------------------------------------------------------------------
# _build_gain_vector
# ---------------------------------------------------------------------------


def test_gain_vector_shape() -> None:
    config = {"target_frequencies": [1, 5], "gain": {1: 0.5, 5: 2.0}}
    g = _build_gain_vector(config, N_FREQS)
    assert g.shape == (N_FREQS,)


def test_gain_vector_defaults_to_one() -> None:
    config = {"target_frequencies": [], "gain": {}}
    g = _build_gain_vector(config, N_FREQS)
    assert torch.allclose(g, torch.ones(N_FREQS))


def test_gain_vector_maps_freq_label_to_index() -> None:
    config = {"target_frequencies": [1], "gain": {1: 0.5}}
    g = _build_gain_vector(config, N_FREQS)
    assert g[0].item() == pytest.approx(0.5)  # freq 1 → index 0
    assert g[1].item() == pytest.approx(1.0)  # freq 2 → unchanged


def test_gain_vector_accepts_string_keys() -> None:
    # JSON round-trip turns int keys to strings
    config = {"target_frequencies": [4], "gain": {"4": 0.3}}
    g = _build_gain_vector(config, N_FREQS)
    assert g[3].item() == pytest.approx(0.3)


def test_gain_vector_enforces_min_gain_floor() -> None:
    config = {"target_frequencies": [1], "gain": {1: 0.0}}
    g = _build_gain_vector(config, N_FREQS)
    assert g[0].item() >= MIN_GAIN


def test_gain_vector_ignores_out_of_range_freq() -> None:
    config = {"target_frequencies": [999], "gain": {999: 0.1}}
    g = _build_gain_vector(config, N_FREQS)
    assert torch.allclose(g, torch.ones(N_FREQS))


# ---------------------------------------------------------------------------
# _apply_frequency_gain
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_attn_out() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(4, 3, D_MODEL)  # (batch=4, seq_len=3, d_model=128)


def test_apply_gain_identity_at_gain_one(
    dummy_attn_out: torch.Tensor,
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    D_sin, D_cos = directions
    gain = torch.ones(N_FREQS)
    result = _apply_frequency_gain(dummy_attn_out, D_sin, D_cos, gain, ramp_factor=1.0)
    assert torch.allclose(result, dummy_attn_out, atol=1e-5)


def test_apply_gain_identity_at_ramp_zero(
    dummy_attn_out: torch.Tensor,
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    D_sin, D_cos = directions
    gain = torch.full((N_FREQS,), 0.3)  # all dampened
    result = _apply_frequency_gain(dummy_attn_out, D_sin, D_cos, gain, ramp_factor=0.0)
    assert torch.allclose(result, dummy_attn_out, atol=1e-5)


def test_apply_gain_output_shape(
    dummy_attn_out: torch.Tensor,
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    D_sin, D_cos = directions
    gain = torch.ones(N_FREQS)
    result = _apply_frequency_gain(dummy_attn_out, D_sin, D_cos, gain, ramp_factor=1.0)
    assert result.shape == dummy_attn_out.shape


def test_apply_gain_preserves_orthogonal_component(
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A vector in the null space of [D_sin; D_cos] is unchanged by any gain."""
    D_sin, D_cos = directions
    D_all = torch.cat([D_sin, D_cos], dim=0)  # (2*n_freqs, d_model)

    # SVD gives null space vectors: rows of Vh beyond the rank of D_all
    _, S, Vh = torch.linalg.svd(D_all, full_matrices=True)
    rank = (S > 1e-5).sum().item()
    null_vec = Vh[rank]  # guaranteed orthogonal to all of D_sin and D_cos

    value = null_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    gain = torch.full((N_FREQS,), 0.1)  # strong dampening
    result = _apply_frequency_gain(value, D_sin, D_cos, gain, ramp_factor=1.0)
    assert torch.allclose(result, value, atol=1e-4)


def test_apply_gain_scales_frequency_projection(
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The D_sin[0] projection is scaled by gain[0] when amp_cos[0]=0.

    The general formula is:
        result · D_sin[0] = gain[0] * amp_sin[0] + (gain[0]-1) * amp_cos[0] * (D_cos[0] · D_sin[0])

    We zero the D_cos[0] component of value so the cross term vanishes,
    then verify that the sin projection is scaled exactly by gain[0].
    """
    D_sin, D_cos = directions
    torch.manual_seed(7)
    raw = torch.randn(1, 1, D_MODEL)
    # Remove D_cos[0] component so amp_cos[0] = 0 and cross term vanishes
    proj_cos_0 = (raw @ D_cos[0]).unsqueeze(-1) * D_cos[0]
    value = raw - proj_cos_0

    gain = torch.ones(N_FREQS)
    gain[0] = 2.0  # only freq 1 is modified

    result = _apply_frequency_gain(value, D_sin, D_cos, gain, ramp_factor=1.0)

    amp_before = (value @ D_sin[0]).item()
    amp_after = (result @ D_sin[0]).item()
    assert amp_after == pytest.approx(2.0 * amp_before, abs=1e-5)


def test_apply_gain_ramp_interpolates_projection(
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """At ramp_factor=0.5, effective gain is 1 + (target-1)*0.5.

    Uses the same amp_cos[0]=0 construction as the scaling test to eliminate
    the cross term, then verifies effective gain = 1 + (3-1)*0.5 = 2.0.
    """
    D_sin, D_cos = directions
    torch.manual_seed(7)
    raw = torch.randn(1, 1, D_MODEL)
    # Remove D_cos[0] component so amp_cos[0] = 0 and cross term vanishes
    proj_cos_0 = (raw @ D_cos[0]).unsqueeze(-1) * D_cos[0]
    value = raw - proj_cos_0

    gain = torch.ones(N_FREQS)
    gain[0] = 3.0  # effective at ramp=0.5: 1 + (3-1)*0.5 = 2.0

    result = _apply_frequency_gain(value, D_sin, D_cos, gain, ramp_factor=0.5)

    amp_before = (value @ D_sin[0]).item()
    amp_after = (result @ D_sin[0]).item()
    assert amp_after == pytest.approx(2.0 * amp_before, abs=1e-5)


# ---------------------------------------------------------------------------
# FrequencyGainHook
# ---------------------------------------------------------------------------


def test_hook_inactive_before_window(
    base_config: dict,
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    D_sin, D_cos = directions
    hook = FrequencyGainHook(base_config, D_sin, D_cos)
    assert hook(0) == []
    assert hook(1499) == []


def test_hook_inactive_at_and_after_epoch_end(
    base_config: dict,
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    D_sin, D_cos = directions
    hook = FrequencyGainHook(base_config, D_sin, D_cos)
    assert hook(6500) == []
    assert hook(9999) == []


def test_hook_active_inside_window(
    base_config: dict,
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    D_sin, D_cos = directions
    hook = FrequencyGainHook(base_config, D_sin, D_cos)
    result = hook(4000)
    assert len(result) == 1
    hook_name, hook_fn = result[0]
    assert hook_name == ATTN_OUT_HOOK
    assert callable(hook_fn)


def test_hook_fn_returns_correct_shape(
    base_config: dict,
    directions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    D_sin, D_cos = directions
    hook = FrequencyGainHook(base_config, D_sin, D_cos)
    _, hook_fn = hook(4000)[0]
    dummy = torch.randn(4, 3, D_MODEL)
    result = hook_fn(dummy, None)
    assert result.shape == dummy.shape


def test_noop_hook_is_identity(
    noop_config: dict,
    directions: tuple[torch.Tensor, torch.Tensor],
    dummy_attn_out: torch.Tensor,
) -> None:
    """No-op intervention (empty target_frequencies) leaves attn_out unchanged."""
    D_sin, D_cos = directions
    hook = FrequencyGainHook(noop_config, D_sin, D_cos)
    hooks = hook(4000)
    # Even with no target frequencies, hook is active (window is open)
    # but gain=1.0 everywhere means output = input
    if hooks:
        _, hook_fn = hooks[0]
        result = hook_fn(dummy_attn_out, None)
        assert torch.allclose(result, dummy_attn_out, atol=1e-5)
    # If hooks is empty (gain all 1.0 and optimized out), also fine
