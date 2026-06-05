"""Tests for REQ_074 / REQ_110D: VariantAnalysisSummary and the variant registry.

The deprecated ``variant_summary`` module (``compute_variant_summary`` /
``write_variant_summary`` / ``extract_learned_frequencies``) was removed in
REQ_110D; ``VariantAnalysisSummary`` is the canonical engine and
``build_variant_registry`` now lives in ``variant_analysis_summary``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from miscope.analysis.variant_analysis_summary import (
    VariantAnalysisSummary,
    build_variant_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D_MLP = 512
_N_EPOCHS = 50
_PRIME = 113


def _make_nd_data(
    freq_per_neuron: list[int] | None = None,
    committed_fraction: float = 1.0,
    threshold: float = 0.75,
) -> dict:
    """Build a minimal neuron_dynamics cross-epoch artifact for testing."""
    if freq_per_neuron is None:
        freq_per_neuron = [14 if i % 2 == 0 else 28 for i in range(_D_MLP)]

    dominant_freq = np.zeros((_N_EPOCHS, _D_MLP), dtype=float)
    dominant_freq[:] = np.array(freq_per_neuron, dtype=float)

    max_frac = np.full((_N_EPOCHS, _D_MLP), threshold + 0.1)
    n_uncommitted = int(_D_MLP * (1.0 - committed_fraction))
    if n_uncommitted > 0:
        max_frac[:, -n_uncommitted:] = threshold - 0.1

    commitment_epochs = np.full(_D_MLP, float("nan"))
    epochs = np.arange(_N_EPOCHS, dtype=float) * 100

    return {
        "dominant_freq": dominant_freq,
        "max_frac": max_frac,
        "threshold": np.array([threshold]),
        "commitment_epochs": commitment_epochs,
        "epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Integration: build_variant_registry
# ---------------------------------------------------------------------------


def test_build_variant_registry_one_entry_per_variant(tmp_path):
    from miscope.families.base_model_family import BaseModelFamily

    family_name = "modulo_addition_1layer"
    config = {
        "name": family_name,
        "display_name": "Modulo Addition (1 Layer)",
        "description": "Test",
        "architecture": {},
        "domain_parameters": {
            "prime": {"type": "int"},
            "seed": {"type": "int"},
            "data_seed": {"type": "int"},
        },
        "analyzers": [],
        "visualizations": [],
        "analysis_dataset": {"type": "modulo_addition_grid"},
        "variant_pattern": "p{prime}_seed{seed}_dseed{data_seed}",
    }
    family = BaseModelFamily(config, data_root=tmp_path)
    family.variants_dir.mkdir(parents=True)

    for prime, mseed in [(113, 485), (113, 999)]:
        vdir = family.variants_dir / f"p{prime}_seed{mseed}_dseed598"
        vdir.mkdir(parents=True)
        summary = {
            "prime": prime,
            "model_seed": mseed,
            "data_seed": 598,
            "family": family_name,
            "computed_at": "2026-03-17T00:00:00+00:00",
            "learned_frequencies": [14, 28],
            "learned_frequency_count": 2,
            "canonical_specialization_threshold": 0.10,
            "second_descent_onset_epoch": 5000,
            "second_descent_completion_epoch": None,
            "second_descent_survived": True,
            "committed_frequencies_at_onset": [14, 28],
            "handshake_failures": [],
            "handshake_succeeded": True,
            "failure_mode": "healthy",
            "max_resid_post_circularity": 0.92,
            "final_specialized_frequency_count": 2,
            "peak_test_loss_epoch": 4000,
            "final_test_loss": 0.00001,
        }
        (vdir / "variant_summary.json").write_text(json.dumps(summary))

    registry_path = build_variant_registry(family)

    assert registry_path.exists()
    registry = json.loads(registry_path.read_text())
    assert len(registry) == 2

    # REQ_107: variant_id is the family-owned composed handle (the variant
    # directory name), not the hardcoded "{prime}_{model_seed}_{data_seed}".
    variant_ids = {e["variant_id"] for e in registry}
    assert "p113_seed485_dseed598" in variant_ids
    assert "p113_seed999_dseed598" in variant_ids

    # The family's declared domain_parameters are added as columns for filtering.
    by_id = {e["variant_id"]: e for e in registry}
    assert by_id["p113_seed485_dseed598"]["prime"] == 113
    assert by_id["p113_seed485_dseed598"]["seed"] == 485
    assert by_id["p113_seed485_dseed598"]["data_seed"] == 598


# ---------------------------------------------------------------------------
# VariantAnalysisSummary: field-population behavior
# ---------------------------------------------------------------------------


def _make_vas_variant(
    nd_data: dict | None = None,
    tf_data: dict | None = None,
    prime: int = _PRIME,
    seed: int = 485,
    data_seed: int = 598,
) -> MagicMock:
    """Build a minimal mock for VariantAnalysisSummary method testing."""
    variant = MagicMock()
    variant.params = {"prime": prime, "seed": seed, "data_seed": data_seed}
    variant.family.name = "modulo_addition_1layer"
    variant.get_available_checkpoints.return_value = list(range(0, 5000, 100))

    def mock_cross_epoch(name):
        if name == "neuron_dynamics" and nd_data is not None:
            return nd_data
        if name == "transient_frequency" and tf_data is not None:
            return tf_data
        raise FileNotFoundError(name)

    variant.artifacts.load_cross_epoch.side_effect = mock_cross_epoch
    return variant


def _make_vas(variant: MagicMock, d_mlp: int = _D_MLP) -> VariantAnalysisSummary:
    """Construct a VariantAnalysisSummary with an empty summary_data dict.

    Stubs ``analysis_data`` with a pre-loaded conformed-dimension handle so the
    REQ_110D ``_attribution()`` accessor returns a fake instead of reaching the
    warehouse — these tests exercise field-population logic, not the loader.
    """
    vas = VariantAnalysisSummary.__new__(VariantAnalysisSummary)
    vas.variant = variant
    vas.analysis_data = SimpleNamespace(
        neurons_loaded=True, attribution=SimpleNamespace(d_mlp=d_mlp)
    )
    vas.summary_data = {
        "prime": variant.params["prime"],
        "second_descent_onset_epoch": None,
        "learned_frequencies": None,
    }
    return vas


def test_vas_transient_metrics_none_without_artifact():
    variant = _make_vas_variant(tf_data=None)
    vas = _make_vas(variant)
    vas._load_transient_metrics()
    assert vas.summary_data["transient_frequencies"] is None
    assert vas.summary_data["transient_frequency_count"] is None
    assert vas.summary_data["homeless_neuron_count"] is None


def test_vas_transient_metrics_populated():
    tf = {
        "ever_qualified_freqs": np.array([13, 39], dtype=np.int32),
        "is_final": np.array([True, False], dtype=bool),
        "homeless_count": np.array([0, 30], dtype=np.int32),
        "_transient_canonical_threshold": np.float32(0.05),
    }
    nd = _make_nd_data()
    variant = _make_vas_variant(nd_data=nd, tf_data=tf)
    vas = _make_vas(variant)
    vas._load_transient_metrics()
    assert vas.summary_data["transient_frequencies"] == [40]  # 0-indexed 39 → 1-indexed 40
    assert vas.summary_data["transient_frequency_count"] == 1
    assert vas.summary_data["homeless_neuron_count"] == 30
    assert vas.summary_data["transient_detection_threshold"] == pytest.approx(0.05)


def test_vas_failure_mode_populated():
    variant = _make_vas_variant()
    vas = _make_vas(variant)
    vas.summary_data["second_descent_onset_epoch"] = 5000
    vas.summary_data["test_loss_final"] = 1e-8
    vas.summary_data["post_descent_test_loss_increase"] = False
    vas._load_failure_mode()
    assert vas.summary_data["failure_mode"] is not None
    assert isinstance(vas.summary_data["failure_mode_reasons"], list)
