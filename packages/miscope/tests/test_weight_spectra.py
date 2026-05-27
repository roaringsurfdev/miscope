"""Tests for REQ_111: WeightSpectraAnalyzer.

Successor to ``effective_dimensionality``. The integration block doubles as
the parity test — both analyzers run on the same trained variant and their
singular values are compared bit-for-bit.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from miscope.analysis import AnalysisPipeline, Analyzer
from miscope.analysis.analyzers import (
    EffectiveDimensionalityAnalyzer,
    WeightSpectraAnalyzer,
)
from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.library.weights import (
    ATTENTION_MATRICES,
    WEIGHT_MATRIX_NAMES,
    compute_weight_spectra,
)
from miscope.families.discovery import discover_families

# ── Library: compute_weight_spectra ───────────────────────────────────


class TestComputeWeightSpectra:
    """Tests for compute_weight_spectra."""

    @pytest.fixture
    def model(self):
        """Create a minimal HookedTransformer."""
        from miscope.architectures import HookedTransformer, HookedTransformerConfig

        cfg = HookedTransformerConfig(
            d_model=32,
            d_head=8,
            n_heads=4,
            n_layers=1,
            d_vocab=10,
            d_mlp=128,
            n_ctx=3,
            act_fn="relu",
        )
        return HookedTransformer(cfg)

    def test_returns_tuple_per_matrix(self, model):
        """Each entry is a (U, S, Vt) triple."""
        spectra = compute_weight_spectra(model)
        for name, triple in spectra.items():
            assert len(triple) == 3, f"{name} did not return (U, S, Vt)"

    def test_non_attention_shapes(self, model):
        """Non-attention matrices return 2D U/Vt and 1D S."""
        spectra = compute_weight_spectra(model)
        for name in ("W_E", "W_in", "W_out", "W_U", "W_pos"):
            if name not in spectra:
                continue
            u, s, vt = spectra[name]
            assert u.ndim == 2, f"{name} U should be 2D"
            assert s.ndim == 1, f"{name} S should be 1D"
            assert vt.ndim == 2, f"{name} Vt should be 2D"

    def test_attention_shapes_carry_head_axis(self, model):
        """Attention matrices stack a leading n_heads axis on every component."""
        spectra = compute_weight_spectra(model)
        for name in ATTENTION_MATRICES:
            u, s, vt = spectra[name]
            assert u.ndim == 3, f"{name} U should be 3D (n_heads, m, k)"
            assert s.ndim == 2, f"{name} S should be 2D (n_heads, k)"
            assert vt.ndim == 3, f"{name} Vt should be 3D (n_heads, k, n)"
            assert u.shape[0] == 4
            assert s.shape[0] == 4
            assert vt.shape[0] == 4

    def test_reconstruction(self, model):
        """U @ diag(S) @ Vt reconstructs each weight matrix."""
        from miscope.analysis.library.weights import extract_parameter_snapshot

        snapshot = extract_parameter_snapshot(model)
        spectra = compute_weight_spectra(model)
        for name, (u, s, vt) in spectra.items():
            if name in ATTENTION_MATRICES:
                for h in range(s.shape[0]):
                    recon = u[h] @ np.diag(s[h]) @ vt[h]
                    np.testing.assert_allclose(recon, snapshot[name][h], atol=1e-5)
            else:
                recon = u @ np.diag(s) @ vt
                np.testing.assert_allclose(recon, snapshot[name], atol=1e-5)


# ── Analyzer protocol tests ──────────────────────────────────────────


class TestWeightSpectraAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        assert isinstance(WeightSpectraAnalyzer(), Analyzer)

    def test_has_correct_name(self):
        assert WeightSpectraAnalyzer().name == "weight_spectra"

    def test_registered_in_registry(self):
        from miscope.analysis.analyzers import AnalyzerRegistry

        assert AnalyzerRegistry.is_registered("weight_spectra")

    def test_summary_keys_match_weight_names(self):
        keys = WeightSpectraAnalyzer().get_summary_keys()
        expected = [f"pr_{name}" for name in WEIGHT_MATRIX_NAMES]
        assert keys == expected

    def test_compute_summary_returns_all_pr_keys(self):
        analyzer = WeightSpectraAnalyzer()
        result = {
            f"sv_{name}": np.array([3.0, 2.0, 1.0])
            for name in WEIGHT_MATRIX_NAMES
            if name not in ATTENTION_MATRICES
        }
        for name in ATTENTION_MATRICES:
            result[f"sv_{name}"] = np.array([[3.0, 2.0], [1.0, 1.0]])
        summary = analyzer.compute_summary(result, {})
        for name in WEIGHT_MATRIX_NAMES:
            assert f"pr_{name}" in summary


# ── Integration + parity against effective_dimensionality ────────────


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry_with_family(temp_dirs):
    """Family registered with both analyzers so the pipeline can run them."""
    data_root = temp_dirs
    family_dir = data_root / "modulo_addition_1layer"
    family_dir.mkdir()

    family_json = {
        "name": "modulo_addition_1layer",
        "display_name": "Modulo Addition (1 Layer)",
        "description": "Single-layer transformer for modular arithmetic",
        "architecture": {
            "n_layers": 1,
            "n_heads": 4,
            "d_model": 128,
            "d_head": 32,
            "d_mlp": 512,
            "act_fn": "relu",
            "normalization_type": None,
            "n_ctx": 3,
        },
        "domain_parameters": {
            "prime": {"type": "int", "description": "Modulus", "default": 113},
            "seed": {"type": "int", "description": "Random seed", "default": 999},
        },
        "analyzers": ["effective_dimensionality", "weight_spectra"],
        "visualizations": [],
        "analysis_dataset": {"type": "modulo_addition_grid"},
        "variant_pattern": "p{prime}_seed{seed}",
    }
    with open(family_dir / "family.json", "w") as f:
        json.dump(family_json, f)

    families = discover_families(data_root=data_root)
    return families, data_root


@pytest.fixture
def trained_variant(registry_with_family):
    families, _ = registry_with_family
    family = families["modulo_addition_1layer"]
    variant = family.create_variant({"prime": 17, "seed": 42, "data_seed": 598})
    variant.train(num_epochs=50, checkpoint_epochs=[0, 25, 49], device="cpu")
    return variant


class TestWeightSpectraIntegration:
    """Integration tests + parity against effective_dimensionality."""

    def test_pipeline_creates_artifact(self, trained_variant):
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(WeightSpectraAnalyzer())
        pipeline.run()
        assert os.path.isdir(os.path.join(pipeline.artifacts_dir, "weight_spectra"))

    def test_per_epoch_contains_sv_u_vt_keys(self, trained_variant):
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(WeightSpectraAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epoch_data = loader.load_epoch("weight_spectra", 0)
        for name in ("W_E", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"):
            assert f"sv_{name}" in epoch_data, f"Missing sv_{name}"
            assert f"u_{name}" in epoch_data, f"Missing u_{name}"
            assert f"vt_{name}" in epoch_data, f"Missing vt_{name}"

    def test_summary_contains_pr_keys(self, trained_variant):
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(WeightSpectraAnalyzer())
        pipeline.run()
        summary = ArtifactLoader(pipeline.artifacts_dir).load_summary("weight_spectra")
        assert "epochs" in summary
        for name in WEIGHT_MATRIX_NAMES:
            assert f"pr_{name}" in summary

    def test_parity_singular_values_match_effective_dimensionality(self, trained_variant):
        """sv_{name} per-epoch artifacts agree to ~1e-10 with the legacy analyzer."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.register(WeightSpectraAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        for epoch in (0, 25, 49):
            ed = loader.load_epoch("effective_dimensionality", epoch)
            ws = loader.load_epoch("weight_spectra", epoch)
            for key in ed:
                if not key.startswith("sv_"):
                    continue
                np.testing.assert_allclose(
                    ws[key],
                    ed[key],
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"epoch={epoch} key={key} disagreement",
                )

    def test_parity_pr_summary_matches_effective_dimensionality(self, trained_variant):
        """pr_{name} summary agrees between weight_spectra and effective_dimensionality."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.register(WeightSpectraAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        ed = loader.load_summary("effective_dimensionality")
        ws = loader.load_summary("weight_spectra")
        for name in WEIGHT_MATRIX_NAMES:
            key = f"pr_{name}"
            if key in ed:
                np.testing.assert_allclose(
                    ws[key],
                    ed[key],
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"summary {key} disagreement",
                )
