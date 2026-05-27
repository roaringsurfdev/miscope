"""REQ_126 PR 3: CentroidFourierAlignmentAnalyzer tests + ``repr_geometry`` defusion.

Two layers:

1. Verify the field is gone — ``repr_geometry`` no longer emits
   ``*_fourier_alignment`` (defusion is complete on the analyzer side).
2. Verify the signal is preserved — the new secondary analyzer's output
   matches the values the legacy fused field used to produce on canon.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miscope.analysis.analyzers.centroid_fourier_alignment import (
    CentroidFourierAlignmentAnalyzer,
)
from miscope.analysis.inputs import ResolvedInputs

CANON_VARIANT_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "modulo_addition_1layer"
    / "variants"
    / "p113_seed999_dseed598"
)
PARITY_EPOCH = 24999
PRIME = 113
PARITY_RTOL = 1e-3
PARITY_ATOL = 1e-5

SITES = ("resid_pre", "attn_out", "mlp_out", "resid_post")


def _load_legacy_repr_geometry(epoch: int) -> dict[str, np.ndarray]:
    path = CANON_VARIANT_DIR / "artifacts" / "repr_geometry" / f"epoch_{epoch:05d}.npz"
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _canon_available() -> bool:
    path = CANON_VARIANT_DIR / "artifacts" / "repr_geometry" / f"epoch_{PARITY_EPOCH:05d}.npz"
    return path.is_file()


skip_no_canon = pytest.mark.skipif(
    not _canon_available(), reason="Canon repr_geometry artifact not present"
)


# ---------------------------------------------------------------------------
# Defusion: ``repr_geometry`` no longer outputs ``*_fourier_alignment``
# ---------------------------------------------------------------------------


def test_repr_geometry_no_longer_emits_fourier_alignment_keys():
    """The defusion is complete — ``_SCALAR_KEYS`` does not list ``fourier_alignment``."""
    from miscope.analysis.analyzers.repr_geometry import _SCALAR_KEYS

    assert "fourier_alignment" not in _SCALAR_KEYS
    assert "circularity" in _SCALAR_KEYS  # geometric, stays per Q3


def test_repr_geometry_source_no_longer_imports_characterize_fourier_alignment():
    """Audit: the defused module no longer references the Fourier-alignment primitive."""
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "miscope"
        / "analysis"
        / "analyzers"
        / "repr_geometry.py"
    )
    src = src_path.read_text()
    assert "characterize_fourier_alignment" not in src


# ---------------------------------------------------------------------------
# Parity: new analyzer reproduces the legacy fused values
# ---------------------------------------------------------------------------


@skip_no_canon
def test_parity_centroid_fourier_alignment_on_canon():
    """New analyzer reproduces canon's legacy ``*_fourier_alignment`` per site."""
    legacy = _load_legacy_repr_geometry(PARITY_EPOCH)
    analyzer = CentroidFourierAlignmentAnalyzer()
    inputs = ResolvedInputs(artifacts={"repr_geometry": legacy}, epoch=PARITY_EPOCH)
    context = {"params": {"prime": PRIME}}
    result = analyzer.analyze(inputs, context)

    # Every site that produced centroids in legacy should produce alignment now.
    for site in SITES:
        legacy_key = f"{site}_fourier_alignment"
        new_key = f"{site}_fourier_alignment"
        if legacy_key not in legacy:
            continue  # site not present in legacy (architecture-dependent)
        assert new_key in result, f"Missing site {site} in new output"
        np.testing.assert_allclose(
            float(result[new_key]),
            float(legacy[legacy_key]),
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg=f"Mismatch at site {site}",
        )


# ---------------------------------------------------------------------------
# Unit tests on synthetic centroids
# ---------------------------------------------------------------------------


def test_analyzer_iterates_only_centroid_keys():
    """Sites are discovered by ``*_centroids`` suffix; other upstream keys are ignored."""
    p = 13
    rng = np.random.default_rng(0)
    centroids = rng.standard_normal((p, 8))
    upstream = {
        "mlp_out_centroids": centroids,
        "mlp_out_mean_radius": np.float64(0.5),
        "mlp_out_pca_var_pc1": np.float64(0.4),
        # ``attn_out_centroids`` absent — should not appear in output.
    }
    analyzer = CentroidFourierAlignmentAnalyzer()
    inputs = ResolvedInputs(artifacts={"repr_geometry": upstream})
    result = analyzer.analyze(inputs, {"params": {"prime": p}})
    assert set(result.keys()) == {"mlp_out_fourier_alignment"}


def test_summary_passes_per_site_scalars_through():
    """``compute_summary`` returns the per-site scalars unchanged for the collector."""
    p = 13
    rng = np.random.default_rng(1)
    upstream = {"resid_post_centroids": rng.standard_normal((p, 8))}
    analyzer = CentroidFourierAlignmentAnalyzer()
    inputs = ResolvedInputs(artifacts={"repr_geometry": upstream})
    context = {"params": {"prime": p}}
    result = analyzer.analyze(inputs, context)
    summary = analyzer.compute_summary(result, context)
    assert set(summary.keys()) == {"resid_post_fourier_alignment"}
    assert isinstance(summary["resid_post_fourier_alignment"], float)
