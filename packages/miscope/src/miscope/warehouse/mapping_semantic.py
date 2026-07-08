"""Semantic-table claims: which analyzer fields feed the designed REQ_110 tables.

A *claim* pulls a named set of an analyzer's columnar fields into one of the
designed semantic tables (``pca_results``, ``shape_characterizations``,
``neuron_frequency_attribution``, ``frequency_spectrum``),
reshaping them into that table's conformed schema and stamping the
``group_type``/``operation_type`` discriminators. Fields no claim takes fall
through to the generic analyzer-named table (REQ_110A: designed tables + generic
fallback).

Each reshaper receives the *already-flattened* long frames for its claimed fields
(coord columns + one value column each) and returns rows in the semantic schema.
Rows from every analyzer feeding a table are row-unioned by the writer.

Scope note (110-A): ``pca_projections`` and the bulk of ``frequency_spectrum``
magnitude come from ``tensor``-kind fields (projection coordinates, Fourier
coefficient cubes) owned by 110-B; they are not populated from the columnar plane
here. ``frequency_spectrum`` carries the one columnar magnitude feeder
(``gradient_site.energy``) for when that windowed analyzer is regenerated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce

import pandas as pd

from miscope.warehouse.schema import GROUP_TYPE, OPERATION_TYPE, GroupType, OperationType

# A reshaper maps {field_name: long_frame} -> semantic rows.
Reshaper = Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


@dataclass(frozen=True)
class Claim:
    """One analyzer's contribution to a semantic table."""

    table: str
    fields: tuple[str, ...]
    reshaper: Reshaper


def _merge_claimed(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Horizontal-merge claimed field frames on their shared coord columns."""
    ordered = list(frames.values())
    if len(ordered) == 1:
        return ordered[0].copy()

    def _join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        on = [c for c in left.columns if c in right.columns]
        if on:
            return left.merge(right, on=on, how="outer")
        return pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)

    return reduce(_join, ordered)


# ---------------------------------------------------------------------------
# pca_results — explained-variance family (pc_index = the row_id axis)
# ---------------------------------------------------------------------------


def _pca_results(
    group_type: GroupType,
    operation_type: OperationType,
    *,
    group_col: str | None,
    group_const: str | None = None,
    value_map: dict[str, str],
) -> Reshaper:
    """Reshape explained-variance fields into ``pca_results`` rows."""

    def fn(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        merged = _merge_claimed(frames)
        out = pd.DataFrame()
        if "epoch" in merged.columns:
            out["epoch"] = merged["epoch"]
        out["group"] = merged[group_col] if group_col else group_const
        out[GROUP_TYPE] = group_type.value
        out[OPERATION_TYPE] = operation_type.value
        out["pc_index"] = merged["row_id"]
        for src, dst in value_map.items():
            out[dst] = merged[src]
        return out

    return fn


def _pca_var_pcs() -> Reshaper:
    """repr_geometry's three scalar ``pca_var_pc{1,2,3}`` -> 3 pca_results rows/(epoch,site)."""

    def fn(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for pc, name in enumerate(("pca_var_pc1", "pca_var_pc2", "pca_var_pc3")):
            f = frames.get(name)
            if f is None:
                continue
            out = pd.DataFrame()
            out["epoch"] = f["epoch"]
            out["group"] = f["site"]
            out[GROUP_TYPE] = GroupType.CENTROID_GROUP.value
            out[OPERATION_TYPE] = OperationType.PCA.value
            out["pc_index"] = pc
            out["explained_variance_ratio"] = f[name]
            rows.append(out)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return fn


# ---------------------------------------------------------------------------
# shape_characterizations — melt named characterizations into (operation_type, value)
# ---------------------------------------------------------------------------


def _shape_melt(group_type: GroupType, group_col: str | None) -> Reshaper:
    """Melt each claimed scalar field into a (operation_type=field, value) row set."""

    def fn(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for name, f in frames.items():
            out = pd.DataFrame()
            for coord in ("epoch", "site", "group"):
                if coord in f.columns:
                    out[coord] = f[coord]
            if "group" not in out.columns:
                out["group"] = f[group_col] if group_col and group_col in f.columns else None
            out[GROUP_TYPE] = group_type.value
            out[OPERATION_TYPE] = name
            out["value"] = f[name]
            rows.append(out)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return fn


# ---------------------------------------------------------------------------
# neuron_frequency_attribution — neuron_dynamics dominant rows
# ---------------------------------------------------------------------------


def _neuron_freq_attribution() -> Reshaper:
    def fn(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        merged = _merge_claimed(frames)  # epoch, neuron, dominant_freq, max_frac
        out = pd.DataFrame()
        out["epoch"] = merged["epoch"]
        out["neuron"] = merged["neuron"]
        out["frequency"] = merged["dominant_freq"]
        out["frac_explained"] = merged["max_frac"]
        out["dominant"] = True
        return out

    return fn


# ---------------------------------------------------------------------------
# frequency_spectrum — gradient_site per-frequency energy (windowed; absent on canon)
# ---------------------------------------------------------------------------


def _frequency_spectrum_energy() -> Reshaper:
    def fn(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        f = frames["energy"]  # epoch, site, frequency, energy
        out = pd.DataFrame()
        for coord in ("epoch", "site", "frequency"):
            out[coord] = f[coord]
        out["magnitude"] = f["energy"]
        out["derivation"] = "gradient_site.energy"
        return out

    return fn


# ---------------------------------------------------------------------------
# circuit_spectra — Layer 4 circuit object table (REQ_156)
# ---------------------------------------------------------------------------


def _circuit_spectra_object() -> Reshaper:
    """Conform per-head spectral metrics into the circuit object table.

    The three metrics share the ``(epoch, site, head)`` key, so they horizontal-merge
    into one row per circuit (``site`` is the circuit discriminator). Weight-derived →
    ``group_type=weight_matrix``. Keeps the metrics as columns (the data model's
    Layer 4 object shape), not melted.
    """
    metrics = ("copying_score", "effective_rank", "operator_norm")

    def fn(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        merged = _merge_claimed(frames)
        out = pd.DataFrame()
        for coord in ("epoch", "site", "head"):
            if coord in merged.columns:
                out[coord] = merged[coord]
        out[GROUP_TYPE] = GroupType.WEIGHT_MATRIX.value
        for metric in metrics:
            if metric in merged.columns:
                out[metric] = merged[metric]
        return out

    return fn


# ---------------------------------------------------------------------------
# Claims registry: analyzer -> the semantic claims it satisfies.
# ---------------------------------------------------------------------------

CLAIMS: dict[str, list[Claim]] = {
    "global_centroid_pca": [
        Claim(
            "pca_results",
            ("explained_variance_ratio",),
            _pca_results(
                GroupType.CENTROID_GROUP,
                OperationType.PCA_SUMMARY,
                group_col="site",
                value_map={"explained_variance_ratio": "explained_variance_ratio"},
            ),
        ),
    ],
    "parameter_trajectory": [
        Claim(
            "pca_results",
            ("explained_variance", "explained_variance_ratio"),
            _pca_results(
                GroupType.WEIGHT_COMPONENT,
                OperationType.PCA_SUMMARY,
                group_col="group",
                value_map={
                    "explained_variance": "explained_variance",
                    "explained_variance_ratio": "explained_variance_ratio",
                },
            ),
        ),
    ],
    "neuron_group_pca": [
        Claim(
            "pca_results",
            ("pc_var",),
            _pca_results(
                GroupType.FREQUENCY_GROUP,
                OperationType.PCA,
                group_col="group",
                value_map={"pc_var": "explained_variance"},
            ),
        ),
        Claim(
            "pca_results",
            ("centroid_pca_var",),
            _pca_results(
                GroupType.CENTROID_GROUP,
                OperationType.PCA_SUMMARY,
                group_col=None,
                group_const="centroids",
                value_map={"centroid_pca_var": "explained_variance"},
            ),
        ),
    ],
    "repr_geometry": [
        Claim("pca_results", ("pca_var_pc1", "pca_var_pc2", "pca_var_pc3"), _pca_var_pcs()),
        Claim(
            "shape_characterizations",
            ("circularity",),
            _shape_melt(GroupType.CENTROID_GROUP, group_col="site"),
        ),
    ],
    "freq_group_weight_geometry": [
        Claim(
            "shape_characterizations",
            ("circularity",),
            # Weight-side geometry (WeightMatrix in the data model, Part VI #1):
            # this is intrinsic to the parameter tensors, NOT a probe-relative
            # activation reading. Stamp WEIGHT_MATRIX so it never collides with
            # repr_geometry's activation-side circularity (CENTROID_GROUP) in
            # shape_characterizations.
            _shape_melt(GroupType.WEIGHT_MATRIX, group_col="site"),
        ),
    ],
    "centroid_fourier_alignment": [
        Claim(
            "shape_characterizations",
            ("fourier_alignment",),
            _shape_melt(GroupType.ACTIVATION_SITE, group_col="site"),
        ),
    ],
    # REQ_141 (bucket-1): the per-epoch attribution table is now sourced from the
    # per-epoch neuron_frequency_attribution analyzer, not the cross-epoch
    # neuron_dynamics stack. neuron_dynamics keeps only its cross-epoch tail
    # (switch_counts/commitment_epochs/threshold), which falls through to its
    # generic analyzer-named table (no semantic claim).
    "neuron_frequency_attribution": [
        Claim(
            "neuron_frequency_attribution",
            ("dominant_freq", "max_frac"),
            _neuron_freq_attribution(),
        ),
    ],
    "gradient_site": [
        Claim("frequency_spectrum", ("energy",), _frequency_spectrum_energy()),
    ],
    # REQ_156: drain the generic fallback — the three stable named spectral metrics
    # become columns on the conformed circuit object table, keyed by the circuit
    # `site` discriminator. Future per-circuit attributes (OVCircuit logit_attribution,
    # QKCircuit operand_symmetry) claim into this same table.
    "circuit_spectra": [
        Claim(
            "circuit_spectra",
            ("copying_score", "effective_rank", "operator_norm"),
            _circuit_spectra_object(),
        ),
    ],
}


def claims_for(analyzer_name: str) -> list[Claim]:
    """Semantic claims an analyzer satisfies (empty if it feeds no designed table)."""
    return CLAIMS.get(analyzer_name, [])
