"""REQ_156: conformed circuit_spectra semantic claim.

The claim conforms circuit_spectra's three per-head spectral metrics into one
object-table row per (epoch, site, head), stamped group_type=weight_matrix.
"""

from __future__ import annotations

import pandas as pd

from miscope.warehouse.mapping_semantic import claims_for
from miscope.warehouse.schema import GROUP_TYPE, NATURAL_WIDE_INDEX, GroupType


def _metric_frame(metric: str) -> pd.DataFrame:
    # Two circuits × two heads, keyed (epoch, site, head) — the flattened shape the
    # writer hands a reshaper (coord columns + one value column).
    rows = []
    for site in ("full_ov", "qk"):
        for head in (0, 1):
            rows.append({"epoch": 100, "site": site, "head": head, metric: 0.1 * head + 0.5})
    return pd.DataFrame(rows)


def test_claim_registered_for_circuit_spectra():
    claims = claims_for("circuit_spectra")
    assert len(claims) == 1
    assert claims[0].table == "circuit_spectra"
    assert set(claims[0].fields) == {"copying_score", "effective_rank", "operator_norm"}


def test_reshaper_conforms_metrics_to_object_rows():
    claim = claims_for("circuit_spectra")[0]
    frames = {m: _metric_frame(m) for m in ("copying_score", "effective_rank", "operator_norm")}
    out = claim.reshaper(frames)

    # One row per (epoch, site, head) with all three metrics as columns.
    assert len(out) == 4
    assert set(out.columns) == {
        "epoch",
        "site",
        "head",
        GROUP_TYPE,
        "copying_score",
        "effective_rank",
        "operator_norm",
    }
    assert set(out[GROUP_TYPE].unique()) == {GroupType.WEIGHT_MATRIX.value}
    assert sorted(out["site"].unique()) == ["full_ov", "qk"]
    # Values survive the horizontal merge (head 1 → 0.6, head 0 → 0.5).
    row = out[(out["site"] == "full_ov") & (out["head"] == 1)].iloc[0]
    assert row["copying_score"] == 0.6
    assert row["operator_norm"] == 0.6


def test_natural_wide_index_registered():
    assert NATURAL_WIDE_INDEX["circuit_spectra"] == ("variant_id", "epoch", "site", "head")
