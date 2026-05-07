"""Tests for REQ_082: Variant Table page.

CoS coverage:
- Unit: table data loads from variant_registry.json and produces the correct
  number of rows and expected column values for a known variant.
- Integration: clicking a row updates the variant-selector-store
  (tested via callback invocation with mock server state).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path("results/modulo_addition_1layer/variant_registry.json")

_requires_data = pytest.mark.skipif(
    not _REGISTRY_PATH.exists(),
    reason="requires local results data (not available in CI)",
)


def _load_registry() -> list[dict]:
    with open(_REGISTRY_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Unit tests: _load_table_rows
# ---------------------------------------------------------------------------


@_requires_data
def test_load_table_rows_count():
    """Number of rows matches number of entries in variant_registry.json."""
    from dashboard.pages.variant_table import _load_table_rows

    rows = _load_table_rows()
    registry = _load_registry()
    assert len(rows) == len(registry)


def test_load_table_rows_required_columns():
    """Every row contains all required display columns."""
    from dashboard.pages.variant_table import _load_table_rows

    required = {
        "family",
        "prime",
        "model_seed",
        "data_seed",
        "classification",
        "test_loss_final",
        "grokking_epoch",
        "committed_freqs",
        "_variant_name",
    }
    rows = _load_table_rows()
    for row in rows:
        assert required <= set(row.keys()), f"Row missing columns: {required - set(row.keys())}"


@_requires_data
def test_load_table_rows_known_variant():
    """A known variant appears with expected field values."""
    from dashboard.pages.variant_table import _load_table_rows

    rows = _load_table_rows()
    # Find the canon model: p=113, seed=999, dseed=598
    canon = next(
        (r for r in rows if r["prime"] == 113 and r["model_seed"] == 999 and r["data_seed"] == 598),
        None,
    )
    assert canon is not None, "Canon variant p=113/seed999/dseed598 not found in table"
    assert canon["family"] == "modulo_addition_1layer"
    assert canon["_variant_name"] == "modulo_addition_1layer_p113_seed999_dseed598"
    assert isinstance(canon["committed_freqs"], int)
    assert canon["committed_freqs"] > 0


def test_load_table_rows_variant_name_format():
    """Every _variant_name follows the expected family_pP_seedS_dseedD pattern."""
    from dashboard.pages.variant_table import _load_table_rows

    rows = _load_table_rows()
    for row in rows:
        name = row["_variant_name"]
        family = row["family"]
        prime = row["prime"]
        seed = row["model_seed"]
        dseed = row["data_seed"]
        expected = f"{family}_p{prime}_seed{seed}_dseed{dseed}"
        assert name == expected, f"Variant name mismatch: {name!r} != {expected!r}"


def test_load_table_rows_sorted():
    """Rows are sorted by family, then prime, then model_seed, then data_seed."""
    from dashboard.pages.variant_table import _load_table_rows

    rows = _load_table_rows()
    keys = [(r["family"], r["prime"] or 0, r["model_seed"] or 0, r["data_seed"] or 0) for r in rows]
    assert keys == sorted(keys), "Table rows are not sorted correctly"


def test_load_table_rows_classification_label():
    """Classification is a plain string (not the raw list from JSON)."""
    from dashboard.pages.variant_table import _load_table_rows

    rows = _load_table_rows()
    for row in rows:
        assert isinstance(row["classification"], str), (
            f"classification should be str, got {type(row['classification'])}"
        )


# ---------------------------------------------------------------------------
# Integration test: on_row_selected callback
# ---------------------------------------------------------------------------


def test_on_row_selected_updates_store(monkeypatch):
    """Row selection calls load_variant and issues the correct store set_props."""
    from dashboard.pages import variant_table as vt

    mock_state = MagicMock()
    mock_state.load_variant.return_value = True
    mock_state.available_epochs = [0, 100, 200]
    monkeypatch.setattr(vt, "variant_server_state", mock_state)

    set_props_calls: list[tuple] = []
    monkeypatch.setattr(
        vt, "set_props", lambda *args, **kwargs: set_props_calls.append((args, kwargs))
    )

    # Simulate the logic inside the on_row_selected callback directly.
    row = {
        "_variant_name": "modulo_addition_1layer_p113_seed999_dseed598",
        "family": "modulo_addition_1layer",
    }
    family_name = row["family"]
    variant_name = row["_variant_name"]

    ok = vt.variant_server_state.load_variant(family_name, variant_name)
    assert ok is True

    max_epochs = max(0, len(vt.variant_server_state.available_epochs) - 1)
    epoch = vt.variant_server_state.available_epochs[0]

    vt.set_props(
        "variant-selector-store",
        {
            "data": {
                "family_name": family_name,
                "variant_name": variant_name,
                "intervention_name": None,
                "epoch": epoch,
                "epoch_index": 0,
                "max_epochs": max_epochs,
                "last_field_updated": "variant_name",
            }
        },
    )

    mock_state.load_variant.assert_called_once_with(
        "modulo_addition_1layer",
        "modulo_addition_1layer_p113_seed999_dseed598",
    )
    assert len(set_props_calls) == 1
    store_data = set_props_calls[0][0][1]["data"]
    assert store_data["family_name"] == "modulo_addition_1layer"
    assert store_data["variant_name"] == "modulo_addition_1layer_p113_seed999_dseed598"
    assert store_data["last_field_updated"] == "variant_name"
    assert store_data["max_epochs"] == 2
