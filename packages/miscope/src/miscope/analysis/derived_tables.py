"""Derived-table declarations (REQ_141) — collected by ``miscope.registry``.

Importing this module registers every derived table via
:func:`~miscope.analysis.derived_table.register_derived_table` side effects, the
same import-side-effect pattern the analyzer and DataView surfaces use. The
registry's ``build_index`` imports this module so the index sees the full set.

Declarations are pure data (SQL text over warehouse table names + an output
schema); the executor that materializes a spec lives in
:mod:`miscope.warehouse.derived`.

The neuron-frequency vertical slice (REQ_141 bucket-2): the cross-epoch
``transient_frequency`` analyzer — a pure aggregation over the conformed
``neuron_frequency_attribution`` table — is expressed as three derived tables:

- ``committed_counts`` — committed-neuron count per ``(epoch, frequency)`` (the
  bucket-2 keystone: ``COUNT(*) ... WHERE frac_explained >= 0.70 GROUP BY``).
- ``transient_frequencies`` — per ever-qualified frequency: peak epoch/count,
  whether it survives to the final epoch, and the homeless-neuron count.
- ``transient_peak_members`` — the ragged peak-cohort membership, flattened to a
  ``(frequency, member_neuron)`` long table (the fork-2 / 2B decision).
"""

from __future__ import annotations

from miscope.analysis.derived_table import DerivedTableSpec, register_derived_table
from miscope.analysis.output_schema import Coord
from miscope.analysis.output_schema import OutputField as F

# Thresholds carried over verbatim from the retired transient_frequency analyzer
# so the derived outputs match value-for-value (REQ_141 validation).
_NEURON_THRESHOLD = 0.70  # per-neuron max_frac commitment gate
_TRANSIENT_FRACTION = 0.05  # fraction of d_mlp for "ever qualified"
_FINAL_FRACTION = 0.10  # fraction of d_mlp for "still canonical at the final epoch"

ATTRIBUTION = "neuron_frequency_attribution"
COMMITTED_COUNTS = "committed_counts"


# ---------------------------------------------------------------------------
# committed_counts — committed-neuron count per (epoch, frequency).
# ---------------------------------------------------------------------------

COMMITTED_COUNTS_TABLE = register_derived_table(
    DerivedTableSpec(
        name=COMMITTED_COUNTS,
        query=f"""
            SELECT epoch, frequency, COUNT(*) AS committed_counts
            FROM {ATTRIBUTION}
            WHERE frac_explained >= {_NEURON_THRESHOLD}
            GROUP BY epoch, frequency
            ORDER BY epoch, frequency
        """,
        input_tables=(ATTRIBUTION,),
        outputs=(
            F.columnar(
                "committed_counts",
                "int64",
                (Coord.VARIANT, Coord.EPOCH, Coord.FREQUENCY),
                "Neurons committed (frac_explained >= 0.70) to each frequency per epoch.",
            ),
        ),
        materialized=True,
    )
)


# Shared CTE prelude: d_mlp, per-frequency peak count + earliest peak epoch, and
# the ever-qualified set. MIN(epoch) at the peak count matches numpy argmax's
# first-occurrence tie-break (epochs ascending) for byte-parity with the old npz.
_TRANSIENT_PRELUDE = f"""
    WITH d_mlp AS (
        SELECT COUNT(DISTINCT neuron) AS n FROM {ATTRIBUTION}
    ),
    final_epoch AS (
        SELECT MAX(epoch) AS e FROM {ATTRIBUTION}
    ),
    peak AS (
        SELECT frequency, MAX(committed_counts) AS peak_count
        FROM {COMMITTED_COUNTS}
        GROUP BY frequency
    ),
    peak_epoch AS (
        SELECT cc.frequency, MIN(cc.epoch) AS peak_epoch
        FROM {COMMITTED_COUNTS} cc
        JOIN peak p ON cc.frequency = p.frequency AND cc.committed_counts = p.peak_count
        GROUP BY cc.frequency
    ),
    ever AS (
        SELECT p.frequency, p.peak_count, pe.peak_epoch
        FROM peak p
        JOIN peak_epoch pe USING (frequency)
        WHERE p.peak_count >= {_TRANSIENT_FRACTION} * (SELECT n FROM d_mlp)
    )
"""


# ---------------------------------------------------------------------------
# transient_frequencies — per ever-qualified frequency summary.
# ---------------------------------------------------------------------------

TRANSIENT_FREQUENCIES_TABLE = register_derived_table(
    DerivedTableSpec(
        name="transient_frequencies",
        query=f"""
            {_TRANSIENT_PRELUDE},
            final_committed AS (
                SELECT frequency, COUNT(*) AS final_count
                FROM {ATTRIBUTION}
                WHERE frac_explained >= {_NEURON_THRESHOLD}
                  AND epoch = (SELECT e FROM final_epoch)
                GROUP BY frequency
            ),
            members AS (
                SELECT e.frequency, a.neuron
                FROM ever e
                JOIN {ATTRIBUTION} a
                  ON a.epoch = e.peak_epoch
                 AND a.frequency = e.frequency
                 AND a.frac_explained >= {_NEURON_THRESHOLD}
            ),
            final_frac AS (
                SELECT neuron, frac_explained AS final_frac
                FROM {ATTRIBUTION}
                WHERE epoch = (SELECT e FROM final_epoch)
            ),
            homeless AS (
                SELECT m.frequency,
                       COUNT(*) FILTER (WHERE ff.final_frac < {_NEURON_THRESHOLD}) AS homeless_count
                FROM members m
                LEFT JOIN final_frac ff USING (neuron)
                GROUP BY m.frequency
            )
            SELECT e.frequency,
                   e.peak_epoch,
                   e.peak_count,
                   (COALESCE(fc.final_count, 0) >= {_FINAL_FRACTION} * (SELECT n FROM d_mlp))
                       AS is_final,
                   COALESCE(h.homeless_count, 0) AS homeless_count
            FROM ever e
            LEFT JOIN final_committed fc USING (frequency)
            LEFT JOIN homeless h USING (frequency)
            ORDER BY e.frequency
        """,
        input_tables=(ATTRIBUTION, COMMITTED_COUNTS),
        outputs=(
            F.columnar(
                "peak_epoch",
                "int64",
                (Coord.VARIANT, Coord.FREQUENCY),
                "Epoch of peak committed-neuron count for each transient frequency.",
            ),
            F.columnar(
                "peak_count",
                "int64",
                (Coord.VARIANT, Coord.FREQUENCY),
                "Committed-neuron count at the peak epoch for each transient frequency.",
            ),
            F.columnar(
                "is_final",
                "bool",
                (Coord.VARIANT, Coord.FREQUENCY),
                "Whether a transient frequency is still canonical at the final epoch.",
            ),
            F.columnar(
                "homeless_count",
                "int64",
                (Coord.VARIANT, Coord.FREQUENCY),
                "Peak-cohort neurons that abandoned the frequency without re-homing.",
            ),
        ),
        materialized=True,
    )
)


# ---------------------------------------------------------------------------
# transient_peak_members — ragged peak cohort flattened to a long table (fork 2B).
# ---------------------------------------------------------------------------

TRANSIENT_PEAK_MEMBERS_TABLE = register_derived_table(
    DerivedTableSpec(
        name="transient_peak_members",
        query=f"""
            {_TRANSIENT_PRELUDE}
            SELECT e.frequency, a.neuron AS member_neuron
            FROM ever e
            JOIN {ATTRIBUTION} a
              ON a.epoch = e.peak_epoch
             AND a.frequency = e.frequency
             AND a.frac_explained >= {_NEURON_THRESHOLD}
            ORDER BY e.frequency, a.neuron
        """,
        input_tables=(ATTRIBUTION, COMMITTED_COUNTS),
        outputs=(
            F.columnar(
                "member_neuron",
                "int64",
                (Coord.VARIANT, Coord.FREQUENCY),
                "Neuron in a transient frequency's peak-epoch committed cohort (one row each).",
            ),
        ),
        materialized=True,
    )
)
