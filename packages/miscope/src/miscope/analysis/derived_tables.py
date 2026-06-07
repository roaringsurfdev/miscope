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

REQ_144 adds the variant-outcome layer's conformed inputs and rollups as further
derived tables (declared below their own header):

- ``participation_ratios`` — per ``(epoch, site, head)`` weight-matrix participation
  ratio ``(Σσ)²/Σσ²`` over the ``weight_spectra`` singular values, so the summary
  engine reads ``pr_W_*`` as a queryable fact instead of an analyzer summary npz.
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


# ===========================================================================
# REQ_144 — variant-outcome layer conformed inputs
# ===========================================================================

WEIGHT_SPECTRA = "weight_spectra"


# ---------------------------------------------------------------------------
# participation_ratios — per (epoch, site, head) weight-matrix participation ratio.
#
# PR = (Σσ)²/Σσ² over a matrix's singular values (``compute_participation_ratio``),
# expressed as a reduction over the ``weight_spectra`` ``sv`` rows. This replaces the
# summary engine's ``load_summary("weight_spectra")`` read of ``pr_W_*`` with a
# conformed warehouse fact (REQ_144 CoS #3). Computed for every site/head; the
# non-attention matrices the engine consumes (W_E/W_in/W_out) sit at head=0.
# ---------------------------------------------------------------------------

PARTICIPATION_RATIOS_TABLE = register_derived_table(
    DerivedTableSpec(
        name="participation_ratios",
        query=f"""
            SELECT epoch, site, head,
                   pow(sum(sv), 2) / sum(sv * sv) AS participation_ratio
            FROM {WEIGHT_SPECTRA}
            GROUP BY epoch, site, head
            ORDER BY epoch, site, head
        """,
        input_tables=(WEIGHT_SPECTRA,),
        outputs=(
            F.columnar(
                "participation_ratio",
                "float64",
                (Coord.VARIANT, Coord.EPOCH, Coord.SITE, Coord.HEAD),
                "Participation ratio (Σσ)²/Σσ² of a weight matrix's singular values "
                "per epoch (head = attention head; head=0 for non-attention sites).",
            ),
        ),
        materialized=True,
    )
)


# ---------------------------------------------------------------------------
# loss_outcomes — the stable loss-curve outcome scalars (one wide row per variant).
#
# A bucket-2 reduction over the dense `losses` table that reproduces the summary
# engine's `_load_train_test_loss_metrics` value-for-value: loss extrema + their
# (first-occurrence) argmin/argmax epochs, threshold-crossing epochs, finals, and
# the second-descent onset (first epoch at/after the test-loss peak where the
# descent fraction clears the onset threshold) + survival. `epoch` equals the dense
# list index, so SQL argmin/argmax with an `epoch` tie-break matches numpy's
# first-occurrence semantics exactly.
# ---------------------------------------------------------------------------

LOSSES = "losses"

# Thresholds carried verbatim from variant_analysis_summary so the outputs match.
_FIRST_DESCENT_TRAIN_LOSS_THRESHOLD = 1.0e-6
_SECOND_DESCENT_TEST_LOSS_THRESHOLD = 1.0e-6
_SECOND_DESCENT_ONSET_DIFF_THRESHOLD = 0.8
_SUCCESSFUL_TEST_LOSS_THRESHOLD = 1.0e-5

LOSS_OUTCOMES_TABLE = register_derived_table(
    DerivedTableSpec(
        name="loss_outcomes",
        query=f"""
            WITH bounds AS (
                SELECT MAX(epoch) AS last_epoch,
                       MIN(train_loss) AS train_loss_min,
                       MIN(test_loss) AS test_loss_min,
                       MAX(test_loss) AS test_loss_max
                FROM {LOSSES}
            ),
            argmin_train AS (SELECT epoch AS e FROM {LOSSES} ORDER BY train_loss, epoch LIMIT 1),
            argmin_test AS (SELECT epoch AS e FROM {LOSSES} ORDER BY test_loss, epoch LIMIT 1),
            argmax_test AS (SELECT epoch AS e FROM {LOSSES} ORDER BY test_loss DESC, epoch LIMIT 1),
            finals AS (
                SELECT train_loss AS train_loss_final, test_loss AS test_loss_final
                FROM {LOSSES} WHERE epoch = (SELECT last_epoch FROM bounds)
            ),
            onset AS (
                SELECT MIN(l.epoch) AS onset_epoch
                FROM {LOSSES} l, bounds b, argmax_test a
                WHERE l.epoch >= a.e
                  AND (b.test_loss_max - l.test_loss) / b.test_loss_max
                      >= {_SECOND_DESCENT_ONSET_DIFF_THRESHOLD}
            )
            SELECT
                b.train_loss_min,
                (SELECT e FROM argmin_train) AS train_loss_min_epoch,
                COALESCE(
                    (SELECT MIN(epoch) FROM {LOSSES}
                     WHERE train_loss <= {_FIRST_DESCENT_TRAIN_LOSS_THRESHOLD}), -1
                ) AS train_loss_threshold_first_epoch,
                f.train_loss_final,
                b.test_loss_min,
                (SELECT e FROM argmin_test) AS test_loss_min_epoch,
                b.test_loss_max,
                (SELECT e FROM argmax_test) AS test_loss_max_epoch,
                (SELECT e FROM argmax_test) AS peak_test_loss_epoch,
                COALESCE(
                    (SELECT MIN(epoch) FROM {LOSSES}
                     WHERE test_loss <= {_SECOND_DESCENT_TEST_LOSS_THRESHOLD}), -1
                ) AS test_loss_threshold_first_epoch,
                f.test_loss_final,
                f.test_loss_final AS final_test_loss,
                o.onset_epoch AS second_descent_onset_epoch,
                CASE WHEN o.onset_epoch IS NOT NULL
                     THEN (f.test_loss_final <= {_SUCCESSFUL_TEST_LOSS_THRESHOLD})
                     ELSE NULL END AS second_descent_survived
            FROM bounds b, finals f, onset o
        """,
        input_tables=(LOSSES,),
        outputs=(
            F.columnar("train_loss_min", "float64", (Coord.VARIANT,), "Minimum training loss."),
            F.columnar(
                "train_loss_min_epoch",
                "int64",
                (Coord.VARIANT,),
                "Epoch (dense index) of minimum training loss (first occurrence).",
            ),
            F.columnar(
                "train_loss_threshold_first_epoch",
                "int64",
                (Coord.VARIANT,),
                "First epoch train loss crosses the first-descent threshold (-1 if never).",
            ),
            F.columnar(
                "train_loss_final", "float64", (Coord.VARIANT,), "Final-epoch training loss."
            ),
            F.columnar("test_loss_min", "float64", (Coord.VARIANT,), "Minimum test loss."),
            F.columnar(
                "test_loss_min_epoch",
                "int64",
                (Coord.VARIANT,),
                "Epoch of minimum test loss (first occurrence).",
            ),
            F.columnar("test_loss_max", "float64", (Coord.VARIANT,), "Maximum test loss."),
            F.columnar(
                "test_loss_max_epoch",
                "int64",
                (Coord.VARIANT,),
                "Epoch of maximum (peak) test loss (first occurrence).",
            ),
            F.columnar(
                "peak_test_loss_epoch",
                "int64",
                (Coord.VARIANT,),
                "Alias of test_loss_max_epoch (the test-loss peak).",
            ),
            F.columnar(
                "test_loss_threshold_first_epoch",
                "int64",
                (Coord.VARIANT,),
                "First epoch test loss crosses the second-descent threshold (-1 if never).",
            ),
            F.columnar("test_loss_final", "float64", (Coord.VARIANT,), "Final-epoch test loss."),
            F.columnar("final_test_loss", "float64", (Coord.VARIANT,), "Alias of test_loss_final."),
            F.columnar(
                "second_descent_onset_epoch",
                "int64",
                (Coord.VARIANT,),
                "First epoch at/after the test-loss peak whose descent fraction clears the "
                "onset threshold (null if no second descent).",
            ),
            F.columnar(
                "second_descent_survived",
                "bool",
                (Coord.VARIANT,),
                "Whether the final test loss stayed below the success threshold after onset "
                "(null if no onset).",
            ),
        ),
        materialized=True,
    )
)


# ---------------------------------------------------------------------------
# dimensionality_outcomes — the W_out↘W_in effective-dimensionality crossover.
#
# Reproduces `_load_effective_dimensionality_key_epochs`: skip the random-init
# period where W_out ≈ W_in, and only after W_out has first risen clearly above
# W_in, report the first epoch it falls back to/below W_in (and W_E's PR there).
# The "rose above first" gate is a running max over earlier epochs; defaults are
# (-1, -1.0) when no crossover occurs. PR values come from `participation_ratios`
# (the conformed fact), not the analyzer summary npz.
# ---------------------------------------------------------------------------

PARTICIPATION_RATIOS = "participation_ratios"

DIMENSIONALITY_OUTCOMES_TABLE = register_derived_table(
    DerivedTableSpec(
        name="dimensionality_outcomes",
        query=f"""
            WITH pivoted AS (
                SELECT epoch,
                       MAX(participation_ratio) FILTER (WHERE site = 'W_E') AS pr_e,
                       MAX(participation_ratio) FILTER (WHERE site = 'W_in') AS pr_in,
                       MAX(participation_ratio) FILTER (WHERE site = 'W_out') AS pr_out
                FROM {PARTICIPATION_RATIOS}
                WHERE head = 0 AND site IN ('W_E', 'W_in', 'W_out')
                GROUP BY epoch
            ),
            flagged AS (
                SELECT epoch, pr_e, pr_in, pr_out,
                       MAX(CASE WHEN pr_out > pr_in THEN 1 ELSE 0 END)
                           OVER (ORDER BY epoch ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                           AS rose
                FROM pivoted
            ),
            crossover AS (
                SELECT epoch, pr_e
                FROM flagged
                WHERE rose = 1 AND pr_out <= pr_in
                ORDER BY epoch
                LIMIT 1
            )
            SELECT
                COALESCE((SELECT epoch FROM crossover), -1)
                    AS effective_dimensionality_cross_over_epoch,
                COALESCE((SELECT pr_e FROM crossover), -1.0)
                    AS effective_dimensionality_crossover_W_E_pr
        """,
        input_tables=(PARTICIPATION_RATIOS,),
        outputs=(
            F.columnar(
                "effective_dimensionality_cross_over_epoch",
                "int64",
                (Coord.VARIANT,),
                "First epoch W_out's participation ratio falls back to/below W_in's after "
                "first rising above it (-1 if no crossover).",
            ),
            F.columnar(
                "effective_dimensionality_crossover_W_E_pr",
                "float64",
                (Coord.VARIANT,),
                "W_E participation ratio at the crossover epoch (-1.0 if no crossover).",
            ),
        ),
        materialized=True,
    )
)
