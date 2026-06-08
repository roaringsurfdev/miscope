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


# ---------------------------------------------------------------------------
# neuron_threshold_outcomes — first-mover + population-threshold key epochs.
#
# Reproduces `_load_neuron_threshold_key_epochs` over the conformed attribution
# table. A neuron is "committed" at an epoch when frac_explained >= 0.70. The
# first-mover is the lowest committed (1-indexed) frequency at the earliest epoch
# any neuron commits; its count-threshold epoch is the first epoch that frequency's
# committed cohort reaches _FIRST_MOVER_COUNT; the specialization epoch is the first
# epoch the total committed count reaches _TOTAL_NEURON_COUNT_OVER_THRESHOLD. The
# attribution `frequency` is 0-indexed, so the 1-indexed summary value is +1
# (matching the NeuronFrequencyAttribution helper). Defaults are -1.
# ---------------------------------------------------------------------------

_FIRST_MOVER_COUNT = 40
_TOTAL_NEURON_COUNT_OVER_THRESHOLD = 100

NEURON_THRESHOLD_OUTCOMES_TABLE = register_derived_table(
    DerivedTableSpec(
        name="neuron_threshold_outcomes",
        query=f"""
            WITH committed AS (
                SELECT epoch, neuron, frequency
                FROM {ATTRIBUTION}
                WHERE frac_explained >= {_NEURON_THRESHOLD}
            ),
            fm_epoch AS (SELECT MIN(epoch) AS e FROM committed),
            fm_freq AS (
                SELECT MIN(frequency) + 1 AS f
                FROM committed
                WHERE epoch = (SELECT e FROM fm_epoch)
            ),
            fm_count_epoch AS (
                SELECT MIN(epoch) AS e FROM (
                    SELECT epoch, COUNT(*) AS c FROM committed
                    WHERE frequency = (SELECT f FROM fm_freq) - 1
                    GROUP BY epoch
                ) WHERE c >= {_FIRST_MOVER_COUNT}
            ),
            total_epoch AS (
                SELECT MIN(epoch) AS e FROM (
                    SELECT epoch, COUNT(*) AS c FROM committed GROUP BY epoch
                ) WHERE c >= {_TOTAL_NEURON_COUNT_OVER_THRESHOLD}
            )
            SELECT
                COALESCE((SELECT e FROM fm_epoch), -1) AS first_mover_epoch,
                COALESCE((SELECT f FROM fm_freq), -1) AS first_mover_frequency,
                COALESCE((SELECT e FROM fm_count_epoch), -1)
                    AS first_mover_frequency_count_threshold_epoch,
                COALESCE((SELECT e FROM total_epoch), -1)
                    AS total_neurons_over_specialization_threshold_epoch
        """,
        input_tables=(ATTRIBUTION,),
        outputs=(
            F.columnar(
                "first_mover_epoch",
                "int64",
                (Coord.VARIANT,),
                "Earliest epoch any neuron commits (frac_explained >= 0.70); -1 if never.",
            ),
            F.columnar(
                "first_mover_frequency",
                "int64",
                (Coord.VARIANT,),
                "Lowest committed (1-indexed) frequency at the first-mover epoch; -1 if never.",
            ),
            F.columnar(
                "first_mover_frequency_count_threshold_epoch",
                "int64",
                (Coord.VARIANT,),
                "First epoch the first-mover frequency's committed cohort reaches 40; -1 if never.",
            ),
            F.columnar(
                "total_neurons_over_specialization_threshold_epoch",
                "int64",
                (Coord.VARIANT,),
                "First epoch the total committed-neuron count reaches 100; -1 if never.",
            ),
        ),
        materialized=True,
    )
)


# ---------------------------------------------------------------------------
# competition_geometry_outcomes — commitment-window span + peak circularity.
#
# Reproduces `_load_competition_and_geometry_summary_metrics`. The competition
# window spans the per-neuron commitment epochs from `recompute_commitment_epochs`,
# whose definition is NOT "first epoch committed" but the earliest epoch from which
# a neuron stays specialized to its *final* dominant frequency continuously through
# the end — and only neurons committed at the final epoch qualify. As a SQL
# gaps-and-islands reduction: per qualifying neuron, the commitment epoch is the
# first epoch *after the last epoch where (frac >= 0.70 AND freq = final freq)
# fails* (or the first epoch, if it never fails). start/end = min/max over those;
# duration = span. Peak circularity is MAX over conformed resid_post circularity.
# All null when no neuron qualifies / no circularity rows (engine: None).
# ---------------------------------------------------------------------------

SHAPE_CHARACTERIZATIONS = "shape_characterizations"

COMPETITION_GEOMETRY_OUTCOMES_TABLE = register_derived_table(
    DerivedTableSpec(
        name="competition_geometry_outcomes",
        query=f"""
            WITH final_epoch AS (SELECT MAX(epoch) AS e FROM {ATTRIBUTION}),
            qualifying AS (
                SELECT neuron, frequency AS final_freq
                FROM {ATTRIBUTION}
                WHERE epoch = (SELECT e FROM final_epoch)
                  AND frac_explained >= {_NEURON_THRESHOLD}
            ),
            good AS (
                SELECT a.neuron, a.epoch,
                       (a.frac_explained >= {_NEURON_THRESHOLD} AND a.frequency = q.final_freq)
                           AS is_good
                FROM {ATTRIBUTION} a
                JOIN qualifying q USING (neuron)
            ),
            last_bad AS (
                SELECT neuron, MAX(epoch) AS lb FROM good WHERE NOT is_good GROUP BY neuron
            ),
            commit_epoch AS (
                SELECT g.neuron, MIN(g.epoch) AS ce
                FROM good g
                LEFT JOIN last_bad lb USING (neuron)
                WHERE lb.lb IS NULL OR g.epoch > lb.lb
                GROUP BY g.neuron
            )
            SELECT
                (SELECT MIN(ce) FROM commit_epoch) AS competition_window_start,
                (SELECT MAX(ce) FROM commit_epoch) AS competition_window_end,
                (SELECT MAX(ce) - MIN(ce) FROM commit_epoch) AS competition_window_duration,
                (SELECT MAX(value) FROM {SHAPE_CHARACTERIZATIONS}
                 WHERE site = 'resid_post' AND operation_type = 'circularity')
                    AS max_resid_post_circularity
        """,
        input_tables=(ATTRIBUTION, SHAPE_CHARACTERIZATIONS),
        outputs=(
            F.columnar(
                "competition_window_start",
                "int64",
                (Coord.VARIANT,),
                "Earliest per-neuron first-commitment epoch (null if no neuron commits).",
            ),
            F.columnar(
                "competition_window_end",
                "int64",
                (Coord.VARIANT,),
                "Latest per-neuron first-commitment epoch (null if no neuron commits).",
            ),
            F.columnar(
                "competition_window_duration",
                "int64",
                (Coord.VARIANT,),
                "Span between the earliest and latest first-commitment epochs.",
            ),
            F.columnar(
                "max_resid_post_circularity",
                "float64",
                (Coord.VARIANT,),
                "Peak resid_post centroid circularity over training (null if absent).",
            ),
        ),
        materialized=True,
    )
)


# ===========================================================================
# REQ_144 — list-valued frequency-portfolio outcomes
#
# These reproduce the engine's attribution-based list fields. A neuron is
# specialized at an epoch when frac_explained >= 0.70; committed_frequencies adds
# a population floor (count >= floor * d_mlp). The attribution `frequency` is
# 0-indexed, so the 1-indexed summary value is +1. Empty results are emitted as
# empty lists (not NULL) to match the engine's `[]`.
# ===========================================================================

_CANONICAL_SPECIALIZATION_THRESHOLD = 0.10  # population floor for "learned" at the final epoch

LOSS_OUTCOMES = "loss_outcomes"


# ---------------------------------------------------------------------------
# learned_frequencies_outcome — committed frequencies at the final epoch.
# ---------------------------------------------------------------------------

LEARNED_FREQUENCIES_OUTCOME_TABLE = register_derived_table(
    DerivedTableSpec(
        name="learned_frequencies_outcome",
        query=f"""
            WITH final_epoch AS (SELECT MAX(epoch) AS e FROM {ATTRIBUTION}),
            d_mlp AS (SELECT COUNT(DISTINCT neuron) AS n FROM {ATTRIBUTION}),
            counts AS (
                SELECT frequency, COUNT(*) AS c
                FROM {ATTRIBUTION}
                WHERE epoch = (SELECT e FROM final_epoch)
                  AND frac_explained >= {_NEURON_THRESHOLD}
                GROUP BY frequency
            ),
            learned AS (
                SELECT frequency + 1 AS f
                FROM counts
                WHERE c >= {_CANONICAL_SPECIALIZATION_THRESHOLD} * (SELECT n FROM d_mlp)
            )
            SELECT
                COALESCE(
                    (SELECT array_agg(f ORDER BY f) FROM learned), CAST([] AS BIGINT[])
                ) AS learned_frequencies,
                (SELECT COUNT(*) FROM learned) AS learned_frequency_count,
                {_CANONICAL_SPECIALIZATION_THRESHOLD} AS canonical_specialization_threshold
        """,
        input_tables=(ATTRIBUTION,),
        outputs=(
            F.columnar(
                "learned_frequencies",
                "int64",
                (Coord.VARIANT,),
                "Sorted 1-indexed frequencies population-committed at the final epoch.",
            ),
            F.columnar(
                "learned_frequency_count",
                "int64",
                (Coord.VARIANT,),
                "Count of learned (final-epoch population-committed) frequencies.",
            ),
            F.columnar(
                "canonical_specialization_threshold",
                "float64",
                (Coord.VARIANT,),
                "Population floor (fraction of d_mlp) for a frequency to count as learned.",
            ),
        ),
        materialized=True,
    )
)


# ---------------------------------------------------------------------------
# onset_portfolio_outcome — committed/specialized frequency portfolio at the
# second-descent onset + the handshake check. Derived-on-derived: the onset epoch
# comes from `loss_outcomes`, the learned set from `learned_frequencies_outcome`.
#
# `epoch_index(onset)` is the nearest stored epoch at or after onset (clamped), so
# the SQL takes MIN(epoch) >= onset (fallback MAX). committed adds the population
# floor; the misnamed `second_descent_onset_committed_frequencies` is actually the
# *specialized* set. Bands classify each specialized (1-indexed) frequency against
# the prime. Every field is NULL when there is no second-descent onset (engine: None).
# ---------------------------------------------------------------------------

ONSET_PORTFOLIO_OUTCOME_TABLE = register_derived_table(
    DerivedTableSpec(
        name="onset_portfolio_outcome",
        query=f"""
            WITH onset AS (SELECT second_descent_onset_epoch AS oe FROM {LOSS_OUTCOMES}),
            onset_epoch AS (
                SELECT CASE WHEN (SELECT oe FROM onset) IS NULL THEN NULL
                    ELSE COALESCE(
                        (SELECT MIN(epoch) FROM {ATTRIBUTION}
                         WHERE epoch >= (SELECT oe FROM onset)),
                        (SELECT MAX(epoch) FROM {ATTRIBUTION})
                    ) END AS e
            ),
            d_mlp AS (SELECT COUNT(DISTINCT neuron) AS n FROM {ATTRIBUTION}),
            prime AS (SELECT MAX(prime) AS p FROM {ATTRIBUTION}),
            at_onset AS (
                SELECT frequency, COUNT(*) AS c
                FROM {ATTRIBUTION}
                WHERE epoch = (SELECT e FROM onset_epoch) AND frac_explained >= {_NEURON_THRESHOLD}
                GROUP BY frequency
            ),
            committed AS (
                SELECT frequency + 1 AS f FROM at_onset
                WHERE c >= {_CANONICAL_SPECIALIZATION_THRESHOLD} * (SELECT n FROM d_mlp)
            ),
            specialized AS (SELECT frequency + 1 AS f FROM at_onset),
            learned AS (SELECT UNNEST(learned_frequencies) AS f FROM learned_frequencies_outcome),
            failures AS (SELECT f FROM committed WHERE f NOT IN (SELECT f FROM learned)),
            banded AS (
                SELECT f,
                    CASE WHEN f <= (SELECT p FROM prime) // 4 THEN 'low'
                         WHEN f > 3 * (SELECT p FROM prime) // 8 THEN 'high'
                         ELSE 'mid' END AS band
                FROM specialized
            )
            SELECT
                CASE WHEN (SELECT oe FROM onset) IS NULL THEN NULL ELSE COALESCE(
                    (SELECT array_agg(f ORDER BY f) FROM committed), CAST([] AS BIGINT[])) END
                    AS committed_frequencies_at_onset,
                CASE WHEN (SELECT oe FROM onset) IS NULL THEN NULL ELSE COALESCE(
                    (SELECT array_agg(f ORDER BY f) FROM failures), CAST([] AS BIGINT[])) END
                    AS handshake_failures,
                CASE WHEN (SELECT oe FROM onset) IS NULL THEN NULL
                    ELSE (SELECT COUNT(*) FROM failures) = 0 END AS handshake_succeeded,
                CASE WHEN (SELECT oe FROM onset) IS NULL THEN NULL ELSE COALESCE(
                    (SELECT array_agg(f ORDER BY f) FROM specialized), CAST([] AS BIGINT[])) END
                    AS second_descent_onset_committed_frequencies,
                CASE WHEN (SELECT oe FROM onset) IS NULL THEN NULL ELSE COALESCE(
                    (SELECT array_agg(band ORDER BY f) FROM banded), CAST([] AS VARCHAR[])) END
                    AS second_descent_onset_frequency_bands,
                CASE WHEN (SELECT oe FROM onset) IS NULL THEN NULL
                    ELSE EXISTS (SELECT 1 FROM banded WHERE band = 'low') END
                    AS second_descent_onset_has_low_band,
                CASE WHEN (SELECT oe FROM onset) IS NULL THEN NULL
                    ELSE (SELECT COUNT(DISTINCT band) FROM banded) END
                    AS second_descent_onset_band_count
        """,
        input_tables=(ATTRIBUTION, LOSS_OUTCOMES, "learned_frequencies_outcome"),
        outputs=(
            F.columnar(
                "committed_frequencies_at_onset",
                "int64",
                (Coord.VARIANT,),
                "1-indexed frequencies population-committed at the onset epoch (null if no onset).",
            ),
            F.columnar(
                "handshake_failures",
                "int64",
                (Coord.VARIANT,),
                "Onset-committed frequencies not in the final learned set (null if no onset).",
            ),
            F.columnar(
                "handshake_succeeded",
                "bool",
                (Coord.VARIANT,),
                "Whether every onset-committed frequency survived into the learned set.",
            ),
            F.columnar(
                "second_descent_onset_committed_frequencies",
                "int64",
                (Coord.VARIANT,),
                "Specialized 1-indexed frequencies at the onset epoch (null if no onset).",
            ),
            F.columnar(
                "second_descent_onset_frequency_bands",
                "str",
                (Coord.VARIANT,),
                "Band (low/mid/high) of each onset specialized frequency (null if no onset).",
            ),
            F.columnar(
                "second_descent_onset_has_low_band",
                "bool",
                (Coord.VARIANT,),
                "Whether any onset specialized frequency falls in the low band.",
            ),
            F.columnar(
                "second_descent_onset_band_count",
                "int64",
                (Coord.VARIANT,),
                "Distinct band count across onset specialized frequencies (null if no onset).",
            ),
        ),
        materialized=True,
    )
)


# ---------------------------------------------------------------------------
# transient_outcome — the variant-level transient rollup over the per-frequency
# `transient_frequencies` derived table (REQ_141). The not-final rows give the
# transient frequencies (1-indexed for the summary) and the homeless-neuron total;
# the homeless fraction divides by d_mlp; the detection threshold is the 0.05
# fraction. Reproduces `_load_transient_metrics`.
# ---------------------------------------------------------------------------

TRANSIENT_FREQUENCIES = "transient_frequencies"

TRANSIENT_OUTCOME_TABLE = register_derived_table(
    DerivedTableSpec(
        name="transient_outcome",
        query=f"""
            WITH nf AS (
                SELECT frequency, homeless_count FROM {TRANSIENT_FREQUENCIES} WHERE NOT is_final
            ),
            d_mlp AS (SELECT COUNT(DISTINCT neuron) AS n FROM {ATTRIBUTION})
            SELECT
                COALESCE((SELECT array_agg(frequency + 1 ORDER BY frequency) FROM nf),
                         CAST([] AS BIGINT[])) AS transient_frequencies,
                (SELECT COUNT(*) FROM nf) AS transient_frequency_count,
                COALESCE((SELECT SUM(homeless_count) FROM nf), 0) AS homeless_neuron_count,
                CAST(COALESCE((SELECT SUM(homeless_count) FROM nf), 0) AS DOUBLE)
                    / (SELECT n FROM d_mlp) AS homeless_neuron_fraction,
                {_TRANSIENT_FRACTION} AS transient_detection_threshold
        """,
        input_tables=(TRANSIENT_FREQUENCIES, ATTRIBUTION),
        outputs=(
            F.columnar(
                "transient_frequencies",
                "int64",
                (Coord.VARIANT,),
                "1-indexed frequencies that peaked above the transient floor but did not survive.",
            ),
            F.columnar(
                "transient_frequency_count",
                "int64",
                (Coord.VARIANT,),
                "Count of transient frequencies.",
            ),
            F.columnar(
                "homeless_neuron_count",
                "int64",
                (Coord.VARIANT,),
                "Total peak-cohort neurons that abandoned a transient frequency without re-homing.",
            ),
            F.columnar(
                "homeless_neuron_fraction",
                "float64",
                (Coord.VARIANT,),
                "Homeless-neuron count as a fraction of d_mlp.",
            ),
            F.columnar(
                "transient_detection_threshold",
                "float64",
                (Coord.VARIANT,),
                "Fraction-of-d_mlp floor for a frequency to count as ever-qualified.",
            ),
        ),
        materialized=True,
    )
)


# ===========================================================================
# REQ_144 Stage 4 — variant_outcomes: the conformed per-variant outcome row.
#
# A pure DerivedTableSpec (CoS #1) joining the seven stable clusters into one wide
# row per variant — the queryable surface that replaces the JSON-flatten co-emission
# (warehouse/outcomes.py). Cross-variant questions ("variants with
# homeless_neuron_fraction > 0.2") are one-line SQL over the family glob; the
# registry (fork a) is its cross-variant view. The two classifications (Python,
# fork b) live alongside in variant_classification, not here.
# ===========================================================================

VARIANT_OUTCOMES = "variant_outcomes"

# (table alias, column, dtype) for every stable field, in summary order. The list is
# the single source for both the SELECT and the declared output schema (no drift).
_VARIANT_OUTCOME_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("lo", "train_loss_min", "float64"),
    ("lo", "train_loss_min_epoch", "int64"),
    ("lo", "train_loss_threshold_first_epoch", "int64"),
    ("lo", "train_loss_final", "float64"),
    ("lo", "test_loss_min", "float64"),
    ("lo", "test_loss_min_epoch", "int64"),
    ("lo", "test_loss_max", "float64"),
    ("lo", "test_loss_max_epoch", "int64"),
    ("lo", "peak_test_loss_epoch", "int64"),
    ("lo", "test_loss_threshold_first_epoch", "int64"),
    ("lo", "test_loss_final", "float64"),
    ("lo", "final_test_loss", "float64"),
    ("lo", "second_descent_onset_epoch", "int64"),
    ("lo", "second_descent_survived", "bool"),
    ("dm", "effective_dimensionality_cross_over_epoch", "int64"),
    ("dm", "effective_dimensionality_crossover_W_E_pr", "float64"),
    ("nt", "first_mover_epoch", "int64"),
    ("nt", "first_mover_frequency", "int64"),
    ("nt", "first_mover_frequency_count_threshold_epoch", "int64"),
    ("nt", "total_neurons_over_specialization_threshold_epoch", "int64"),
    ("cg", "competition_window_start", "int64"),
    ("cg", "competition_window_end", "int64"),
    ("cg", "competition_window_duration", "int64"),
    ("cg", "max_resid_post_circularity", "float64"),
    ("lf", "learned_frequencies", "int64"),
    ("lf", "learned_frequency_count", "int64"),
    ("lf", "canonical_specialization_threshold", "float64"),
    ("op", "committed_frequencies_at_onset", "int64"),
    ("op", "handshake_failures", "int64"),
    ("op", "handshake_succeeded", "bool"),
    ("op", "second_descent_onset_committed_frequencies", "int64"),
    ("op", "second_descent_onset_frequency_bands", "str"),
    ("op", "second_descent_onset_has_low_band", "bool"),
    ("op", "second_descent_onset_band_count", "int64"),
    ("tr", "transient_frequencies", "int64"),
    ("tr", "transient_frequency_count", "int64"),
    ("tr", "homeless_neuron_count", "int64"),
    ("tr", "homeless_neuron_fraction", "float64"),
    ("tr", "transient_detection_threshold", "float64"),
)

_VARIANT_OUTCOME_INPUTS = (
    "loss_outcomes",
    "dimensionality_outcomes",
    "neuron_threshold_outcomes",
    "competition_geometry_outcomes",
    "learned_frequencies_outcome",
    "onset_portfolio_outcome",
    "transient_outcome",
)

_VARIANT_OUTCOMES_SELECT = ",\n                   ".join(
    f"{alias}.{column}" for alias, column, _ in _VARIANT_OUTCOME_COLUMNS
)

VARIANT_OUTCOMES_TABLE = register_derived_table(
    DerivedTableSpec(
        name=VARIANT_OUTCOMES,
        query=f"""
            SELECT {_VARIANT_OUTCOMES_SELECT}
            FROM loss_outcomes lo
            JOIN dimensionality_outcomes dm USING (variant_id, run_set)
            JOIN neuron_threshold_outcomes nt USING (variant_id, run_set)
            JOIN competition_geometry_outcomes cg USING (variant_id, run_set)
            JOIN learned_frequencies_outcome lf USING (variant_id, run_set)
            JOIN onset_portfolio_outcome op USING (variant_id, run_set)
            JOIN transient_outcome tr USING (variant_id, run_set)
        """,
        input_tables=_VARIANT_OUTCOME_INPUTS,
        outputs=tuple(
            F.columnar(
                column,
                dtype,
                (Coord.VARIANT,),
                f"{column} (conformed per-variant outcome, rolled up from the stable clusters).",
            )
            for _, column, dtype in _VARIANT_OUTCOME_COLUMNS
        ),
        materialized=True,
    )
)
