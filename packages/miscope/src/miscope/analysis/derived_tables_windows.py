"""Provisional training-window derived tables (REQ_144 Stage 2b, fork (f)).

**Quarantined on purpose.** The window *vocabulary* is stable — ``first_descent``,
``plateau``, ``second_descent``, ``final`` are established grokking phases;
``cascade`` is an experimental pre-onset probe; ``neural_collapse`` is a likely
future member — so :class:`~miscope.analysis.output_schema.Coord.WINDOW` is a
first-class coordinate. What is provisional is the *boundary-derivation method*.
This module is the **proxy** producer of window boundaries (threshold/crossover
epochs snapped to the checkpoint grid). A future ``dmd_window_ranges`` (windowed
DMD) would be a sibling producer emitting the *same* ``(variant, WINDOW)`` schema,
so proxy and DMD boundaries compare by a one-line join — cheap proxies as a
first-line predictor of the expensive method.

Because the boundary definitions are soft, this layer is **parity-relaxed**
(fork (e)): the proxy logic is translated faithfully so the dashboard's window
overlays don't shift, but a value divergence is a finding, not a gate. Nothing in
the stable outcome layer depends on these tables; they depend on it (one-way DAG).

Kept in its own module (imported by ``registry.build_index``) so the unstable
definitions cannot leak into the stable layer's file.
"""

from __future__ import annotations

from miscope.analysis.derived_table import DerivedTableSpec, register_derived_table
from miscope.analysis.output_schema import Coord
from miscope.analysis.output_schema import OutputField as F

# The conformed input tables (stable layer + the checkpoint-grid source).
_LOSS_OUTCOMES = "loss_outcomes"
_NEURON_THRESHOLD_OUTCOMES = "neuron_threshold_outcomes"
_DIMENSIONALITY_OUTCOMES = "dimensionality_outcomes"
_PARTICIPATION_RATIOS = "participation_ratios"  # supplies the checkpoint epoch grid


# ---------------------------------------------------------------------------
# window_ranges — proxy boundary producer: (variant, WINDOW) -> start/end epoch.
#
# Reproduces `_load_window_ranges`. Boundaries that come from dense-loss epochs
# (first-descent end, second-descent start/end) are snapped to the nearest
# checkpoint epoch (min |epoch - target|, lower epoch on ties — matching the
# engine's first-occurrence pick over ascending checkpoints). Cascade's bounds are
# already checkpoint epochs (first-mover count epoch, dimensionality crossover) and
# are used raw. Plateau/final derive from the other windows' boundaries. A window
# with no basis collapses to (0, 0) — the engine's empty-window sentinel.
# ---------------------------------------------------------------------------

WINDOW_RANGES_TABLE = register_derived_table(
    DerivedTableSpec(
        name="window_ranges",
        query=f"""
            WITH grid AS (SELECT DISTINCT epoch FROM {_PARTICIPATION_RATIOS}),
            last_ckpt AS (SELECT MAX(epoch) AS e FROM grid),
            src AS (
                SELECT
                    lo.train_loss_threshold_first_epoch AS tl_thr,
                    lo.second_descent_onset_epoch AS onset,
                    lo.test_loss_threshold_first_epoch AS te_thr,
                    lo.test_loss_min_epoch AS te_min,
                    nt.first_mover_frequency_count_threshold_epoch AS fm_cnt,
                    dm.effective_dimensionality_cross_over_epoch AS dim_x
                FROM {_LOSS_OUTCOMES} lo,
                     {_NEURON_THRESHOLD_OUTCOMES} nt,
                     {_DIMENSIONALITY_OUTCOMES} dm
            ),
            fd_end AS (
                SELECT epoch FROM grid
                ORDER BY abs(epoch - (SELECT tl_thr FROM src)), epoch LIMIT 1
            ),
            sd_start AS (
                SELECT epoch FROM grid
                ORDER BY abs(epoch - (SELECT onset FROM src)), epoch LIMIT 1
            ),
            sd_end_thr AS (
                SELECT epoch FROM grid
                ORDER BY abs(epoch - (SELECT te_thr FROM src)), epoch LIMIT 1
            ),
            sd_end_min AS (
                SELECT epoch FROM grid
                ORDER BY abs(epoch - (SELECT te_min FROM src)), epoch LIMIT 1
            ),
            bounds AS (
                SELECT
                    0 AS fd_start,
                    (SELECT epoch FROM fd_end) AS fd_end,
                    CASE WHEN (SELECT onset FROM src) IS NOT NULL
                         THEN (SELECT epoch FROM sd_start) ELSE 0 END AS sd_start,
                    CASE WHEN (SELECT onset FROM src) IS NOT NULL THEN
                         CASE WHEN (SELECT te_thr FROM src) > 0
                              THEN (SELECT epoch FROM sd_end_thr)
                              ELSE (SELECT epoch FROM sd_end_min) END
                         ELSE 0 END AS sd_end,
                    CASE WHEN (SELECT fm_cnt FROM src) <> 0 AND (SELECT dim_x FROM src) <> 0
                         THEN (SELECT fm_cnt FROM src) ELSE 0 END AS casc_start,
                    CASE WHEN (SELECT fm_cnt FROM src) <> 0 AND (SELECT dim_x FROM src) <> 0
                         THEN (SELECT dim_x FROM src) ELSE 0 END AS casc_end,
                    (SELECT e FROM last_ckpt) AS last_e
                FROM src
            )
            SELECT 'first_descent' AS "window", fd_start AS start_epoch, fd_end AS end_epoch
            FROM bounds
            UNION ALL SELECT 'plateau', fd_end, sd_start FROM bounds
            UNION ALL SELECT 'second_descent', sd_start, sd_end FROM bounds
            UNION ALL SELECT 'cascade', casc_start, casc_end FROM bounds
            UNION ALL SELECT 'final', sd_end, last_e FROM bounds
        """,
        input_tables=(
            _LOSS_OUTCOMES,
            _NEURON_THRESHOLD_OUTCOMES,
            _DIMENSIONALITY_OUTCOMES,
            _PARTICIPATION_RATIOS,
        ),
        outputs=(
            F.columnar(
                "start_epoch",
                "int64",
                (Coord.VARIANT, Coord.WINDOW),
                "Proxy start epoch of a training-phase window (checkpoint-snapped).",
            ),
            F.columnar(
                "end_epoch",
                "int64",
                (Coord.VARIANT, Coord.WINDOW),
                "Proxy end epoch of a training-phase window (checkpoint-snapped).",
            ),
        ),
        materialized=True,
    )
)


# ---------------------------------------------------------------------------
# window_metrics — conformed scalar facts sampled at each window boundary epoch.
#
# Reproduces the scalar half of `_load_window_metrics`: at each non-degenerate
# window's start and end boundary epoch, the train/test loss (direct dense-loss
# index — the boundary is a checkpoint epoch), the W_in/W_out participation ratios
# (the misnamed `resid_post_pr_w_*`), and the resid_post circularity + fisher mean.
# Long-keyed (variant, WINDOW, EPOCH) with a start/end `boundary` role; the Stage-3
# accessor reassembles the legacy `*_start` / `*_end` dict for renderers. Degenerate
# windows (start >= end) contribute no rows (the engine's skip). Parity-relaxed.
# ---------------------------------------------------------------------------

_LOSSES = "losses"
_SHAPE_CHARACTERIZATIONS = "shape_characterizations"
_REPR_GEOMETRY = "repr_geometry"

WINDOW_METRICS_TABLE = register_derived_table(
    DerivedTableSpec(
        name="window_metrics",
        query=f"""
            WITH boundaries AS (
                SELECT "window", 'start' AS boundary, start_epoch AS epoch
                FROM window_ranges WHERE start_epoch < end_epoch
                UNION ALL
                SELECT "window", 'end', end_epoch
                FROM window_ranges WHERE start_epoch < end_epoch
            ),
            pr AS (
                SELECT epoch,
                       MAX(participation_ratio) FILTER (WHERE site = 'W_in') AS pr_w_in,
                       MAX(participation_ratio) FILTER (WHERE site = 'W_out') AS pr_w_out
                FROM {_PARTICIPATION_RATIOS}
                WHERE head = 0 AND site IN ('W_in', 'W_out')
                GROUP BY epoch
            ),
            circ AS (
                SELECT epoch, value AS circularity
                FROM {_SHAPE_CHARACTERIZATIONS}
                WHERE site = 'resid_post' AND operation_type = 'circularity'
            ),
            fish AS (
                -- repr_geometry is exploded by row_id (per-class radii/dimensionality),
                -- so fisher_mean — a per-(epoch,site) scalar — is non-null on exactly one
                -- row per epoch; collapse to that one value.
                SELECT epoch, fisher_mean
                FROM {_REPR_GEOMETRY}
                WHERE site = 'resid_post' AND fisher_mean IS NOT NULL
            )
            SELECT b."window", b.epoch, b.boundary,
                   l.train_loss, l.test_loss,
                   pr.pr_w_in, pr.pr_w_out,
                   circ.circularity, fish.fisher_mean
            FROM boundaries b
            LEFT JOIN {_LOSSES} l ON l.epoch = b.epoch
            LEFT JOIN pr ON pr.epoch = b.epoch
            LEFT JOIN circ ON circ.epoch = b.epoch
            LEFT JOIN fish ON fish.epoch = b.epoch
            ORDER BY b."window", b.epoch
        """,
        input_tables=(
            "window_ranges",
            _LOSSES,
            _PARTICIPATION_RATIOS,
            _SHAPE_CHARACTERIZATIONS,
            _REPR_GEOMETRY,
        ),
        outputs=(
            F.columnar(
                "boundary",
                "str",
                (Coord.VARIANT, Coord.WINDOW, Coord.EPOCH),
                "Which window boundary this epoch is (start or end).",
            ),
            F.columnar(
                "train_loss",
                "float64",
                (Coord.VARIANT, Coord.WINDOW, Coord.EPOCH),
                "Training loss at the window boundary epoch.",
            ),
            F.columnar(
                "test_loss",
                "float64",
                (Coord.VARIANT, Coord.WINDOW, Coord.EPOCH),
                "Test loss at the window boundary epoch.",
            ),
            F.columnar(
                "pr_w_in",
                "float64",
                (Coord.VARIANT, Coord.WINDOW, Coord.EPOCH),
                "W_in participation ratio at the boundary epoch (engine's resid_post_pr_w_in).",
            ),
            F.columnar(
                "pr_w_out",
                "float64",
                (Coord.VARIANT, Coord.WINDOW, Coord.EPOCH),
                "W_out participation ratio at the boundary epoch (engine's resid_post_pr_w_out).",
            ),
            F.columnar(
                "circularity",
                "float64",
                (Coord.VARIANT, Coord.WINDOW, Coord.EPOCH),
                "resid_post centroid circularity at the boundary epoch.",
            ),
            F.columnar(
                "fisher_mean",
                "float64",
                (Coord.VARIANT, Coord.WINDOW, Coord.EPOCH),
                "resid_post mean Fisher discriminant at the boundary epoch.",
            ),
        ),
        materialized=True,
    )
)


# ---------------------------------------------------------------------------
# window_frequencies — frequency membership at window boundaries (fork f).
#
# The list-valued half of `_load_window_metrics`, flattened to one row per
# (variant, WINDOW, FREQUENCY) with a `boundary` (start/end/window) + `role`
# discriminator and the frequency's `band`. Roles: `specialized` (any committed
# neuron at the boundary epoch) and `committed` (count >= 0.10 * d_mlp) at each
# boundary; `gain`/`loss` are window-level set differences of the committed sets
# (end minus start, start minus end). Frequencies are 1-indexed; bands classify
# against the prime. The Stage-3 accessor reassembles the engine's per-window
# `*_start`/`*_end` lists and the unique-band sets. Parity-relaxed.
# ---------------------------------------------------------------------------

_ATTRIBUTION = "neuron_frequency_attribution"
_NEURON_THRESHOLD = 0.70
_POPULATION_FLOOR = 0.10

WINDOW_FREQUENCIES_TABLE = register_derived_table(
    DerivedTableSpec(
        name="window_frequencies",
        query=f"""
            WITH boundaries AS (
                SELECT "window", 'start' AS boundary, start_epoch AS epoch
                FROM window_ranges WHERE start_epoch < end_epoch
                UNION ALL
                SELECT "window", 'end', end_epoch
                FROM window_ranges WHERE start_epoch < end_epoch
            ),
            d_mlp AS (SELECT COUNT(DISTINCT neuron) AS n FROM {_ATTRIBUTION}),
            prime AS (SELECT MAX(prime) AS p FROM {_ATTRIBUTION}),
            bcounts AS (
                SELECT b."window", b.boundary, a.frequency + 1 AS f, COUNT(*) AS c
                FROM boundaries b
                JOIN {_ATTRIBUTION} a
                  ON a.epoch = b.epoch AND a.frac_explained >= {_NEURON_THRESHOLD}
                GROUP BY b."window", b.boundary, a.frequency
            ),
            specialized AS (SELECT "window", boundary, f, 'specialized' AS role FROM bcounts),
            committed AS (
                SELECT "window", boundary, f, 'committed' AS role FROM bcounts
                WHERE c >= {_POPULATION_FLOOR} * (SELECT n FROM d_mlp)
            ),
            cstart AS (SELECT "window", f FROM committed WHERE boundary = 'start'),
            cend AS (SELECT "window", f FROM committed WHERE boundary = 'end'),
            gains AS (
                SELECT e."window", 'window' AS boundary, e.f, 'gain' AS role
                FROM cend e LEFT JOIN cstart s ON s."window" = e."window" AND s.f = e.f
                WHERE s.f IS NULL
            ),
            losses AS (
                SELECT s."window", 'window' AS boundary, s.f, 'loss' AS role
                FROM cstart s LEFT JOIN cend e ON e."window" = s."window" AND e.f = s.f
                WHERE e.f IS NULL
            ),
            all_rows AS (
                SELECT "window", boundary, role, f FROM specialized
                UNION ALL SELECT "window", boundary, role, f FROM committed
                UNION ALL SELECT "window", boundary, role, f FROM gains
                UNION ALL SELECT "window", boundary, role, f FROM losses
            )
            SELECT "window", f AS frequency, boundary, role,
                   CASE WHEN f <= (SELECT p FROM prime) // 4 THEN 'low'
                        WHEN f > 3 * (SELECT p FROM prime) // 8 THEN 'high'
                        ELSE 'mid' END AS band
            FROM all_rows
            ORDER BY "window", boundary, role, frequency
        """,
        input_tables=("window_ranges", _ATTRIBUTION),
        outputs=(
            F.columnar(
                "boundary",
                "str",
                (Coord.VARIANT, Coord.WINDOW, Coord.FREQUENCY),
                "Window boundary the membership is sampled at (start / end / window-level).",
            ),
            F.columnar(
                "role",
                "str",
                (Coord.VARIANT, Coord.WINDOW, Coord.FREQUENCY),
                "Membership role: specialized, committed, gain, or loss.",
            ),
            F.columnar(
                "band",
                "str",
                (Coord.VARIANT, Coord.WINDOW, Coord.FREQUENCY),
                "Frequency band (low / mid / high) relative to the prime.",
            ),
        ),
        materialized=True,
    )
)
