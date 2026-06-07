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
