# Milestone v0.8.2 — Artifact Freshness

**Released:** 2026-03-29
**Branch:** `feature/req-080-artifact-freshness`

## Requirements

- **REQ_080** — Artifact Freshness, Completeness, and Incremental Analysis

## Key Decisions

- **Staleness taxonomy codified in code**: Three distinct types — epoch-incomplete,
  epoch-stale, summary-stale — each with its own detection path and status label.
- **Cross-epoch staleness is metadata-only**: `epochs` key read from cross_epoch.npz
  via `np.load` (no full artifact load). Absent key treated conservatively as stale.
- **Incremental is now the default**: `force=False` in pipeline no longer silently
  skips stale cross-epoch artifacts; it rebuilds them. Full rerun still available via
  `force=True`.
- **Summary regeneration in pipeline**: `VariantAnalysisSummary` and
  `build_variant_registry` called at end of `_run_analysis_thread` — no separate
  notebook step required.

## Key Files

- `src/miscope/analysis/freshness.py` — `FreshnessReport`, `check_freshness()`,
  `cross_epoch_is_stale()`
- `src/miscope/analysis/pipeline.py` — cross-epoch stale check integrated
- `dashboard/pages/analysis_run.py` — CoS 4 (summary regen) + CoS 5 (freshness alert)
- `tests/test_freshness.py` — 32 tests covering all CoS items
