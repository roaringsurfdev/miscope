# Milestone v0.8.2 — Checkpoint Schedule, Transient Analysis, and Pipeline

**Released:** 2026-04-04

## Requirements

- **REQ_080** — Artifact Freshness, Completeness, and Incremental Analysis
- **REQ_083** — Checkpoint Schedule Manager (retrain with denser schedule)
- **REQ_084** — Transient Frequency Analyzer

Also archived here (cancelled or implemented elsewhere):
- **REQ_014** — Click-to-Navigate (implemented in Dash; written against Gradio)
- **REQ_059** — Cross-Variant Metric Refinements (implemented via variant_summary.json)

## Key Decisions

### REQ_080 — Artifact Freshness
- Three staleness types codified: epoch-incomplete, epoch-stale, summary-stale
- Cross-epoch staleness is metadata-only (`epochs` key from cross_epoch.npz)
- Incremental is now the default; `force=True` for full rerun
- `VariantAnalysisSummary` and `build_variant_registry` called at end of analysis thread

### REQ_083 — Checkpoint Schedule Manager
- Visual range-based checkpoint schedule builder integrated with global variant selector
- Supersedes REQ_015 (which addressed checkpoint *selection*, not *retraining*)

### REQ_084 — Transient Frequency Analyzer
- Cross-epoch analyzer: ragged storage for `peak_members`
- `TRANSIENT_CANONICAL_THRESHOLD=0.05` (5%) vs `FINAL_CANONICAL_THRESHOLD=0.10` (10%)
- Key finding: high `frac_explained` ≠ structural stability; blob geometry at peak = unstable
- PC1 cohesion asymptote: ~0.70 = re-assignment (recovery); ~1.0 = attrition (failure)
- 13/30 variants have transient frequencies; ds999 dominates high-homeless cases

### Housekeeping (no requirement)
- `variant.dir` property added as alias for `variant_dir`
- `variant.view("name", **kwargs)` and `EpochContext.view("name", **kwargs)` supported
- Node20 → Node24 CI migration (`FORCE_JAVASCRIPT_ACTIONS_TO_NODE24`)
- Pre-push git hook: ruff + pyright must pass before push
- README and DOMAIN_MODEL fully updated to v0.8.x state

## Key Files

- `src/miscope/analysis/freshness.py` — FreshnessReport, check_freshness(), cross_epoch_is_stale()
- `src/miscope/analysis/analyzers/transient_frequency.py` — TransientFrequencyAnalyzer
- `src/miscope/families/variant.py` — variant.dir alias, view(**kwargs) passthrough
- `src/miscope/views/catalog.py` — BoundView kwargs storage and merge
- `dashboard/pages/transient_frequencies.py` — Transient Frequencies dashboard page
- `dashboard/pages/checkpoint_schedule.py` — Checkpoint Schedule Manager page
- `.git/hooks/pre-push` — local ruff + pyright gate
- `.github/workflows/ci.yml` — Node24 migration
