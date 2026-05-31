# REQ_134: Regression Harness → Canonical Analyzer-Set Pattern

**Status:** Stub — *surfaced while validating REQ_132; scope/CoS to be developed.*
**Priority:** Low-Medium — the v1.0.0 byte-regression safety net is currently inoperable.
**Branch:** TBD
**Dependencies:**
- REQ_132 (collapsed the register triad; this harness now calls `register(...)`).

**Attribution:** Engineering Claude (stubbed 2026-05-30 from a REQ_132 validation run).

---

## Problem

`scripts/run_regression_check.py` hand-curates the analyzer set it registers on
the pipeline. That curated list has fallen behind the analyzer dependency graph:
it omits upstreams that current Specs declare as dependencies, so a real run
fails midway.

Observed 2026-05-30 (variant `p109_seed485_dseed598`):
- `fourier_frequency_quality` depends on `neuron_grouping` (not registered) → skipped.
- `neuron_dynamics` requires `activation_basis_projection` (not registered) →
  cross-epoch blocked-by `RuntimeError`, aborting the run.

This is **pre-existing** (predates REQ_132 — the omitted dependencies merged with
REQ_126/127 and REQ_130) and reproduces identically on `develop`. REQ_132 only
made the symptom visible by prompting a harness run.

## Direction (provisional)

The dashboard and `scripts/run_analysis.py` already use the canonical pattern and
*cannot* fall behind the dependency graph:

```python
specs = AnalyzerRegistry.list_for_family(variant.family)
plan  = plan_analysis(variant, specs, force=force)
pipeline.run(plan=plan)
```

`run_regression_check.py` should adopt the same pattern rather than enumerating
analyzers by hand. Open questions to settle during scoping:

- The script intentionally **excludes** some analyzers from regression (stochastic
  output: `landscape_flatness`, `fourier_nucleation`, etc. — see `EXCLUDED_ANALYZERS`).
  The canonical pattern pulls the full family set, so excludes must be applied as a
  filter over `list_for_family` rather than by omission from a hand list.
- `reference_checksums.json` will need a regen once the analyzer set it covers
  changes (REQ_102 retirements already left stale entries — see memory
  `project_req102_retirement_surface`).
- Confirm whether `run_analysis_regression.py` shares the same staleness and should
  be folded in.

## Conditions of Satisfaction

*(Deferred — develop during a dedicated session.)*

## Notes

- Until fixed, the byte-regression safety net for v1.0.0 is unavailable; end-to-end
  validation currently leans on the dashboard analysis-run flow + the unit suite.
