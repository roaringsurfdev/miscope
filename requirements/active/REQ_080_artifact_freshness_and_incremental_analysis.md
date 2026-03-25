# REQ_080: Artifact Freshness, Completeness, and Incremental Analysis

**Status:** Draft
**Branch:** TBD (post `feature/neuron-group-pca`)
**Priority:** High — becoming critical as training runs are extended and re-analyzed

---

## Problem

As models are extended past their original training windows and analysis is rerun with `force=True`, there is currently no way to verify whether artifacts are up-to-date relative to available checkpoints. Staleness and incompleteness manifest in several ways:

1. **Per-epoch artifacts lag new checkpoints.** When training extends from 25k to 35k epochs, per-epoch analyzers (e.g. `neuron_freq_norm`, `repr_geometry`) produce artifacts only for the original checkpoints. New checkpoints have no corresponding artifacts until a full rerun.

2. **Cross-epoch artifacts are frozen at rerun time.** A `neuron_group_pca` cross-epoch artifact run at epoch 24999 silently remains at 24999 even when checkpoints now extend to 34999. The dashboard shows stale group structure with no indication it is incomplete.

3. **`variant_summary.json` is a separate concern.** It is currently regenerated via a separate notebook call (`compute_variant_metrics.py`) and is not part of the analysis pipeline. After extending training and rerunning analysis, it must be manually regenerated to reflect updated grokking epochs, labels, and loss metrics.

4. **No gap detection.** There is no tooling to answer: "which analyzers have fewer epochs than checkpoints?" or "which specific epochs are missing?" The only current path is manual inspection or artifact loading surprises at dashboard time.

---

## Goals

Allow a researcher to:
- Quickly verify which artifacts are fresh vs stale for any variant
- Run analysis incrementally (new checkpoints only) without full reruns
- Regenerate `variant_summary.json` as part of or immediately after an analysis run
- Detect and report gaps between available checkpoints and existing artifact coverage

---

## Conditions of Satisfaction

### CoS 1 — Freshness report
Given a variant, a freshness check reports for each per-epoch analyzer:
- Total checkpoints available
- Epochs covered by existing artifacts
- Missing epochs (checkpoints with no artifact)
- Whether the artifact is "fresh" (covers all checkpoints) or "stale" (gap exists)

### CoS 2 — Cross-epoch artifact staleness
Cross-epoch artifacts (e.g. `neuron_group_pca`, `gradient_site`) report the epoch range they cover. A staleness check compares this range against available checkpoints and flags artifacts that were built on a subset of available data.

### CoS 3 — Incremental per-epoch analysis
The analysis pipeline supports an incremental mode: for per-epoch analyzers, only checkpoints without existing artifacts are processed. Full reruns remain available via `force=True` at the analyzer level.

### CoS 4 — `variant_summary.json` regeneration in pipeline
`variant_summary.json` can be regenerated as a pipeline step, triggerable from the analysis run configuration (not only via a separate notebook). After an incremental or full rerun, the summary reflects the updated checkpoint range.

### CoS 5 — Notebook and dashboard surfacing
- A notebook cell (or standalone utility function) can print the freshness report for one or all variants
- The dashboard Analysis page surfaces a staleness indicator when a selected variant has known artifact gaps

---

## Constraints

- Must not break existing full-rerun (`force=True`) behavior
- Incremental mode must produce artifacts identical to what a full rerun would produce for those epochs
- Cross-epoch artifact staleness detection should not require reloading artifact data — it should be inferrable from artifact metadata (e.g. stored `epochs` array)
- `variant_summary.json` regeneration logic currently lives in `compute_variant_metrics.py` — this should be reused, not duplicated

---

## Out of Scope

- Automatic background re-analysis when new checkpoints are detected
- Artifact versioning or migration (separate concern)
- SQLite-backed artifact indexing (deferred until forcing function arrives)

---

## Notes

**Motivation context (2026-03-24):** When p101/s485/ds42 was extended from 25k to 35k epochs and rerun with `force=True`, the workflow required: (1) manual analysis run, (2) separate `compute_variant_metrics.py` call to update `variant_summary.json`, (3) manual inspection to confirm the grokking epoch was now captured (34173). The `neuron_group_pca` cross-epoch artifact silently remained at epoch 24999 coverage until explicitly rerun. No tooling existed to detect this gap.

**Staleness is becoming a research correctness issue**, not just an ergonomic one. Phase diagram metrics, group PCA geometry, and circularity measures drawn from stale artifacts produce misleading results that are difficult to catch without knowing to look for them.

**Artifact staleness taxonomy:**
- *Epoch-incomplete*: per-epoch artifact missing checkpoints at the tail (most common after training extension)
- *Epoch-stale*: cross-epoch artifact built on a subset of available checkpoints
- *Summary-stale*: `variant_summary.json` does not reflect the latest checkpoint range or analysis results
