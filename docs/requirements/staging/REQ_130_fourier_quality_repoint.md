# REQ_130: Re-point `fourier_frequency_quality` off `dominant_frequencies` (unblock retirement)

**Status:** Completed — merged to `develop` 2026-05-29. User confirmed a clean canon analysis run. Full miscope suite green (1528 passed, 27 skipped). Metric redefined with the user as **neuron-weighted task coverage**. Scope included the DataView straggler REQ_127 missed and a planner topological-sort fix (first secondary→secondary dependency). Unblocks REQ_102's deletion of `dominant_frequencies` — with REQ_131 (merged), REQ_102's two deferrals are both cleared.
**Priority:** Medium — unblocks a REQ_102 deferral (the only surviving consumer of `dominant_frequencies`).
**Branch:** `feature/req-130-fourier-quality-repoint` (off `develop`).
**Dependencies:**
- REQ_102 (Analyzer Deprecation — deferred the `dominant_frequencies` retirement specifically because this consumer blocks it).
- REQ_118 (neuron_grouping — the new source; modadd Fourier override makes the group index a frequency index).
- REQ_131 (sibling consumer-migration; merged 2026-05-29 — established the "re-point a sole consumer to unblock a REQ_102 deletion" pattern).

**Attribution:** Engineering Claude (stubbed 2026-05-28 from the REQ_102 analyzer-dependency audit; CoS + metric redefinition decided with the user 2026-05-29).

---

## Problem

`dominant_frequencies` is retired in spirit (REQ_126 absorbed its weight-side Fourier into `weight_basis_projection`) but **cannot be deleted** because `fourier_frequency_quality` still consumes it: `depends_on = "dominant_frequencies"` + `ArtifactInput("dominant_frequencies", scope="epoch")` (secondary analyzer). It is the sole remaining consumer.

The fix is to re-point `fourier_frequency_quality` from `dominant_frequencies` to **`neuron_grouping`** (per user). If feasible, that frees `dominant_frequencies` for retirement under REQ_102.

## Research context (shapes the approach)

Per the user's assessment (see [[frequency-choice-frame]]): `fourier_frequency_quality` is **not currently yielding valuable information** — itself a finding. It sits at the hard, unresolved question of whether model performance depends on *which* frequencies are chosen and *when*; the literature's "frequencies follow from the prime / group math" frame is not supported by the data here. Consequently:

- **Bit-wise parity is likely unnecessary.** This is not a mechanical port like `weight_spectra`; the analyzer will need close re-examination of what it should measure. Treat the re-point as an opportunity to reconsider the metric, not just swap the input source.

## Design (developed 2026-05-29, decided with the user)

### What each source provides (the re-point is not mechanical)

- **`dominant_frequencies` (old source)** emits `coefficients` `(p,)`: the per-frequency L2 energy of the **embedding** `W_E` projected onto the Fourier basis. The old metric thresholds it (`> 3×mean`) to a hard dominant-frequency set, then scores R² of the ideal `p×p×p` mod-p logit tensor onto the 2D Fourier subspace those frequencies span. The "dominant" notion is **embedding-side**.
- **`neuron_grouping` (new source)** emits per-neuron group assignments. For the modadd family the Fourier **override** (`feature_basis_name="fourier_w_in"`) groups each neuron by its dominant *input-side* frequency, so **group index `g` = frequency `g+1`** and `n_per_group[g]` = neurons computing with frequency `g+1`. The "dominant" notion becomes **neuron-side** (which frequencies the MLP actually organizes around).

### Decision: redefine as neuron-weighted task coverage

Per the user (and [[frequency-choice-frame]]: the old metric yields no real signal), the re-point is taken as a chance to **redefine**, not preserve. New definition:

> Weight each Fourier frequency by neuron occupancy from `neuron_grouping`, then compute the R² of the ideal mod-p tensor onto that **weighted** 2D Fourier subspace.

Concretely, per epoch:
1. Load `neuron_grouping` at the current epoch; require it is frequency-indexed (`feature_basis_name == "fourier_w_in"` / `had_family_override`), else this metric is undefined for the architecture → skip/raise (documented).
2. Per-frequency weight `w_g = n_per_group[g] / max_g(n_per_group)` (relative occupancy, `[0,1]`, top frequency = 1; all-zero if no group populated). Max-normalization makes the metric **reduce to the old hard-subspace R² in the clean limit** (used frequencies → 1, unused → 0).
3. Map to basis rows: `w_rows[2g+1] = w_rows[2g+2] = w_g` (sin/cos of frequency `g+1`); constant row 0 weight 0.
4. `quality_score = Σ_{i,j,c} w_rows[i]·w_rows[j]·T_2D[i,j,c]² / p²`, where `T_2D` is the ideal-tensor projection computed over the **full** basis (generalizes the old `_compute_quality_score`; binary weights recover it exactly).

### Output contract

Per-epoch + summary (`produces_summary=True`, unchanged). Keys: `quality_score` (the neuron-weighted R²), `active_frequencies` (frequencies with `n_per_group>0`), `k` (count of active frequencies), and `coverage_hard` (the binary-set R² over the active set — a reference point that equals the old metric's score *if* the old set matched the active set, useful for interpretation). `reconstruction_error = 1 - quality_score` retained for continuity.

### Not bit-wise parity (by design)

This is a redefinition, so parity with the old `dominant_frequencies`-based output is **not** a CoS (the old metric measured something different and yielded no signal). Validation is: the analyzer runs over canon, produces sane `[0,1]` trajectories, and the clean-limit reduction holds (a unit test with synthetic frequency-pure grouping recovers the hard-subspace R²).

## Conditions of Satisfaction

### Re-point & redefine

- [x] `fourier_frequency_quality` declares `ArtifactInput("neuron_grouping")` (drops `dominant_frequencies`); `depends_on` updated; the planner orders it after `neuron_grouping` (see planner fix under Discovered).
- [x] Metric redefined per Design: neuron-occupancy-weighted R² of the ideal mod-p tensor; `_weighted_quality_score` accepts per-basis-row weights, binary weights recover the prior behavior (emitted as `coverage_hard`).
- [x] `prime` and `fourier_basis` sourced from context; per-frequency weights derived from `neuron_grouping`'s `n_per_group` (max-normalized relative occupancy), mapped to basis rows `{2g+1, 2g+2}`.
- [x] Guard `_require_frequency_indexed` raises a clear error on non-frequency-indexed `neuron_grouping` (universal kmeans path) — the metric is mod-p-task-specific.

### Output

- [x] Per-epoch keys: `quality_score`, `coverage_hard`, `active_frequencies`, `k`, `reconstruction_error`; summary keys `quality_score`/`coverage_hard`/`reconstruction_error`/`k`. All existing output consumers (`renderers/fourier_frequency_quality`, `renderers/input_trace`, View Catalog `quality_score` loader) read only retained keys → still load.

### Tests

- [x] Unit: clean-limit reduction (`test_clean_limit_*`) — frequency-pure grouping → `quality_score == coverage_hard` and matches an independent hard-subspace R².
- [x] Unit: occupancy monotonicity (`test_occupancy_monotonic`).
- [x] Unit: non-frequency-indexed grouping guard fires (`test_guard_rejects_universal_kmeans_grouping`).
- [x] Planner: secondary→secondary topological ordering + cycle detection (`test_plan_secondary_depends_on_secondary_ordered_and_unblocked`, `test_plan_secondary_cyclic_dependency_raises`). Full suite green.

### Closure

- [x] No live consumer (analyzer / View Catalog / DataView) reads `dominant_frequencies` → hand its retirement to REQ_102 (→ `weight_basis_projection`, embedding site). (`export.py` `_VISUALIZATION_REGISTRY` + a few docstring/README examples still name it — the broader REQ_102/REQ_129 retirement-doc cleanup, out of scope here, same as REQ_131.)

## Discovered during implementation (2026-05-29)

- **First secondary→secondary dependency in the codebase.** Every prior secondary depended on a *primary* (always complete before the secondary phase). `fourier_frequency_quality → neuron_grouping` is the first secondary-on-secondary edge, and the family lists `fourier_frequency_quality` *before* `neuron_grouping`. The planner built `projected_completed` in list order, so on a fresh run the dependent would plan as **blocked**. Fixed with `_order_secondaries` (stable topological sort over intra-secondary `depends_on` edges) in `planner.py`; raises on cycles. This is a general planner correctness improvement, not modadd-specific.
- **DataView straggler** (`parameters.embeddings.fourier_coefficients` in `views/dataview_universal.py`) read `dominant_frequencies` directly — the tabular twin of the figure view REQ_127 re-pointed. Folded in (consistent with the REQ_131 "include the views" decision): re-pointed through the existing `_adapt_embedding_coefficients_legacy` adapter over `weight_basis_projection`'s embedding site.
- **Metric interpretation note.** `quality_score` now answers "how well do the frequencies the MLP's neurons actually organize around cover the task?" — a neuron-side measure, distinct from the old embedding-energy one. `coverage_hard` is the binary-set companion (equals the old metric's score *if* the active set matched the old embedding-threshold set), kept as an interpretive reference. Whether this resurrects usable signal is an empirical question for canon analysis (see [[frequency-choice-frame]]).

## Notes

- **Closure feeds REQ_102.** When `fourier_frequency_quality` no longer reads `dominant_frequencies` and no other consumer remains, REQ_102 retires `dominant_frequencies` (→ `weight_basis_projection`, embedding site).
- **Sibling, not bundled:** the `neuron_freq_clusters`/`neuron_freq_norm` consumer migration is a *separate* follow-up (candidate REQ_131), best sequenced after REQ_128 so its backbone consumers (`neuron_dynamics`, `freq_group_weight_geometry`, `neuron_group_pca`) re-point to the granular `activation_basis_projection` with selective/lazy loading rather than re-importing whole-artifact memory pressure.
