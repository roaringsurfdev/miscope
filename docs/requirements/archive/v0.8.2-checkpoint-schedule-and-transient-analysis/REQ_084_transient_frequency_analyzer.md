# REQ_084: Transient Frequency Analyzer

**Status:** Draft
**Branch:** TBD
**Priority:** High — motivated by multi-session exploratory analysis (2026-03-31 to 2026-04-01)

---

## Problem

During second descent, some models transiently recruit neurons to frequency groups
that ultimately collapse before training ends. These **transient frequency groups**
are currently invisible to the pipeline:

- `neuron_group_pca` groups neurons by **final-epoch** dominant frequency. Neurons that
  were organized around a transient frequency are silently distributed across final groups,
  where they appear as diffuse noise rather than as a coherent historical signal.
- `variant_summary.json` captures only final learned frequencies — transient frequencies
  that came and went leave no record.
- No metric currently captures the **homeless neuron fraction**: neurons that passed
  through a transient group but failed to commit to any frequency by the final epoch.

This invisibility matters because transient frequencies appear to be mechanistically
related to second-descent failure modes. Evidence from manual analysis across 8+ variants:

- Transient groups leave a detectable signature in centroid class PCA: the rate of
  variance decline accelerates at exactly the epoch the transient frequency crosses
  threshold. In p113/s999/ds999, the baseline decline rate of ~0.1%/step jumped to
  ~0.9%/step at transient onset — a 9× acceleration that persisted as permanent
  geometric damage (79.1% → 52.3% total variance).
- **Homeless fraction** (neurons in transient groups that are uncommitted at the final
  epoch) correlates with outcome. In p113/s999/ds999 (no recovery): 48.6%. In
  p101/s485/ds999 (no recovery): 68.9%. The recovering variant p101/s999/ds999
  appears to have smaller transient groups that fall below the 10% detection threshold.
- Transient groups show **blob geometry** at peak (PC1 ≈ PC2 ≈ 50%), not ring geometry.
  High per-neuron frac_explained is not evidence of structural commitment — neurons can
  be functionally computing a frequency without having joined a weight-space coalition.
- **Frequency proximity** is a possible upstream cause: across multiple variants, the
  multi-stream view shows MLP committing to freq N while Embeddings carry a residual
  bump at N±1. This mixed-site signal may create transient recruitment of neurons to
  the "wrong" neighboring frequency before settling.

A formal analyzer is needed to make these patterns measurable across all variants
rather than hand-examined one at a time.

---

## Goals

1. Implement a cross-epoch analyzer `transient_frequency` that identifies ever-qualified
   frequencies, their peak epochs, and the fate of their neuron cohorts.
2. Add homeless fraction and transient group metadata to `variant_summary.json`.
3. Expose transient group data through the view catalog so notebook and dashboard
   consumers can access it without reimplementing the detection logic.
4. Condense the notebook-level prototype code (`neuron_group_pca_scatter.ipynb`) into
   thin calls to the new analyzer and views.

---

## Conditions of Satisfaction

### CoS 1 — `transient_frequency` cross-epoch analyzer

A new cross-epoch analyzer `TransientFrequencyAnalyzer` in
`src/miscope/analysis/analyzers/transient_frequency.py`.

`requires = ["neuron_dynamics"]`

Artifact keys produced:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `ever_qualified_freqs` | `(n_transient,)` | int32 | 0-indexed frequencies that ever crossed the transient threshold |
| `is_final` | `(n_transient,)` | bool | True if the frequency is in the final learned set |
| `peak_epoch` | `(n_transient,)` | int32 | Epoch of maximum committed neuron count |
| `peak_count` | `(n_transient,)` | int32 | Committed neuron count at peak epoch |
| `peak_members` | ragged — see note | int32 | Neuron indices committed at peak epoch per group |
| `committed_counts` | `(n_epochs, n_ever_qualified)` | int32 | Committed count per frequency per epoch |
| `homeless_count` | `(n_transient,)` | int32 | Neurons in group at peak that are uncommitted at final epoch |
| `epochs` | `(n_epochs,)` | int32 | Epoch array (matches neuron_dynamics epochs) |

**Ragged storage note:** `peak_members` cannot be stored as a rectangular array since
groups have different sizes. Store as flat `peak_members_flat` (int32) and
`peak_members_offsets` (int32, length n_transient+1) using the standard offset encoding.
Loader helper `load_peak_members(artifact, group_idx)` returns the slice for one group.

**Threshold parameters** (configurable, stored in artifact metadata):
- `neuron_threshold`: per-neuron max_frac gate (default 0.70)
- `transient_canonical_threshold`: fraction of d_mlp required to qualify (default 0.05)
- `final_canonical_threshold`: fraction of d_mlp required to count as "learned" in the
  final set (default 0.10 — matches `variant_summary` semantics)

Using two separate thresholds is intentional: transient detection needs 5% sensitivity;
"learned frequency" classification needs 10% to match existing pipeline semantics.

### CoS 2 — Homeless fraction in `variant_summary.json`

Add to `compute_variant_summary()` in `variant_summary.py`:

```python
"transient_frequencies": [int, ...],        # 1-indexed; ever-qualified but not final
"transient_frequency_count": int,
"homeless_neuron_count": int,               # sum across all transient groups
"homeless_neuron_fraction": float,          # homeless_count / d_mlp
"transient_detection_threshold": float,     # 0.05
```

These fields are `None` when the `transient_frequency` artifact is absent.

### CoS 3 — View catalog entries

Three new views registered in `universal.py`:

**`transient.committed_counts`** — cross-epoch line chart of committed neuron count per
ever-qualified frequency. Transient frequencies (not in final set) rendered as dashed
red lines; persistent frequencies as solid. Accepts optional `show_persistent=True`
kwarg to include final frequencies alongside transients for context.

**`transient.peak_scatter`** — PC1×PC2 scatter of a transient group's neurons at its
peak epoch, colored by final dominant frequency. Requires `freq` kwarg (0-indexed).
Reveals whether the group had ring geometry (PC1 >> PC2) or blob geometry (PC1 ≈ PC2)
at the moment of maximum specialization.

**`transient.pc1_cohesion`** — PC1 variance explained of a transient group's neuron
cohort projected onto the peak-epoch basis, tracked across all epochs. Reveals whether
coherent weight-space structure was forming and collapsing, or was never present.
Requires `freq` kwarg.

All three views load from the `transient_frequency` cross-epoch artifact. The
`transient.peak_scatter` and `transient.pc1_cohesion` views also load
`parameter_snapshot` to compute W_in projections.

### CoS 4 — Notebook condensation

The prototype cells in `neuron_group_pca_scatter.ipynb`
(`compute_committed_counts`, `load_ever_specialized_groups`, `run_deep_analysis`,
and their callers) are replaced with thin view calls:

```python
v = family.get_variant(prime=113, seed=999, data_seed=999)
v.view("transient.committed_counts").show()
v.view("transient.peak_scatter", freq=39).show()
v.view("transient.pc1_cohesion", freq=39).show()
```

The prototype helper functions are removed from the notebook once the views are
verified to produce equivalent output. A note cell explains the migration.

### CoS 5 — `variant_summary.json` regeneration

Re-run `write_variant_summary()` for all 30+ variants in
`results/modulo_addition_1layer/` after the new fields are added. Re-run
`build_variant_registry()` to update `variant_registry.json`.

This requires that the `transient_frequency` artifact has been computed for each
variant. If a variant is missing the artifact, the new fields are `None` (not an error).

---

## Constraints

- `requires = ["neuron_dynamics"]` only — no `parameter_snapshot` dependency in the
  analyzer itself. W_in loading for scatter/cohesion views happens in the view loader,
  not the analyzer. This keeps the analyzer fast and the artifact small.
- Thresholds must be stored in artifact metadata so the values used to produce a given
  artifact are always recoverable.
- The `transient_canonical_threshold` (5%) and `final_canonical_threshold` (10%) must
  remain distinct constants — do not unify them. They serve different semantic purposes.
- Do not modify `neuron_group_pca` grouping logic as part of this requirement. The
  question of whether `neuron_group_pca` should optionally use ever-specialized
  grouping is a separate decision deferred until this analyzer's output is validated
  across more variants.
- Views are universal instruments — no family-specific logic in the analyzer or views.

---

## Out of Scope

- **Frequency proximity metric**: identifying freq N±1 co-activation across sites
  (MLP commits to N, Embeddings carry N±1 bump). This is the hypothesized upstream
  cause of transient recruitment and warrants its own requirement once the transient
  analyzer confirms the pattern across variants.
- **Manifold capacity estimation**: predicting whether a given frequency portfolio can
  be geometrically realized without aliasing given neuron counts. Requires theoretical
  grounding from the frequency distribution data this requirement will produce.
- **Dashboard surface**: no new dashboard page in this requirement. The view catalog
  entries make the data accessible; dashboard integration follows once the metrics
  are validated.
- **Automated neuron_group_pca regrouping** by ever-specialized membership (rejected
  for now — not enough evidence that transient group PCA is more informative than
  the current final-epoch grouping for the existing views).

---

## Notes

**Motivation context (2026-03-31 to 2026-04-01):** Exploratory analysis in
`notebooks/neuron_group_pca_scatter.ipynb` across p113/s999/ds999, p101/s999/ds999,
p101/s485/ds999, and ~6 additional variants from `notes/findings_interfering_frequencies.md`.
Key findings that directly motivate this requirement:

- Blob geometry at peak (PC1 ≈ PC2 ≈ 50%) is the structural signature of a transient
  group. It indicates functional tuning without weight-space coalition — the group is
  doing the computation but has not committed structurally. This is distinct from a
  healthy group at early training (which also has diffuse geometry) because transient
  groups are at *peak* committed count when the blob is measured.
- The homeless fraction is the most direct outcome metric: 48.6% in p113/s999/ds999,
  68.9% in p101/s485/ds999, both non-recovering. The recovering variant (p101/s999/ds999)
  has transient groups below the 10% threshold — smaller disturbances.
- Centroid class PCA variance decline accelerates 9× at transient onset in p113/s999/ds999,
  producing permanent geometric damage that the model never repairs.

**Naming note:** "Interfering frequencies" (from `notes/findings_interfering_frequencies.md`)
may ultimately be the more precise framing — the transient recruitment may be a symptom
of frequency proximity / signal interference rather than an independent phenomenon.
This requirement uses "transient" because it describes the observable (arrival and
departure of a frequency group) rather than the hypothesized mechanism. Once the
frequency proximity metric is built (future requirement), the causal chain can be tested.

**Prototype code location:** `notebooks/neuron_group_pca_scatter.ipynb`, cells
`atmkejub9wj` (helpers), `f84d4m0yr5` (summary), `k23qa5twenf` (committed count plot),
`jwgeajdaoka` (scatter at peak), `ppd5vxp0a3n` (PC1 cohesion), `5u6j2aosdre`
(homeless fraction), `prkq3y5rqd` (5% threshold comparison), and the `run_deep_analysis`
function in the final cell block.
