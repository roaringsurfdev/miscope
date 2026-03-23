# REQ_074: Variant Outcome Registry

**Status:** Active
**Priority:** High
**Related:** REQ_065 (Second Descent Diagnostics), REQ_052 (Frequency Quality Scoring), REQ_058 (Band Concentration)
**Last Updated:** 2026-03-15

---

## Problem Statement

Findings about individual variants are scattered: learned frequencies are re-derived from whichever probe is in scope, second descent timing lives only in loss curve visuals, handshake failure is inferred informally, and categorical summaries (clean grokker, failed handshake, long grokker) exist only in research notes. This causes interpretation drift — the same model may be described differently depending on which analysis lens was most recently in view.

There is also a practical gap: when asked to reason across variants, both the researcher and the analysis system have to re-derive the same facts repeatedly. A lightweight persistent summary per variant, plus a family-level index, would give both human and AI collaborators a single authoritative reference.

The handshake failure case is now well-defined enough to detect algorithmically: a frequency is committed if neurons specialize to it during the second descent window, and fails the handshake if it is absent from the model's final specialization profile. Capturing this as a computable flag — not just a research note — makes it testable across all variants.

---

## Conditions of Satisfaction

### 1. Per-variant `variant_summary.json`

A computation (analyzer, script, or pipeline stage) produces `variant_summary.json` in each variant's results directory: `results/{family}/{variant}/variant_summary.json`.

The file must contain:

**Identity**
- `prime`: int
- `model_seed`: int
- `data_seed`: int
- `family`: str
- `computed_at`: ISO 8601 timestamp

**Learned frequencies**
- `learned_frequencies`: list[int] — frequencies with neuron specialization fraction above `canonical_specialization_threshold` at the final available checkpoint
- `learned_frequency_count`: int
- `canonical_specialization_threshold`: float — the threshold used (stored for reproducibility)

**Second descent window** (None if model never entered second descent)
- `second_descent_onset_epoch`: int | None — first epoch where test loss has dropped ≥ 80% from peak
- `second_descent_completion_epoch`: int | None — first epoch after onset where test loss is stable below `grokking_threshold` for at least 500 epochs (or end of training if shorter)
- `second_descent_survived`: bool | None

**Handshake analysis** (None if second descent was not detected)
- `committed_frequencies_at_onset`: list[int] | None — frequencies above threshold at `second_descent_onset_epoch`
- `handshake_failures`: list[int] | None — frequencies in `committed_frequencies_at_onset` that are absent from `learned_frequencies`
- `handshake_succeeded`: bool | None — True if `handshake_failures` is empty

**Outcome metrics**
- `failure_mode`: str — one of: `healthy`, `late_grokker`, `degraded_recovery`, `degraded`, `no_grokking` (per REQ_065 classification logic)
- `max_resid_post_circularity`: float | None — maximum circularity achieved at `resid_post` site across all checkpoints (requires `repr_geometry` artifact)
- `final_specialized_frequency_count`: int | None — same as `learned_frequency_count` (kept for clarity)
- `peak_test_loss_epoch`: int
- `final_test_loss`: float

### 2. Family-level `variant_registry.json`

A `variant_registry.json` in the family root (`results/{family}/variant_registry.json`) that is an array of objects, one per variant, containing every field from `variant_summary.json` plus a `variant_id` string (`"{prime}_{model_seed}_{data_seed}"`).

This file must be self-contained and readable without loading any artifact. It is the primary cross-variant lookup table.

### 3. Canonical threshold

A `canonical_specialization_threshold` is defined in one place (config or rules object) and used consistently for both `learned_frequencies` and `committed_frequencies_at_onset`. Default: fraction of `d_mlp` neurons required to qualify (suggest 0.10 as starting value — same order as neuron dynamics views). The threshold value is stored in every summary it was used to produce.

### 4. Regeneration

Running the computation again overwrites the existing summary. The timestamp updates. This is the correct behavior — summaries should always reflect the latest artifacts and threshold settings.

### 5. Registry rebuild

A single call can regenerate `variant_registry.json` from all existing `variant_summary.json` files in the family. The registry does not require re-running artifact analysis — it aggregates already-computed summaries.

### 6. Graceful degradation

- If `neuron_dynamics` artifact is absent: `learned_frequencies`, `committed_frequencies_at_onset`, `handshake_*` fields are `null`.
- If `repr_geometry` artifact is absent: `max_resid_post_circularity` is `null`.
- The summary is still written; absent fields do not block computation of present fields.

### 7. Tests

- Unit: `handshake_failures` correctly identifies frequencies in committed but not learned
- Unit: `handshake_failures` is empty when all committed frequencies survive to final epoch
- Unit: `second_descent_completion_epoch` is None when test loss never stabilizes
- Unit: `canonical_specialization_threshold` stored in output matches threshold used to compute `learned_frequencies`
- Integration: summary computes without error on at least one real variant with all artifacts present
- Integration: `variant_registry.json` contains one entry per variant in the family

---

## Constraints

- `variant_summary.json` must be human-readable JSON — not binary, not pickle.
- The file must not contain large arrays. All values are scalars, short lists, or short strings. If a computed intermediate is large, store only its summary (e.g., peak value and epoch, not the full series).
- Computation must use existing artifacts only. No new per-epoch analyzer pass is required.
- `second_descent_onset_epoch` logic reuses or delegates to REQ_065 implementation to avoid divergence.
- The registry is append-safe: if a variant's summary is updated, the registry entry for that variant updates; other entries are unchanged.

---

## Notes

- **Why a separate registry file?** The researcher and AI collaborator both benefit from a single readable file that answers "what did each variant learn?" without filesystem traversal or artifact loading. The registry makes cross-variant questions (e.g., "which models had handshake failures?") answerable in seconds.
- **Handshake failure definition:** A frequency that neurons committed to during the second descent window but that failed to survive to the end of training. This is the computational definition of the p=113/dseed999 failure mode (Freq 40: neuron commitment, failed attention-MLP handshake). It may not capture all failure modes but it captures the one we can now detect.
- **Canonical threshold rationale:** The neuron dynamics view uses a runtime slider. The summary artifact needs a fixed canonical value for reproducibility. Storing the threshold used in the output file means any future re-analysis at a different threshold is clearly distinguishable from prior runs.
- **Future:** Once per-input trace analysis (REQ_075) is implemented, pair-level graduation statistics could be added to the summary (e.g., "epoch at which 90% of training pairs were correctly predicted").
