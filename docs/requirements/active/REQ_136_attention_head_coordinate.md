# REQ_136: Head coordinate for attention weight spectra / basis projection

**Status:** Completed (implemented on `feature/REQ_110_lakehouse_surface`)
**Priority:** Medium — corrects a lossy columnar declaration; recommended before REQ_110B.
**Branch:** feature/REQ_110_lakehouse_surface
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Attention weight matrices (`W_Q`, `W_K`, `W_V`, `W_O`) carry a **head axis** that
two analyzers' columnar declarations do not name:

- `weight_spectra.sv` — declared coords `(variant, epoch, site, row_id)`, but the
  on-disk array for an attention site is `(n_heads, n_sv)` (per-head SVD), e.g.
  `(4, 32)`. Non-attention sites (`W_E`, `W_in`, …) are `(n_sv,)` — **no head axis**.
- `weight_basis_projection.dominant_frequency` — same `(n_heads, …)` shape for the
  per-head attention site (`attn_v`).

REQ_110A's flattener folds the undeclared leading axis into `row_id` (lossless in
*values*, but it conflates head index with singular-value index). This is lossy
even for **group-level** head analysis: a consumer cannot recover which head a row
came from, so it cannot cleanly aggregate across heads — the exact operation the
group-level approach needs. Surfaced during REQ_110A review; the fold is currently
documented as intentional in the analyzer declarations (interim honesty note) so
it is not mistaken for a bug.

The root issue is **site-conditional rank**: a single flat `OutputField` cannot
say "extra axis for attention sites only." The declaration mechanism (REQ_107) is
fine; what is lossy is these two analyzers' *use* of it. So this is a small
follow-up, not a reopening of REQ_107.

## Conditions of Satisfaction

- [x] **Add a `HEAD` coordinate** to `Coord` (REQ_107 vocabulary,
  `analysis/output_schema.py`): "attention head index."
- [x] **Uniform-rank columnar emission** for the affected fields: attention sites
  emit `(n_heads, n_sv)`; non-attention sites emit `(1, n_sv)` (a one-line reshape,
  `head = 0`), so the field has uniform rank and declares
  `(variant, epoch, site, head, row_id)`. Chose the uniform-rank approach over a
  dedicated per-head field — keeps one declared shape and one flatten path. Same
  treatment for `weight_basis_projection.dominant_frequency`. `compute_summary`
  collapses the non-attention singleton head axis so per-matrix PR stays scalar
  (no `variant_summary` drift); attention PR stays per-head.
- [x] **Warehouse preserves `head` as its own column** (no flatten behavior change —
  `head` added to the positional-axis set and the canonical coord order only).
  Re-materialized the canonical three; both tables now key on
  `by__variant_epoch_site_head_row_id` — attention sites resolve heads `[0,1,2,3]`,
  non-attention to `[0]`, and `row_id` is the singular-value / output index alone.
- [x] **Dropped the interim honesty note** — `_conform_rank`'s head-fold docstring
  now frames the fold as a defensive fallback, and finding 1 in
  `req110a_decomposition_map.md` is marked resolved.

## Constraints

- **Columnar-scoped.** Do **not** reshape the tensor fields (`u`, `vt`, the coeff
  cubes). A tensor's internal axes are never coordinates (kind/coords convention),
  and 110-B records each blob's *actual* per-head shape in its `TensorRef`
  regardless. Reshaping them would be churn for no gain and would entangle this
  with 110-B.
- Honor the storage-encapsulation invariant: changes are in the analyzers'
  declarations + the columnar emission; the warehouse consumes the declaration
  unchanged.

## Sequencing vs REQ_110B

**Recommended before 110-B, as a soft (churn-avoidance) ordering — not a hard
dependency.** 110-B indexes *tensor* blobs and reads their true per-head shape at
co-emission, so it is not blocked by the missing `head` coordinate. The reason to
land 136 first is to settle the `Coord` vocabulary and the two attention-bearing
analyzers' schemas before 110-B writes descriptors/tests that reference them —
avoiding rework. If 136 is kept columnar-scoped (per the constraint above), the
two are otherwise independent.

## Notes

- Surfaced 2026-06-04 during REQ_110A review; details in
  `docs/notes/req110a_decomposition_map.md` (Implementation status → finding 1).
- `n_heads` for the modadd family is 4; `n_sv` per head is 32. Confirm head count
  comes from the model config, not a hardcode.
- Worth a parallel sanity check: any *other* analyzer whose attention-site output
  silently carries a head axis (audit alongside, don't expand scope reactively).
- Audit result: `activation_basis_projection`'s `attn_pattern` site does carry a
  head axis, but only in its **tensor** fields (coefficient cubes,
  `dominant_frequency_pair`); its sole columnar field is `frequencies` (no head).
  Tensors are not flattened (110-B records true per-head shape), so no columnar
  head loss there — only the two named analyzers were affected. `n_heads` is read
  from the model config (4 for modadd), not hardcoded.
