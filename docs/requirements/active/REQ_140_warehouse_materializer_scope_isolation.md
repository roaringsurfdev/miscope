# REQ_140: Warehouse Materializer Scope + Per-Analyzer Isolation

**Status:** Active (defect surfaced during REQ_110 smoke testing).
**Priority:** High — a Force=True run over any variant carrying a deprecated/stale on-disk analyzer silently loses dashboard views.
**Branch:** `feature/REQ_110_lakehouse_surface`
**Parent:** REQ_110 (Lakehouse Surface) — hardens the 110-A columnar materializer.
**Dependencies:** 110-A (`warehouse.writer.materialize_variant_columnar`), 110-D (`neuron_frequency.load` self-heal).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

A Force=True analysis run, on a variant whose `artifacts/` directory still holds a
**deprecated analyzer folder**, fails its post-run summary and silently drops
dashboard views. Deleting `artifacts/` and re-running fixes it. Root cause is a
single mechanism with two compounding faults:

1. **Scope divergence.** The run plan scopes analyzers to the family's declared
   set (`AnalyzerRegistry.list_for_family` → `family.json`, 23 analyzers).
   `warehouse.writer.materialize_variant_columnar` instead iterates
   `reg.index().analyzers` — **every registered analyzer** (24), which includes
   `gradient_site`: registered in code, **absent from `family.json`**, deprecated,
   with a stale March artifact still on disk. The two paths disagree on "what
   analyzers are in scope for this family."

2. **No per-analyzer isolation.** `materialize_variant_columnar` builds each
   analyzer's field frames in one loop with no error boundary. The first analyzer
   whose on-disk artifact is malformed raises and **aborts the entire columnar
   build**. On `p101_seed485_dseed999`, `gradient_site` raises
   `IndexError: index 3 is out of bounds for axis 0 with size 3` (its stale
   `key_frequencies` axis), killing the pass before any table is written.

The downstream blast radius runs through `neuron_frequency.load`, which is
self-healing (REQ_110D): on a warehouse cache-miss it calls
`variant.warehouse.materialize()` and retries. When materialize aborts, the
`neuron_frequency_attribution` table is never written, the retry fails too, and
`VariantAnalysisSummary` / `build_variant_registry` cannot complete — so the
dashboard renders with missing views. The deprecated folder also costs memory:
materialize stacks frames into pandas up to the point it crashes, then the
self-heal retries the whole pass.

This is the warehouse-side instance of the architectural rule that
**`family.json` is the single source of truth for a family's analyzer scope**
(constraint 2: families are context providers; they declare their analyzers).
The materializer must honor that declaration, not the global registry.

---

## Conditions of Satisfaction

- [ ] **Materializer scopes to the family's declared analyzers.**
  `materialize_variant_columnar` derives its analyzer set from the variant's
  family declaration (the same `list_for_family` source the run plan uses), not
  from `reg.index().analyzers`. `gradient_site` (registered, not in `family.json`)
  is never reached by materialize. One source of truth for analyzer scope across
  run and warehouse.
- [ ] **A single analyzer's failure is contained, not fatal.** A field-frame
  build that raises is caught, recorded, and skipped; every other analyzer still
  materializes and the pass writes its tables (including
  `neuron_frequency_attribution`). One bad artifact can no longer take down the
  whole warehouse.
- [ ] **Failures are visible, not silent.** `MaterializeReport` distinguishes
  `skipped_analyzers` (no data / empty frames — already exists) from a new
  `failed_analyzers` (name + error summary). A failed analyzer is logged at
  warning level. Silence is the failure mode we are removing.
- [ ] **The self-heal path completes on a variant with a deprecated folder.**
  `neuron_frequency.load` succeeds (table present after materialize), so
  `VariantAnalysisSummary` and `build_variant_registry` complete and views render
  — with the deprecated folder still physically on disk.

## Validation

- [ ] **Reproduction, before vs. after** on `p101_seed485_dseed999` (carries both
  a deprecated `gradient_site` artifact and a stale-shape `intragroup_manifold`):
  - *Before:* `variant.warehouse.materialize()` raises `IndexError` on
    `gradient_site`; no tables written.
  - *After:* materialize completes; `gradient_site` is out of scope (never
    attempted); `intragroup_manifold` lands in `failed_analyzers` (stale shape,
    an in-family artifact — a REQ_137 staleness artifact, not a real finding);
    all healthy tables written; `nf.load` succeeds.
- [ ] **First-pass success on a sibling stale variant.** Pick another variant with
  the same profile (dense checkpointing + deprecated analyzers on disk) and confirm
  a Force=True run's *first* pass completes — materialize builds, views render —
  without the delete-and-rerun workaround. This is the real acceptance bar: the
  deprecated folder stays on disk and the run still succeeds. (`p101_seed485_dseed999`
  is left in its current partially-materialized state as a **self-heal canary** —
  used separately to confirm a later run recovers it, not as the first-pass test.)
- [ ] **No regression on the three baselines** (p113/s999/ds598, p109/s485/ds598,
  p101/s999/ds598): the set of materialized tables and their row counts is
  unchanged for in-`family.json` analyzers (the baselines carry no deprecated
  folders, so scope-narrowing is a no-op there and output is byte-identical).
- [ ] **Unit test for isolation:** a synthetic variant with one analyzer whose
  artifact is deliberately malformed materializes every *other* analyzer and
  reports the bad one in `failed_analyzers`.
- [ ] **Unit test for scope:** a family declaring a subset of registered
  analyzers materializes only the declared subset; a registered-but-undeclared
  analyzer with an on-disk artifact is not materialized.

---

## Implementation Notes

- The change is localized to `warehouse/writer.py::materialize_variant_columnar`:
  swap the `reg.index().analyzers` iteration for the family-declared specs, and
  wrap the per-spec `_build_field_frames` + emit in a try/except that appends to
  `report.failed_analyzers`. `MaterializeReport` gains one field.
- `materialize_variant_columnar(variant)` currently takes only the variant; it
  reaches the family via `variant.family`. Use `AnalyzerRegistry.list_for_family`
  intersected with the specs that declare `COLUMNAR` outputs (today's filter).
- This does **not** retire `gradient_site` — that is REQ_102's deprecation
  surface. REQ_140 makes materialize robust whether or not a deprecated analyzer
  is still registered or has a leftover folder.
- Consider (non-blocking) whether `AnalyzerRegistry.list_for_family`'s silent drop
  of declared-but-unregistered names (`if n in _specs`) should become a loud error
  — the inverse drift. Logged as an observation; not required here.

---

## Constraints

Inherits REQ_110's constraints. Defect-specific:
- **One scope source.** Run plan and warehouse materializer must agree on a
  family's analyzer scope; both derive from `family.json` via `list_for_family`.
- **Isolation must not hide failures.** Skip-and-report, never skip-and-silence —
  a stale in-family artifact (e.g. `intragroup_manifold` here) must surface so the
  REQ_137 refresh can act on it.
- **Storage-encapsulation invariant** holds: no new path literals; reach artifacts
  through the loader/accessors as today.

---

## Notes

- **Out of scope — steady-state memory.** Even when it completes,
  `materialize_variant_columnar` stacks every analyzer's full trajectory into
  pandas in one pass, and `build_variant_registry` opens DuckDB across all
  variants — the ~20GB-per-run footprint. That is a separate design stream
  (streaming/columnar materialization, leveraging the warehouse/query surface),
  deliberately deferred until **after REQ_110 closes and merges to `develop`**.
  See `docs/notes/platform_ideas.md` (steady-state analysis memory).
- Surfaced by user smoke testing of the REQ_110 body of work, 2026-06-06. The
  diagnostic harness lives at
  `apps/research/sketches/profile_analysis_memory.py`.
