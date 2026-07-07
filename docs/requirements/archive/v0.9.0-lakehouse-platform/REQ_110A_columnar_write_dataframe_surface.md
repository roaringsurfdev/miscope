# REQ_110A: Columnar Write + DataFrame Surface

**Status:** Completed — merged to `develop` 2026-06-06 (REQ_110 lakehouse line).
**Priority:** High — critical-path start of the storage engine.
**Branch:** TBD
**Parent:** REQ_110 (Lakehouse Surface) — this is child task 110-A.
**Dependencies:** REQ_107 (per-field `kind` + coordinate declarations — the write-routing source); REQ_109 (Measurement Primitives — typed results that flatten to tabular form).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Analyzers emit `.npz` artifacts. Cross-variant SQL and the in-memory DataFrame surface need the *columnar* outputs (scalars and series) as long-format Parquet — the internal warehouse. This child delivers that write path and the in-memory DataFrame contract. The DuckDB query layer is 110-C; tensors are 110-B; publication is 110-E. See the parent REQ_110 for the full tabular schema (discriminator columns, PCA/Fourier/shape tables) — this child implements that schema's **columnar write**; it does not redefine it.

---

## Conditions of Satisfaction

- [ ] **Schema-driven routing.** A writer consults each analyzer's REQ_107 schema and writes every `columnar`-kind field to Parquet. `tensor`-kind fields are skipped here (owned by 110-B). No analyzer hand-authors its Parquet/`.npz` split — it is derived from the declared `kind`.
- [ ] **Long-format canonical** with the explicit dimension/discriminator columns defined in REQ_110 (`variant_id` + family params, `epoch`, `group_type`, `group`, `operation_type`, value columns). Wide is a consumer-side `to_wide()` pivot.
- [ ] **Batched per `(variant, analyzer)`, not per-epoch.** The columnar writer accumulates a variant/analyzer's rows and writes few, large, column-oriented files with `epoch` as a *column* — never one Parquet per epoch (the small-files anti-pattern). (Decided gotcha, parent REQ_110.)
- [ ] **Internal warehouse path** sibling to `.npz` artifacts (canonicalized in implementation, e.g. `results/{family}/{variant}/dataviews/`), **gitignored**, deterministically regeneratable.
- [ ] **`pyarrow` writer**; compression snappy/zstd (parent's guidance: snappy/none internal, zstd published).
- [ ] **Columnar catalog rows co-emitted.** Each columnar field's catalog row is written in the same pass as its Parquet payload (the catalog relation is shared with 110-B; this child populates the `columnar` rows). Co-emission, not a later index scan.
- [ ] **In-memory DataFrame surface.** DataView returns long-format pandas; `BoundDataView.to_wide(index, columns, values)` wraps `pandas.pivot`; cross-variant concatenation (`pd.concat([...])`) is a no-op because `variant_id` is a column. Each DataView documents its natural `to_wide()` index.

## Validation

- [ ] Regenerate one variant's columnar Parquet from checkpoints; the materialized schema matches the analyzer's declared schema (column names + dtypes).
- [ ] File-count check: a multi-thousand-epoch analyzer produces few `(variant, analyzer)` Parquet files, not thousands of per-epoch files.
- [ ] `pd.concat` across three variants yields one long-format frame with no schema reconciliation.

---

## Constraints

Inherits REQ_110's constraints. Chunk-specific:
- **Columnar only.** Tensor fields are out of scope (110-B); this child must not change `.npz` blob output.
- **No per-epoch Parquet files.** Batched per `(variant, analyzer)`.
- **Internal Parquet gitignored**, never committed.

---

## Notes

- Parallelizable with 110-B (both consume only REQ_107 declarations; they share the catalog relation but write disjoint row kinds).
- The conformed-dimension collapse and consumer migration are 110-D, not here — this child only *produces* the columnar warehouse.
