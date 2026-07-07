# REQ_110C: DuckDB Query Surface

**Status:** Completed — merged to `develop` 2026-06-06 (REQ_110 lakehouse line).
**Priority:** High — the cross-variant query layer; turns the two materialized planes into one queryable surface.
**Branch:** `feature/REQ_110_lakehouse_surface`
**Parent:** REQ_110 (Lakehouse Surface) — this is child task 110-C.
**Dependencies:** 110-A (columnar Parquet + `_catalog/` rows), 110-B (tensor descriptors + `_tensor_catalog/` rows).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

110-A and 110-B materialize two planes — columnar metrics and tensor descriptors — co-emitting a shared catalog relation split across `dataviews/_catalog/` and `dataviews/_tensor_catalog/`, per variant. Useful data, but to ask a cross-variant question today you still hand-glob Parquet files and `pd.concat`. This child adds the thin DuckDB surface the lakehouse pattern was for: ergonomic views (`SELECT * FROM frequency_spectrum`, not a file path), cross-variant in one `SELECT` (`variant_id` is a column), the unified `catalog` view (`UNION ALL BY NAME` over the `kind` discriminator), and cross-table joins — all resolved natively by DuckDB, no custom join/read logic in `miscope`.

---

## Conditions of Satisfaction

- [x] **DuckDB available as a `miscope` dep.** `duckdb>=1.5.3` (root `pyproject.toml`).
- [x] **`miscope.query.open(...)`** opens a connection bound to a chosen Parquet root and returns a thin `QueryConnection` wrapper (`.sql(...)`, `.df(...)`, `.tables()`, `.con`, context manager). `con.sql("...").df()` is the documented surface. *Exactly one of `family=` / `root=` selects the mode.*
- [x] **DuckDB views over the root for ergonomic naming.** `SELECT * FROM frequency_spectrum`, not `SELECT * FROM 'results/.../frequency_spectrum.parquet'`. Each materialized table becomes a view globbed across every variant (`variant_id` already a column, so the union is cross-variant with no reconciliation).
- [x] **The unified `catalog` view** is the `UNION ALL BY NAME` of the columnar (110-A) and tensor (110-B) descriptor relations over the `kind` discriminator. Built from only the planes that exist (a family may have one without the other) so no empty glob is scanned. Plane-specific columns (`parquet_uri` vs `tensor_uri`/`tensor_shape`/…) coexist null-padded — the index over *every* field, columnar and tensor, in one relation.
- [x] **Cross-table joins work via DuckDB's native Parquet handling** — no custom join logic. `JOIN ... ON (variant_id, epoch)` is plain SQL over the views.
- [x] **Local and HTTP read paths, same surface.** Warehouse (family) mode globs the local variant tree. Bundle mode (`root=<dir|url>`) registers one view per flat `{root}/{table}.parquet`; a URL root lazily `LOAD`s `httpfs` so DuckDB issues range requests. The `con.sql(...)` surface is identical either way.

## Validation

- [x] **Cross-variant query test** — three canonical one-line questions expressible against the warehouse: distinct-variant scan, `frac_explained DESC` ranking, and a kind-breakdown over the unified catalog (`test_canonical_one_line_question`, `test_cross_variant_query_is_a_plain_select`, `test_catalog_unions_both_planes_by_name`).
- [x] **Catalog union test** — both `columnar` and `tensor` rows present in one `catalog` view; a tensor row carries `tensor_uri` with `parquet_uri` null (proof of by-name union, not a positional one).
- [x] **Cross-table join test** — `neuron_frequency_attribution ⋈ fourier_frequency_quality` on `(variant_id, epoch)` runs in DuckDB and spans both variants.
- [x] **Bundle-mode test** — a flat `{root}/{table}.parquet` is queryable by the same surface (`test_bundle_mode_over_flat_local_files`).
- [x] **Verified end-to-end on the three baselines** (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598): 23 table views + `catalog` (432 columnar + 84 992 tensor rows); a SQL selection of `parameter_snapshot.W_E` descriptors resolves through 110-B's `TensorResolver` to the original `(p+1, d_model)` arrays — closing the select→resolve cycle through the query surface.

## Implementation Notes

- **New surface:** `miscope.query.{open, QueryConnection}` (top-level module; also re-exported on `miscope` so both `import miscope.query` and `import miscope; miscope.query` work). Path composition stays in the storage primitive — `warehouse.paths.{family_table_glob, family_catalog_glob, family_tensor_catalog_glob, list_family_tables, family_has_catalog_rows, flat_table_uri}` — so the query module holds no path literal (storage-encapsulation invariant).
- **Two root shapes, one query surface.**
  - *Warehouse (family) mode* — `open(family=...)`. Glob views across `{family.variants_dir}/*/dataviews/...`; the family owns the `variants/` segment, paths.py owns the `dataviews/` layout. Local (globbing needs a filesystem). The day-to-day cross-variant surface — does **not** wait on the publication chain.
  - *Bundle mode* — `open(root=<dir|url>, tables=[...])`. One view per flat file; `tables` required for a URL (a remote dir can't be listed), auto-discovered for a local dir. This is the seam 110-E's published bundles plug into; the live range-request/CORS verification against a real Release URL is 110-E/F's acceptance bar, not network-tested here.
- **View granularity.** Semantic tables are a single `long.parquet` per variant → a clean glob-union across variants. Generic analyzer tables have multiple coord-signature files; their view globs+unions them `BY NAME` (heterogeneous columns null-pad). Semantic tables stay the clean designed query surface; the generic union is a fallback.
- **`tables=` gates data-table views, not the `catalog` index.** The catalog spans the whole warehouse regardless of which tables you select, so it is always registered in warehouse mode (cheap, and it is the index, not a data table).

---

## Constraints

Inherits REQ_110's constraints. Chunk-specific:
- **Re-implement no Parquet reads / joins.** DuckDB handles scanning, range requests, and joins natively; `miscope.query` only turns warehouse addresses into named views.
- **No path literals outside the storage primitive.** All globs/URIs come from `warehouse.paths`.
- **Thin wrapper.** `QueryConnection` delegates to DuckDB; `.con` exposes the raw connection for anything the wrapper doesn't.

---

## Notes

- Depends on both 110-A and 110-B being materialized; it reads their output, adds no schema.
- Next on the critical path: 110-D (consumer migration) re-points renderers/summaries onto this surface and collapses the hand-rolled `variant_registry`/`variant_summary` aggregations into SQL.
