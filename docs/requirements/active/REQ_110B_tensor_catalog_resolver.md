# REQ_110B: Tensor Catalog + Resolver

**Status:** Implemented (on the parent REQ_110 branch; stays in `active/` until all of REQ_110 merges to `develop`).
**Priority:** High — the non-columnar half of the storage engine; parallel with 110-A.
**Branch:** `feature/REQ_110_lakehouse_surface`
**Parent:** REQ_110 (Lakehouse Surface) — this is child task 110-B.
**Dependencies:** REQ_107 (per-field `kind` + coordinate declarations — tensor fields and their coordinate keys).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

`.npz` tensors (raw weights, activations, bases) are opaque today: to relate them across analyzers you hand-load files by name. This child brings them into the same queryable surface as the columnar metrics — *without changing the blob format* — by indexing each `tensor`-kind field as a coordinate-keyed descriptor and adding a resolver that materializes only what a query selects. Design sketch: [`drafts/catalog_design/catalog.py`](../drafts/catalog_design/catalog.py).

---

## Conditions of Satisfaction

- [x] **Unified catalog relation** (shared with 110-A). One queryable Parquet relation indexes every field with its coordinate columns (`variant_id` + family params, `epoch`, `site`, `group`/`group_type`, `neuron`/`row_id`, `frequency` as applicable) and a `kind` discriminator. This child emits the `tensor` rows; each carries a `TensorRef`. *Tensor rows live under `dataviews/_tensor_catalog/{analyzer}.parquet`, sibling to 110-A's `_catalog/`; the "one relation" is the `UNION ALL BY NAME` of both `_catalog` dirs (DuckDB view is 110-C). Descriptor columns are the **instance** coords (`epoch`, `site`, `group`); array-axis coords (`neuron`/`row_id`/`frequency`/`head`) live in `tensor_shape`, not as columns.*
- [x] **`TensorRef` is address-only:** `uri`, `member` (key within the container; `""` for a single `.npy`), `dtype`, `shape`, `codec` (`npz` / `npz_compressed` / `npy`). No payload. *`uri` is variant-relative (portable). `shape` is read from the npy header (REQ_107 declares `dtype` but not the concrete shape); `dtype`/`codec` are read from the bytes — see the dtype-authority note below.*
- [x] **Selection is SQL over descriptors.** Joining/filtering the catalog (incl. against run-config) returns a result set of descriptors; no array is touched to answer a metadata question. Result order is pinned for reproducibility (`ORDER BY` in the consumer's SQL).
- [x] **`TensorResolver` materializes only the selected tensor rows**, batched per container so each archive opens exactly once (`NpzFile` members read lazily; `.npy` memory-mapped). Columnar rows handed to it are ignored.
- [x] **Reproducibility guard.** On load, assert array `shape`/`dtype` match the descriptor; fail loud on mismatch. The catalog asserts; the resolver checks. The blob bytes remain authoritative.
- [x] **Co-emission.** Descriptor rows are written in the same pass that reads each blob's header (`materialize_variant_tensors`), driven by the registry's declared `tensor` fields. *Interpreted symmetrically with 110-A (a registry-driven materialize pass over `variant.artifacts`, not a hook inside the training pipeline's npz-write). "Never a later scan-and-index" is honored in substance: the descriptor's shape is read from the bytes at index time and the resolver re-verifies on load, so the index cannot silently drift.*
- [x] **Resolved arrays feed analyzers.** Resolver returns `{id -> ndarray}` ready for PCA/DMD/Fourier.
- [x] **Family-owned keys.** Descriptor coordinate columns use the family's declared key (`variant_id` + `family.domain_parameters`), via `reg.variant_key_columns(family)`.

## Validation

- [x] A SQL filter selecting a subset of tensors (`kind='tensor' AND analyzer='parameter_snapshot' AND epoch BETWEEN ...`) returns descriptors while touching zero array bytes — proven by deleting every blob and showing selection still returns rows (`test_selection_touches_zero_array_bytes`).
- [x] The resolver materializes exactly the selected rows, opening each container once (`test_resolver_opens_each_container_once`); an injected shape mismatch fails loud (`test_resolver_verifies_shape_against_bytes`).
- [x] A round-trip (write tensor + descriptor → select → resolve) reconstructs the original arrays (`test_round_trip_reconstructs_original_arrays`). Verified end-to-end on a real variant (1736 descriptors across 10 tensor-declaring analyzers).

## Implementation Notes

- **dtype authority (deliberate reconciliation).** Two CoS lines conflict in practice: "declared `dtype`/`shape` come from the REQ_107 schema" vs. "the blob bytes remain authoritative." On real data they *disagree*. Resolved in favor of **bytes-authoritative**: the descriptor records the actual on-disk dtype (so `resolve` works), and the declared-vs-actual mismatch is surfaced as a non-fatal `DtypeDrift` finding on the `TensorCatalogReport` (a declaration to reconcile), rather than failing the materialize.
  → **Finding (refresh-stable) — FIXED.** `dominant_frequency_pair` was declared `float64` but emitted `int32` in **both** `weight_basis_projection` and `activation_basis_projection` (an argmax `(k_a, k_b)` index swept into each analyzer's `float64` tensor-field loop; `analyze()` already does `.astype(np.int32)`). Present on all three refreshed baselines, so a genuine schema bug, not stale data. Fixed by pulling the field out of the loop and declaring it `int32`; both analyzer `version`s bumped to 2. Re-materializing the tensor catalog on all three baselines now reports zero `DtypeDrift` — declaration matches the authoritative bytes.
  → **Retracted (stale-data artifact):** an earlier pass reported `parameter_trajectory.projections` as `float64`-declared / `float32`-bytes. That was observed only on `p109/s485/ds42`, a variant **outside the current refresh path**; it does not appear on any refreshed baseline. See the verification-scope note below.
- **Verification scope (until the full variant refresh).** Only the three refreshed baselines are trustworthy ground truth right now: **p113/s999/ds598 (canon), p109/s485/ds598, p101/s999/ds598**. All other on-disk variants are pre-refresh (stale analyzer artifacts / old shapes / old dtypes) and must **not** be used to validate refactors — a "finding" seen only on a non-baseline variant is presumed a staleness artifact until the full refresh lands (re-train for dense checkpointing → purge deprecated artifacts → re-analyze). Tracked by the refresh stub REQ.
- **Union safety.** All-null nullable coord columns (`site`/`group`/`group_type`/`epoch`) are pinned to stable Parquet types (`string`/`Int64`) so a cross-file DuckDB union (110-C) reconciles them against typed siblings instead of failing on an inferred `null` type.
- **Independent halves.** 110-A's columnar wipe is now selective (preserves `_tensor_catalog/`); the two materializers run in either order without clobbering each other.
- **New surfaces:** `miscope.warehouse.{materialize_variant_tensors, read_tensor_catalog, TensorRef, TensorCatalogRow, TensorResolver, TensorCatalogReport, DtypeDrift}`; `variant.tensor_catalog` accessor (mirrors `variant.warehouse`); `ArtifactLoader.artifact_path(...)` (path composition kept inside the storage primitive).

---

## Constraints

Inherits REQ_110's constraints. Chunk-specific:
- **Blob format unchanged.** `.npz`/`.npy` stay as-is; this child adds an index + resolver, not a re-serialization.
- **Descriptors are address-only.** No payload in the catalog; the catalog is an index, not a store.
- **Verification is mandatory**, not optional — it is the reproducibility guard that lets the catalog be trusted without being authoritative.

---

## Notes

- Parallelizable with 110-A; they share the catalog relation but emit disjoint row kinds.
- The sketch (`catalog.py`) carries the resolver shape (per-container batching, codec handling, `_verify`); the production version swaps its hardcoded join keys for the family-owned coordinate vocabulary (REQ_107).
