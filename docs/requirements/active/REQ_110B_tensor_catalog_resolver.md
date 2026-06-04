# REQ_110B: Tensor Catalog + Resolver

**Status:** Draft
**Priority:** High — the non-columnar half of the storage engine; parallel with 110-A.
**Branch:** TBD
**Parent:** REQ_110 (Lakehouse Surface) — this is child task 110-B.
**Dependencies:** REQ_107 (per-field `kind` + coordinate declarations — tensor fields and their coordinate keys).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

`.npz` tensors (raw weights, activations, bases) are opaque today: to relate them across analyzers you hand-load files by name. This child brings them into the same queryable surface as the columnar metrics — *without changing the blob format* — by indexing each `tensor`-kind field as a coordinate-keyed descriptor and adding a resolver that materializes only what a query selects. Design sketch: [`drafts/catalog_design/catalog.py`](../drafts/catalog_design/catalog.py).

---

## Conditions of Satisfaction

- [ ] **Unified catalog relation** (shared with 110-A). One queryable Parquet relation indexes every field with its coordinate columns (`variant_id` + family params, `epoch`, `site`, `group`/`group_type`, `neuron`/`row_id`, `frequency` as applicable) and a `kind` discriminator. This child emits the `tensor` rows; each carries a `TensorRef`.
- [ ] **`TensorRef` is address-only:** `uri`, `member` (key within the container; `""` for a single `.npy`), `dtype`, `shape`, `codec` (`npz` / `npz_compressed` / `npy`). No payload. Declared `dtype`/`shape` come from the analyzer's REQ_107 schema.
- [ ] **Selection is SQL over descriptors.** Joining/filtering the catalog (incl. against run-config) returns a result set of descriptors; no array is touched to answer a metadata question. Result order is pinned for reproducibility.
- [ ] **`TensorResolver` materializes only the selected tensor rows**, batched per container so each archive opens exactly once (`NpzFile` members read lazily; `.npy` memory-mapped). Columnar rows handed to it are ignored.
- [ ] **Reproducibility guard.** On load, assert array `shape`/`dtype` match the descriptor; fail loud on mismatch. The catalog asserts; the resolver checks. The blob bytes remain authoritative.
- [ ] **Co-emission.** Each tensor field's catalog row is written in the same pass as its blob — never a later scan-and-index.
- [ ] **Resolved arrays feed analyzers.** Resolver returns `{id -> ndarray}` ready for PCA/DMD/Fourier — the tensor plane is upstream of new columnar/tensor fields.
- [ ] **Family-owned keys.** Descriptor coordinate columns use the family's declared key (`variant_id` + `family.domain_parameters`), replacing the sketch's hardcoded `model/seed/dataset`.

## Validation

- [ ] A SQL filter selecting a subset of tensors (e.g. `kind='tensor' AND site='mlp_out' AND epoch BETWEEN 20000 AND 30000`) returns descriptors while touching zero array bytes.
- [ ] The resolver materializes exactly the selected rows, opening each container once; an injected shape/dtype mismatch fails loud with a clear message.
- [ ] A round-trip (write tensor + descriptor → SQL select → resolve) reconstructs the original arrays.

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
