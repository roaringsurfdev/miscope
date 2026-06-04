# REQ_107: Discoverability Registry (INFORMATION_SCHEMA for Analysis)

**Status:** Draft
**Priority:** Medium-high — cultural complement to REQ_106; not strictly blocking, but high value before publication.
**Branch:** TBD
**Dependencies:** REQ_106 (defines what gets registered: analyzers, DataViews, their schemas).
**Feeds:** REQ_110 (Lakehouse Surface) consumes the per-field `kind` + coordinate declarations added here as its write-routing table and join-key source; the tensor-catalog sketch (`drafts/catalog_design/catalog.py`) consumes the same declarations to index tensor blobs by descriptor. This REQ declares *what each field is and how it is keyed*; REQ_110 persists and queries the columnar fields; the catalog/resolver indexes and materializes the tensor fields.
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Even with clean layering (REQ_106), analyzers re-derive data their author didn't know already existed. The current codebase shows this directly: [`freq_group_weight_geometry._build_group_labels`](../../src/miscope/analysis/analyzers/freq_group_weight_geometry.py#L179) re-implements an argmax over `neuron_freq_norm` not because consuming `neuron_dynamics.dominant_freq` was hard, but because the author was solving the local problem without surveying what was upstream.

The mitigation is **discoverability**. SQL has `INFORMATION_SCHEMA`. Build systems have query interfaces (e.g., `bazel query`). The codebase needs a single canonical surface that enumerates "here is every analyzer field, every DataView, every derivation, with what it means and how to consume it."

Without it, even the cleanest layering will accumulate re-derivation through normal forgetting. The user has wanted this for a while — it has been deferred to research several times, justifiably. With research now driving toward shareable publication, the deferral cost has flipped: a published library that doesn't show external researchers what it contains is failing at its primary job.

---

## Conditions of Satisfaction

### Registry surface

- [ ] A canonical registry module — proposed location `miscope.registry` or `miscope.core.registry` — exposes:
  - `analyzers()` → list of registered analyzers with their output schemas and brief semantic descriptions per field.
  - `dataviews()` → list of registered DataViews with their source dependencies, output schemas, and brief semantic descriptions.
  - `field(name)` → reverse lookup: returns the producing analyzer (or DataView), the consumers, and the field's semantic description.
  - `search(query)` → free-text or substring match over names and descriptions; returns matching analyzers and DataViews.
- [ ] Registry is browsable from a notebook with one import. The intended researcher experience: open a notebook, type `miscope.registry.search("frequency")`, see a ranked list of frequency-related analyzers and views with one-line descriptions.

### Schema declaration

- [ ] Every Analyzer declares its output schema explicitly: field name, dtype, brief semantic description. Today field names emerge implicitly from the keys returned by `analyze()`; this REQ makes the declaration explicit and registered.
- [ ] Every DataView declares: source dependencies (analyzer name + minimum version + fields consumed), output schema (column names + dtypes), and a brief semantic description.
- [ ] Schema registration is enforced at registry-load time. An analyzer or DataView without a declared schema fails registration loudly.
- [ ] Schema declarations live with the analyzer/dataview class definition, not in a separate manifest. Locality.

### Field kind + coordinate keying (the write-routing keystone)

The schema declaration is not only a discoverability surface — it is the **write-routing table** that REQ_110 (Lakehouse Surface) and the tensor catalog consume. Two declarations per field carry that weight:

- [ ] **Each output field declares a `kind`: `columnar` or `tensor`.** `columnar` = scalar or series that flattens to a Parquet row (queryable, joinable); `tensor` = dense array kept as a blob and referenced by descriptor (retrieved for linear algebra, not queried). This mirrors `DataViewField.field_type` (`"dataframe"`/`"ndarray"`) already in [`views/dataview_catalog.py`](../../src/miscope/views/dataview_catalog.py#L37) and `PayloadKind` in the catalog sketch ([`drafts/catalog_design/catalog.py`](../drafts/catalog_design/catalog.py)). The `kind` lets the pipeline route a field to Parquet vs. a tensor blob *from the declaration*, so REQ_110's per-analyzer Parquet/`.npz` choice becomes **derived, not authored**.
- [ ] **Each field declares the coordinates that key it**, drawn from a canonical vocabulary: universal coords (`epoch`, `neuron`/`row_id`, `site`, `group`/`group_type`, `frequency`) plus the variant identity coords. The coordinate declaration is what makes a cross-analyzer join both *possible* (shared keys exist) and *discoverable* (`field(name)` reports what a field is keyed by). It is the structural fix for the re-derivation this REQ's Problem Statement cites: `neuron_dynamics.dominant_freq` keyed by `(variant_id, epoch, neuron)` becomes a join target instead of something each consumer recomputes (the pattern recurs across `activation_basis_projection`, `neuron_dynamics`, `neuron_grouping`, `freq_group_weight_geometry`, `intragroup_manifold` — the same neuron→freq/group relation built 4–5 times, consistent only by coincidence of threshold).
- [ ] **Variant identity coordinates come from `family.domain_parameters`, not a hardcoded key.** The variant key is family-owned and family-extensible ([`ModelFamily.domain_parameters`](../../src/miscope/families/base_model_family.py#L112) + `variant_pattern`; `Variant.name` is the composed handle). The registry records the family's declared key composition; the catalog/Parquet carry `variant_id` (the opaque composed handle — for cross-family joins) **plus** the family's declared param columns (for `WHERE prime > 100`). The hardcoded `variant_id = "{prime}_{model_seed}_{data_seed}"` in [`analysis/variant_summary.py`](../../src/miscope/analysis/variant_summary.py#L45) is the anti-pattern this replaces; it breaks for any non-modadd family.
- [ ] **Catalog rows are co-emitted with their payload.** When the pipeline writes a field (a Parquet row or a tensor blob), it emits that field's catalog row in the *same* write — never a later scan-and-index pass. Co-emission makes "the index cannot drift from the bytes" true by construction, the same way schema-lives-with-code makes "the declaration cannot drift from the code" true.

### Drift detection

- [ ] When an analyzer's output schema changes (field added, removed, dtype changed), every DataView that declares it as a source must either declare compatibility with the new version, or fail loudly at registry-load time.
- [ ] CI test: `python -c "import miscope.registry; miscope.registry.load()"` succeeds. A failure at registry-load is the mechanical signal that an upstream change broke a downstream consumer.

### Notebook ergonomics

- [ ] `help(miscope.registry)` returns a guided tour: examples for each entry point.
- [ ] `miscope.registry.dataviews()` and `miscope.registry.analyzers()` return rendered tables in Jupyter (rich `_repr_html_` or pandas DataFrame), not raw dicts.
- [ ] A "first 5 minutes" example in `templates/` (REQ_103) walks a researcher through: query the registry → find the relevant DataView → load it → write the 5-line pandas query.

### CLAUDE.md update

- [ ] CLAUDE.md adds a guidance line: before authoring a new analyzer or inlining a derivation, check the registry for existing fields that match the intended computation. This codifies the discoverability-as-first-step culture.

---

## Constraints

**Must:**

- The registry is a Python module — single source of truth, in-process, no separate database or service.
- Schema declarations live with the analyzer/dataview, not in a parallel manifest. Drift between code and declaration is impossible by construction.
- The registry is the canonical answer to "do we already have this?" Documented as such in CLAUDE.md.

**Must avoid:**

- Building a heavy schema language. Plain Python dataclasses or dicts are sufficient.
- Treating the registry as a substitute for documentation. The registry enumerates and cross-references; it does not explain. Long-form explanations belong in docstrings.
- Coupling the registry to the dashboard or to any specific consumer. The registry is a library-level API.

**Flexible:**

- Form of schema declaration: dataclass attribute, decorator, dict-typed class attribute. Implementation detail.
- Whether the registry is built lazily (on first import) or eagerly (at module load). Either works.
- Whether registry queries return pandas DataFrames or typed dataclasses. Default: DataFrame for browsability; typed underlying API for programmatic consumers.
- Search ranking algorithm. Substring match is sufficient for v1; fuzzy/relevance ranking deferred.

---

## Architecture Notes

The registry is the codebase's `INFORMATION_SCHEMA`. It exists to close the cultural loop: REQ_106 makes the architecture sound; REQ_107 makes it discoverable. They reinforce each other — neither alone prevents the re-derivation pattern.

This REQ is scoped separately from REQ_106 because the deliverables are independent. REQ_106 changes protocols, contracts, and access verbs. REQ_107 adds an inventory layer on top of those contracts. Either can land first, though both are needed for the cultural-architectural reinforcement to take.

The registry is also the natural surface for **publication discoverability**. A researcher landing on the published miscope library should be able to type three commands and understand what the platform contains:

```python
miscope.registry.analyzers()    # what computes things
miscope.registry.dataviews()    # what queryable views exist
miscope.registry.search("X")    # do we have something for X?
```

That is the first-five-minutes experience. Without it, a publisher is asking external researchers to read source code to figure out what's available.

### The schema as write-routing table (composition with REQ_110)

REQ_110 (Lakehouse Surface) specifies the columnar half of the storage engine: long-format Parquet, DuckDB query, publication bundles. This REQ supplies what REQ_110 routes *by*. The division of labor across one shared declaration surface — the analyzer's output schema:

- **REQ_107** declares *what each field is, its `kind`, and how it is keyed*.
- **REQ_110** persists the `columnar` fields as Parquet and makes them queryable.
- **The tensor catalog/resolver** ([`drafts/catalog_design/catalog.py`](../drafts/catalog_design/catalog.py)) indexes the `tensor` fields as descriptors and materializes only the selected ones.

A field's storage form, its query-ability, and its discoverability therefore all derive from a single authored statement, rather than being decided separately in three places.

This also closes a loop REQ_110 deliberately left open. REQ_110 keeps raw tensors as opaque `.npz` and holds DuckLake "in reserve" for catalog/registry needs. The coordinate-keyed schema here is the lighter, in-house answer to that reserved need: a `tensor`-kind field carries the *same* coordinate columns as a `columnar` field, so tensors become **joinable by descriptor** without a tensor query engine — you join over descriptors in SQL, then a resolver materializes only the selected blobs for linear algebra (which in turn produce more columnar/tensor fields, closing the cycle).

---

## Notes

- The user has wanted discoverability infrastructure for a while; deferred during research priority. With research now driving toward publication, the deferral cost flips and this becomes worth doing.
- Pairs naturally with REQ_103 (PyPI Publication Hardening) and REQ_106 (Analysis Layer Architecture). The registry is what publication exposes; REQ_106 is what the registry registers.
- The example query in the user's framing — "neurons that switch frequency over training, with the alternatives they considered" — should be discoverable in this manner: search for "frequency switch" → find the relevant DataView → use it. This is the acceptance experience for the registry's value.
- May extend in a follow-up to include "deprecated" entries (with replacement pointer) so the registry remains useful through analyzer rotation. Out of scope for v1 of REQ_107.
- **Catalog/resolver sketch (2026-06-04).** [`drafts/catalog_design/catalog.py`](../drafts/catalog_design/catalog.py) sketches the unified catalog seam: one queryable relation where `columnar` rows carry their value and `tensor` rows carry a `TensorRef` (URI + member + dtype + shape + codec — address only, no payload); exploration and joins happen in SQL over descriptors; a `TensorResolver` materializes only the selected tensor rows, batched per container (each archive opens once), and verifies declared shape/dtype against the bytes (the reproducibility guard). The artifact hardcodes `model/seed/dataset` as join keys — to be replaced by the family-owned key per the "coordinate keying" CoS above.
- **Scoping decision (resolved 2026-06-04, user).** The tensor-catalog + resolver **extends REQ_110** (added there as the *Tensor catalog + resolver* CoS section), to keep the whole storage engine in one requirement; REQ_110 is then chunked into child tasks. This REQ (107) carries only the per-field `kind` + coordinate declarations REQ_110 routes by.
- **REQ_128 hand-off (2026-05-28):** REQ_128 (Analyzer Input Provisioning) explicitly defers all *declared output-schema* work to this REQ. Its `deps` accessor validates requested `fields` against the upstream artifact's **runtime npz keys only** (a data-presence check) and leaves the manifest untouched — it makes no schema *declaration*. The "field names emerge implicitly from `analyze()` keys" gap (CoS line 34) remains REQ_107's to close, in code per Constraint line 61. When this REQ lands, REQ_128's accessor can additionally cross-check `fields` against the declared schema.
