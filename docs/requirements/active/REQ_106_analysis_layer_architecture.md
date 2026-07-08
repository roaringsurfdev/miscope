# REQ_106: Analysis Layer Architecture (Layering Principles + Wider Handoffs)

**Status:** Draft
**Priority:** High — logically precedes REQ_097–105 in the consolidation effort.
**Branch:** TBD
**Dependencies:** None blocking. Subordinates 097, 099, 100, 101, 102, 103, 104, 105 — they each implement aspects of these principles.
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

The consolidation effort (REQ_097–105) was originally framed as a set of independently-scoped cleanups: PCA primitives in 098, frequency primitives in 097, storage abstraction in 100, dataframe support in 101, and so on. Each REQ targets a specific surface. None names the architectural principle that ties them together.

That principle has emerged through discovery: **analysis is ETL, not measurement.** The codebase has been treating analyzers as one-off computations that produce arrays in dicts. They are not. They are the expensive transformation step that produces durable, queryable, first-class data products from raw model state. The expense is exactly what justifies materialization; the materialization is exactly what makes downstream research cheap.

When this principle is not honored, the symptoms are recognizable and frequently present in the current codebase:

1. **Measure functions silently fit their own derivations.** [`compute_circularity(centroids)`](../../src/miscope/analysis/library/geometry.py#L141) and [`compute_fourier_alignment(centroids, p)`](../../src/miscope/analysis/library/geometry.py#L177) each call `_pca_project_2d(centroids)` internally — same matrix, three SVDs per (epoch, site).

2. **Derivation functions reach to the data plane.** [`site_spectrum(variant, epoch, site)`](../../src/miscope/analysis/library/frequency.py#L164) loads from `variant.artifacts.load_epoch(...)` and computes the Fourier projection in one step. Convenient for notebooks; not unit-testable; not swappable when storage changes.

3. **Analyzers re-derive across files.** Frequency-group membership computation is inlined separately in [`freq_group_weight_geometry._build_group_labels`](../../src/miscope/analysis/analyzers/freq_group_weight_geometry.py#L179) and [`neuron_group_pca`](../../src/miscope/analysis/analyzers/neuron_group_pca.py), with subtly different membership rules.

4. **The artifact-dict handoff is too narrow.** `analyze()` and `compute_summary()` cannot share computed values because the boundary is `dict[str, np.ndarray]`. So summary re-fits PCA on the same matrix the analyzer already worked.

5. **No single answer to "where does this number come from?"** A given measure has multiple producers (analyzer-side, library-side, renderer-side) and external readers cannot determine which is canonical.

The user's framing: *"this refactor is its own kind of analysis stress test."* Surfacing what the codebase doesn't know about itself is the precondition for the platform exposing what it contains. Sharing research with other researchers requires the codebase be self-explanatory at the architectural level. That is the heft this REQ takes on.

---

## Conditions of Satisfaction

### Layering principle (testable)

The analysis surface is three layers, named and enforced:

1. **Data plane** — sources of raw bytes. `ActivationBundle` (or its successor under REQ_105's `ArchitectureAdapter`), `ArtifactSource` (LocalFS, HFHub), and accessors on `Variant` / `Family`. Information-hides storage layout.
2. **Derivation** — pure functions: `(structured_input, params) → typed_result`. Returns typed objects (`PCAResult`, `FrequencySpectrum`, `FrequencySet`, etc.). Deterministic, fakeable, cacheable.
3. **Measure** — pure functions: `(typed_result) → scalar`. Unit-testable with handcrafted inputs.

Acceptance criteria, each grep-able or schema-checkable:

- [ ] No measure function references `variant.artifacts`, `variant.bundle`, or any storage primitive. Test: grep for these names inside `library/geometry*.py` returns zero hits.
- [ ] No derivation function in pure form takes a `Variant`. The pure form takes structured arrays and parameters; convenience wrappers may take `Variant` and are 1–3 lines. Test: every public function in `library/frequency.py` and `library/pca.py` has a pure-input variant; grep `from miscope.families` in those modules returns zero hits in pure-form code.
- [ ] No measure or derivation function does file I/O. Test: grep for `open(`, `np.load`, `np.save`, `Path` reads in `library/*.py` (excluding `storage.py`) returns zero hits.
- [ ] Every measure function has at least one unit test calling it with `np.array(...)` literals, asserting a known-correct numeric result. No measure test depends on running a model.
- [ ] No analyzer's `analyze()` method calls `np.linalg.svd`, `np.linalg.eigh`, or `sklearn.decomposition.*` directly. SVD/eig live in the PCA primitive (REQ_098). Test: grep across `analysis/analyzers/`.
- [ ] No code outside `families/` and `analysis/library/storage.py` constructs filesystem paths into the artifacts tree. Test: grep for `epoch_{`, `os.path.join(...artifacts...`, `Path(...) / "artifacts"` outside the allowed modules returns zero hits.

### Result is first-class data (vocabulary + access)

- [ ] **Public access verb is `result`.** `Variant.result(name, epoch)`, `Variant.at(epoch).result(name)`, `Family.result(name)`. Selection rationale (see Architecture Notes): a computed analysis product that took GPU time to produce is data — it has the same dignity as a checkpoint.
- [ ] **`Variant.artifacts` (the `ArtifactLoader`) becomes internal.** Hidden from public docs; type hints expose it as private. No public consumer references it.
- [ ] **No `load_*` verb on the public access surface for analysis data.** The verb collides with model-checkpoint hydration. Reserved for that use only.
- [ ] **Access surface symmetric across Variant and Family.** If `Variant.result(...)` exists, `Family.result(...)` exists for family-scoped artifacts (cross-variant aggregates).
- [ ] **Storage shape is the analyzer's contract, not the access API's.** `Variant.result(name, epoch)` returns whatever shape the analyzer produces — typed object, structured dict, or `np.ndarray`. The access API does not dictate format.

### Per-epoch derivation context (wider analyze→summary handoff)

- [ ] The handoff between `analyze` and `compute_summary` carries computed structured objects, not just `dict[str, np.ndarray]`. Specifically: an analyzer that fits PCA in `analyze()` makes the `PCAResult` available to `compute_summary()` without re-fitting.
- [ ] Acceptance test: no analyzer's `compute_summary()` re-fits a PCA that `analyze()` already fit on the same matrix in the same epoch.
- [ ] Implementation choice (typed dataclass, return-shape change to `analyze()`, or per-epoch context object) deferred to implementation. Constraint: the wider handoff must not require artifact-format changes; the on-disk shape can remain `np.ndarray`-flattened, while in-memory the typed object survives.

### Generalized analyzer dependencies

- [ ] The three analyzer protocols (`Analyzer`, `SecondaryAnalyzer`, `CrossEpochAnalyzer`) collapse into a single `Analyzer` protocol with declared `requires` (artifacts and/or DataViews).
- [ ] The pipeline schedules analyzers by topological sort over declared dependencies. **No DAG engine** — a topological sort over a small dependency dict is sufficient at this scale.
- [ ] Existing analyzer kinds (primary / secondary / cross-epoch) become emergent from declared dependencies, not type distinctions. An analyzer with `requires=["model_bundle"]` is "primary" in the old sense; with `requires=["dominant_frequencies"]` is "secondary"; with `requires=["neuron_dynamics:cross_epoch"]` is "cross-epoch."
- [ ] An analyzer can declare it consumes another analyzer's output and the pipeline guarantees the upstream artifact exists before the downstream runs. Cross-analyzer re-derivation of artifact-shape data is structurally avoidable.

### DataView layer as first-class peer to artifacts

- [ ] DataViews are first-class peers to artifacts: versioned, importable, declared, and tested.
- [ ] Every DataView declares:
  - **Source dependencies** — which analyzer artifacts it consumes, by name and required field.
  - **Output schema** — column names and dtypes of the projection it produces.
  - **Version** — bumped when the projection logic changes.
- [ ] **View-on-artifact rule (v1):** DataViews read only from artifacts (and from `Variant`/`Family` metadata). DataViews do not read other DataViews. Reserved for v2 if a real composition need surfaces.
- [ ] DataView materializations are content-addressed: cache key is `(view_name, view_version, source_artifact_paths_and_versions)`. Because per-(variant, epoch) artifacts are immutable once written, cache invalidation is trivially correct.
- [ ] **Drift detection:** a CI test loads every DataView's declared dependencies and asserts the named fields exist in the upstream artifact. An analyzer that removes a field breaks downstream view tests at CI time, not silently in a notebook six months later.

### Cross-cutting impact on existing REQs

| REQ | Impact |
|---|---|
| 097 | Frequency primitives split into pure-input core + variant-coupled convenience wrappers. `site_spectrum(variant, epoch, site)` becomes a thin wrapper over `site_spectrum(matrix, basis, site)`. |
| 098 | Stable per user. Layering principle is consistent with what 098 already does — no scope change. |
| 099 | Existing partial articulation of the three-layer rule (renderer → loader → library → analyzer) is consistent but narrower. REQ_106 supersedes the principle statement; REQ_099 remains the migration mechanism. |
| 100 | `ArtifactSource` protocol moves into `core/sources.py` as a `Protocol` only. Implementations stay in `analysis/library/storage.py`. |
| 101 | DataFrame conventions remain as scoped. REQ_106 provides the "what is a DataView" contract that REQ_101 implements concretely. |
| 102 | Deprecation criteria gain "violates layering principle and cannot be cleanly migrated" as a deprecation reason. |
| 103 | Gate criteria add "REQ_106 acceptance criteria pass on grep tests" and "REQ_107 registry is browsable from a notebook." |
| 104 | Geometry consolidation lands under the layering rules. Functions in `geometry.py` take typed structured inputs (e.g., `PCAResult.projections`), not raw matrices the function then re-derives from. |
| 105 | `ArchitectureAdapter` is the data plane edge for live model state. The layering principle requires derivations consume bundle/adapter outputs (already pure tensors), not the bundle itself. |

---

## Constraints

**Must:**

- Every principle in this REQ has a testable acceptance criterion (grep, unit test, schema check, or CI test). No abstract-only rules.
- Public access verb across Variant and Family is `result`. No alternative names introduced in parallel.
- Generalized analyzer dependencies stay light — topological sort over a dependency dict, no DAG engine.
- DataView dependencies are declared and enforced at registry-load time.
- The view-on-artifact rule (v1) holds. Views do not read views.

**Must avoid:**

- Designing for hypothetical research that hasn't shown up. Principles encode what the project's analysis actually is, not a generic platform.
- Forcing every existing analyzer to migrate immediately. The principles apply to new work and to the in-flight refactor passes (097, 099–105). REQ_102 handles deprecation cleanups under these principles.
- Over-architecting the per-epoch derivation context. The simplest mechanism that lets `analyze` and `compute_summary` share computed values is sufficient.
- Introducing a `DataView` that reads other DataViews. Reserved for v2; explicitly out of scope here.

**Flexible:**

- Implementation form of the per-epoch derivation context (typed dataclass vs return-shape change vs context object).
- Whether DataView versioning is semantic (string `"v1.2"`) or content-addressed (hash of the projection function source). Either works; pick one in implementation.
- Whether `Family.result(...)` accepts a `variant_id` kwarg or whether family-scoped views are name-only. Implementation detail.
- Schema declaration form for analyzers and DataViews (decorator, dataclass, dict). Implementation detail.

---

## Architecture Notes

### The three layers

The principle the codebase commits to:

> **Analysis is ETL, not measurement.** Analyzers transform expensive raw model state into queryable data products. Measures consume structured data; they do not derive it.

| Layer | Returns | Pure? | Examples |
|---|---|---|---|
| Data plane | bytes / tensors | side-effecting (I/O) | `ActivationBundle`, `ArtifactSource`, `Variant.result(...)` |
| Derivation | typed structured object | yes | `compute_pca(matrix)` → `PCAResult`; `site_spectrum(matrix, basis)` → `FrequencySpectrum` |
| Measure | scalar | yes | `compute_circularity(projection, var_explained)` → float |

Analyzers compose layers; they do not blur them. A renderer that needs a number it doesn't have walks back the chain: ask the loader; if the loader can't shape it, ask library; if the library doesn't have the path, build the analyzer. Never compute inline.

### Why "result" rather than "derivation" or "artifact"

The user's framing (worth recording): *"In SQL, just because I had to run an operation to create data doesn't mean the resulting data is 'derived'. It's still data and can earn a first-class weight of its own."*

A computed analysis product that took an hour of GPU time is not a transient by-product. It is data with the same dignity as a checkpoint. "Derivation" carries the connotation of secondary; "artifact" is correct but technical and overloaded with the dict-of-arrays storage shape. "Result" matches what the thing actually is: the result of running a named analyzer at a named scope. The vocabulary supports the principle.

### Why DataViews are first-class peers (and why the v1 rule)

The codebase has chosen a hybrid lakehouse pattern — raw artifact blobs (npz, the durable layer) with codified DataViews (the queryable surface). This is a known pattern; it works in production data systems precisely because not all data wants the same shape and queries cannot all be predicted in advance.

The drift problem in SQL — materialized views going stale relative to mutating tables — is milder here, because per-(variant, epoch) artifacts are immutable once written. The drift that matters is **code drift**: a DataView's projection logic getting out of step with the upstream analyzer's output schema. Declared dependencies + version-keyed cache invalidation is the structural defense.

The view-on-artifact rule (v1) is self-discipline: views earn the right to read views in v2 only after a real composition need surfaces. Deferring view-on-view eliminates a class of cache-invalidation bugs and forces clarity about which projections deserve their own materialization.

### Why the per-epoch derivation context matters

Today, `analyze()` returns `dict[str, np.ndarray]` and that dict is the only memory-of-computation that crosses the boundary to `compute_summary()`. The dict is too narrow: it cannot carry a `PCAResult` (rich object, multiple fields, methods), only the flattened arrays. So `compute_summary()` re-fits PCA on the same matrix.

The fix is wider in-memory handoff while preserving on-disk format. Either an analyzer's `analyze()` returns a richer object (typed), and the pipeline serializes the array-valued fields to npz; or `compute_summary()` accepts a context that the pipeline assembles from `analyze()`'s output. Implementation detail — the constraint is no second SVD.

This is the same shape as the measure-takes-derivation-as-input fix: push the data dependency out of the consumer and into the caller's responsibility to materialize once.

### Why generalized analyzer dependencies (and why light)

Today, the protocol distinguishes three kinds and the registry maintains three pools. Cross-analyzer reuse goes through `SecondaryAnalyzer.depends_on` (one upstream artifact). Primary analyzers cannot declare dependencies and so they reach into `parameter_snapshot` directly.

A single `Analyzer` protocol with `requires: list[str]` collapses the distinction. The pipeline computes a topological sort and schedules accordingly. The user's instinct (research topology unlikely to be deeply nested) means a topological sort over a small dependency dict is sufficient. **No DAG engine.** Future research may produce disorganized analyzers as part of exploratory phases; that's expected, and the platform should not become unforgiving — refactorability is the contract, not enforcement.

### The cultural complement

Even with perfect architecture, analyzers will re-derive when their author doesn't know the upstream field exists. That is REQ_107's domain: a discoverability registry — the codebase's `INFORMATION_SCHEMA` — that is the canonical answer to "do we already have this?"

Architecture removes the friction; the registry removes the ignorance. Without both, even clean layering accumulates re-derivation through normal forgetting.

---

## Notes

- This REQ logically precedes REQ_097–105 in the consolidation effort. It is numbered higher only because it was authored later. The `consolidation_overview.md` sequencing table reflects the precedence.
- The user explicitly chose research over infrastructure during the project's first year and does not regret it. The principles in this REQ are the lessons learned from that choice. The REQ encodes what the codebase has shown should be true; it is not speculative architecture.
- The "this refactor is its own kind of analysis stress test" framing came from the user during discovery. The platform's value is exposing what's there. Surfacing what the codebase doesn't know about itself is the precondition for that exposure.
- These principles may be promoted to `/policies/architecture/` once stable, similar to how `/policies/debugging/` lives. For now, they live in this REQ. Promotion is a follow-up if the principles prove out.
- REQ_098 (PCA Strategy Cleanup) is treated as stable per user direction and is not modified by this REQ. The principles are consistent with REQ_098's existing scope.
- REQ_107 (Discoverability Registry) is the cultural complement: it makes the architecture browsable and prevents re-derivation through ignorance. Scoped separately because it has standalone value.
