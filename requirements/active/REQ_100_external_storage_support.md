# REQ_100: External Storage Support (ArtifactSource + Caching Decorator)

**Status:** Draft
**Priority:** High
**Branch:** TBD
**Dependencies:** REQ_106 (layering principle — `ArtifactSource` is a data-plane abstraction; the protocol moves to `core/sources.py` while implementations stay in `analysis/library/storage.py`). REQ_103 (PyPI hardening) consumes the HF source.
**Attribution:** Engineering Claude

---

## Problem Statement

Today, artifacts live exclusively on the local filesystem under `results/`.
The dashboard reads from there; notebooks read from there; analyzers write
there. There is no abstraction over storage location.

For external publication of research, the model is: notebooks pull pre-computed
artifacts from a HuggingFace repository, so a researcher who wants to read
the analysis doesn't need to re-run the pipeline. The same library code must
also continue to read locally for the dashboard / interactive analysis.

The library functions need a clean seam between *what to compute* and *where
to read/write the result*. This is the `ArtifactSource` abstraction. With it,
"stored is canon" becomes structurally enforceable via a `@cached_artifact`
decorator: if the artifact exists at the source, return it; otherwise compute,
write, return.

---

## Conditions of Satisfaction

### Library

- [ ] **`ArtifactSource` Protocol** lives in `miscope/core/sources.py`. `core/` keeps its no-I/O, no-heavy-deps invariant — sources is a Protocol declaration only, no implementation. The dashboard and renderers can import the protocol without pulling in `huggingface_hub` or `fsspec`.
- [ ] `miscope/analysis/library/storage.py` exposes the **implementations**:
  - `LocalFSSource(root)` — fully implemented, matches existing
    `results/{family}/{variant}/artifacts/{name}/` layout.
  - `HFHubSource(repo_id, *, revision=None)` — fully implemented, reads from
    a HuggingFace dataset/model repo. Write may be unsupported (researcher
    consumption is read-only).
  - `@cached_artifact(name, *, keys, source=None)` decorator.
- [ ] Source resolution from environment variable `MISCOPE_ARTIFACT_SOURCE`.
  Examples:
  - Unset → `LocalFSSource(variant.artifacts_dir)` (current behavior).
  - `local:/path/to/results/...` → `LocalFSSource('/path/to/results/...')`.
  - `hf:user/repo[@revision]` → `HFHubSource('user/repo', revision='...')`.
- [ ] `@cached_artifact` returns the cached result without recomputing if the
  artifact exists at the source. This is the "stored is canon" enforcement.
- [ ] Decorator-wrapped functions can be called with an explicit `source=`
  override (kwarg passes through), allowing notebooks to switch sources.

### Migration

- [ ] Library functions that return persistable results (e.g., `site_spectrum`)
  decorated with `@cached_artifact`. Initial set: `site_spectrum`,
  `learned_frequencies`, `expressed_frequencies`, plus PCA result-bearing
  functions where caching is sensible (per-matrix PCA on parameter snapshots).
- [ ] `ArtifactLoader` continues to work for the existing analyzer-produced
  artifacts. No forced migration of legacy paths in this REQ.

### Validation

- [ ] Round-trip test: call decorated function → artifact written → call
  again → result returned without recomputation (verify via instrumented
  compute counter or timing).
- [ ] HFHubSource read works against a published artifact (test repo TBD as
  part of REQ_103).
- [ ] Notebook + dashboard exercise both sources without code change to the
  consuming function (only the source resolution changes).

---

## Constraints

**Must:**
- ArtifactSource interface is small: `exists`, `read`, `write`. Anything
  more (listing, batch operations) deferred until needed.
- The default source preserves current behavior (LocalFSSource at
  `variant.artifacts_dir`). No existing call site changes behavior unless
  the user explicitly opts into a different source.
- `_format` placeholder key written to artifacts (already in discovery
  scaffolding) is a no-op until REQ_103 mints the v1.0 schema.

**Must avoid:**
- Coupling `@cached_artifact` to the family / variant abstractions in a way
  that prevents using it on functions that don't take a variant. (For now,
  variant-first is fine; broader use deferred.)
- Silent recomputation when a cached artifact exists. The decorator's job
  is to *not* compute when it doesn't have to.
- Implicit network calls. HFHubSource access must be explicit (env var or
  source kwarg), never the default.

**Flexible:**
- Caching layer: in-memory cache on top of the ArtifactSource for hot
  artifacts (e.g., `functools.lru_cache` keyed on `(source, name, keys)`)
  is acceptable but not required for v1.0.
- File naming convention beyond the existing `epoch_NNNNN.npz` pattern.
  Discovery scaffolding adds suffixes for site / extras; canonicalize during
  this REQ.

---

## Architecture Notes

**Stored is canon, full stop.** A function that recomputes when an artifact
exists is wrong. Reviewers reading the published code should see clearly
that the library reads stored values rather than re-deriving them.

**The decorator is the enforcement mechanism.** Discipline-only ("remember
to check the cache first") doesn't survive contact with new code. The
decorator makes the contract structural.

**Schema versioning is deferred to REQ_103.** The `_format` key is a marker
slot — no enforcement, just a hook. v1.0 mint adds versioning + validation.

**HF Hub interaction details (deferred to implementation):**
- Probably uses `huggingface_hub.hf_hub_download` for read.
- Authentication via `HF_TOKEN` env var.
- Local download cache managed by `huggingface_hub` (not miscope's concern).
- Write surface (researcher uploading new artifacts) deferred — primary
  use case is read-only consumption.

---

## Notes

- Discovery has `LocalFSSource` and the `@cached_artifact` decorator working;
  HFHubSource is stubbed.
- The path layout in `LocalFSSource._path()` (epoch + site + extras hash)
  needs review once we have real per-site artifacts to write — current
  shape is a guess.
- The "stored is canon" framing came from the user during discovery: scattered
  on-the-fly recomputation paths exist due to the fluid middle tier; this
  REQ closes that hole structurally.
