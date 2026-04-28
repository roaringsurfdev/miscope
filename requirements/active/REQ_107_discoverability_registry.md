# REQ_107: Discoverability Registry (INFORMATION_SCHEMA for Analysis)

**Status:** Draft
**Priority:** Medium-high — cultural complement to REQ_106; not strictly blocking, but high value before publication.
**Branch:** TBD
**Dependencies:** REQ_106 (defines what gets registered: analyzers, DataViews, their schemas).
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

---

## Notes

- The user has wanted discoverability infrastructure for a while; deferred during research priority. With research now driving toward publication, the deferral cost flips and this becomes worth doing.
- Pairs naturally with REQ_103 (PyPI Publication Hardening) and REQ_106 (Analysis Layer Architecture). The registry is what publication exposes; REQ_106 is what the registry registers.
- The example query in the user's framing — "neurons that switch frequency over training, with the alternatives they considered" — should be discoverable in this manner: search for "frequency switch" → find the relevant DataView → use it. This is the acceptance experience for the registry's value.
- May extend in a follow-up to include "deprecated" entries (with replacement pointer) so the registry remains useful through analyzer rotation. Out of scope for v1 of REQ_107.
