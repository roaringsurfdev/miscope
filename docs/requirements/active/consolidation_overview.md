# Consolidation Overview — REQ_097 through REQ_110

**Status:** Discovery complete; ready to begin execution.
**Target version:** v1.0 (PyPI mint + first published Parquet bundle).
**Attribution:** Engineering Claude (under user direction).

> **Precedence note:** REQ_106 (Analysis Layer Architecture) was authored after REQ_097–105 but **logically precedes them all** — it names the layering principles that the other REQs each implement aspects of. REQ_107 (Discoverability Registry) is the cultural complement to REQ_106; both are subordinated by the implementation REQs. REQ_100 (HFHubSource) is deferred and not on the v1.0 critical path.

> **Consolidation pass (post-discovery):** After discovery surfaced two cohesive workstreams — *measurement primitives* and *tabular surface* — REQs 097, 098, 104 were superseded by **REQ_109 (Measurement Primitives Library)** and REQs 101, 108 were superseded by **REQ_110 (Lakehouse Surface)**. The superseded REQs remain in `active/` for archaeological reference (their problem framings carry forward); execution tracks under 109 and 110. See `consolidation_overview_feedback.md` for the user direction that drove the consolidation.

> **REQ_111 carve-out (post-primitive-design-pass):** During REQ_109 implementation, primitive design passes surfaced multiple latent bugs in pre-REQ_109 derivation paths (arctan vs arctan2, DC fictitious row, normalization mismatch in Fourier; float32 accumulation in clustering). In-place analyzer migration would have silently fixed or encoded those bugs without an audit trail. The "Analyzer migration" content originally inside REQ_109 was carved out into **REQ_111 (Parallel Analyzer Build-Out)**: build new analyzers in parallel, validate against existing on a reference variant set, and only then hand off to REQ_102 for retirement. This lets REQ_109 close cleanly once `procrustes_align` (phase 2d) lands.

---

## Why this is one cohesive effort

These REQs consolidate the analysis library in preparation for external
publication on PyPI + GitHub Pages (fieldnotes site) with derived Parquet
bundles attached as GitHub Releases. The driver is share-ability: scattered
implementations and overloaded vocabulary erode reviewer trust; ungrounded
findings without inspectable data erode reviewer trust further.

The user framed it directly:

> "We've been pretty disciplined so far, but the emphasis hasn't been on
> share-ability. ... if I were a researcher and I saw a bunch of scattered
> implementations of PCA, I'd likely stop review."

And later, on the publication frame specifically:

> "If I can provide trust-worthy access to the necessary data while also
> providing pointers to the github repo for this platform, I'm hopeful that
> another researcher will feel encouraged to review the work at whatever
> level feels appropriate without having to install and run an entire
> platform or download huge files."

Each REQ is independently scoped and individually reviewable. Together they
form the v1.0 milestone gate.

---

## The REQs

| REQ | Name | Rough phase | Depends on |
|-----|------|-------------|-----------|
| REQ_106 | Analysis Layer Architecture (precedent) | **Precedent** | — |
| REQ_109 | Measurement Primitives Library *(consolidates 097, 098, 104)* | Foundation | REQ_106 |
| REQ_111 | Parallel Analyzer Build-Out *(carved out of REQ_109's integration phase)* | Integration | REQ_106, REQ_109 |
| REQ_099 | Visualizations Plot-Only | Migration | REQ_106, REQ_109, REQ_111 |
| REQ_100 | External Storage Support (ArtifactSource) | **Deferred** | REQ_106 — not on v1.0 critical path; HF Hub publication reframed to GitHub Pages + Releases (now in REQ_110). Re-activates if/when raw-artifact publication demand surfaces. |
| REQ_110 | Lakehouse Surface (Parquet + DuckDB + Releases + DuckDB-WASM) *(consolidates 101, 108)* | Surface / Gate | REQ_106, REQ_107, REQ_109, REQ_111, REQ_103 |
| REQ_102 | Analyzer Deprecation | Cleanup | REQ_106, REQ_109, REQ_111 (parity validation gates retirement) |
| REQ_103 | PyPI Publication Hardening (package + repo + docs surfaces) | Gate | All others substantially in place; REQ_110 for the data-publication half |
| REQ_107 | Discoverability Registry | Cultural complement | REQ_106 |

### Superseded (kept for archaeology)

| REQ | Superseded by | Why |
|-----|---------------|-----|
| REQ_097 | REQ_109 | Fourier primitives folded into the measurement primitives library |
| REQ_098 | REQ_109 | PCA primitives folded into the measurement primitives library |
| REQ_104 | REQ_109 | Geometry / shape characterization folded into the measurement primitives library |
| REQ_101 | REQ_110 | Long-format DataFrame convention folded into the lakehouse surface |
| REQ_108 | REQ_110 | Parquet + DuckDB + Releases publication folded into the lakehouse surface |

REQ_105 (Architecture Adapter) is closely related but tracked separately — it addresses TransformerLens coupling rather than analysis-layer cleanup. It depends on REQ_106 for the data-plane edge framing.

---

## Suggested sequencing

The user's framing post-consolidation: *clean up what we have first, then add the new output surface.*

1. **REQ_106 first** — establishes the layering principles, the `result` vocabulary, the per-epoch derivation context, the generalized analyzer dependencies, and the DataView first-class status. The other REQs implement aspects of these principles.
2. **REQ_109 (Measurement Primitives Library)** — the foundation cleanup. Six categories (PCA, Fourier, clustering, shape characterization, shape comparison, velocity) extracted into pure-input tensor-friendly primitives. Phasing built in: PCA + clustering + velocity first (already substantially in discovery), then Fourier, then shape characterization, then comparison. Honors REQ_106's layering rules throughout. **Closes cleanly after phase 2d (`procrustes_align`).**
3. **REQ_111 (Parallel Analyzer Build-Out)** — the integration phase. New analyzers built alongside existing ones, consuming only the REQ_109 primitive layer. Each new-vs-old pair validated on a reference variant set; outcomes recorded (matches / old-has-bug / new-has-bug / both-kept) before REQ_102 retires the old analyzer. Carved out of REQ_109's original "Analyzer migration" CoS section after the primitive design pass surfaced bugs in pre-REQ_109 paths that in-place migration would have silently encoded.
4. **REQ_099 + REQ_102** as REQ_111 produces parity validation outcomes. REQ_099 is the renderer-side cleanup against the new analyzer contracts. REQ_102 retires the old analyzers per recorded validation outcomes (no listing without a recorded outcome).
5. **REQ_107** as the cultural complement. Can land anytime after REQ_106 is in place; ideally before REQ_110 (the publication build script reads from the registry).
6. **REQ_110 (Lakehouse Surface)** once REQ_109 stabilizes the upstream measurement-result types and REQ_111's new analyzers populate them. Adds the long-format tabular schema, Parquet persistence (coexisting with `.npz` for extract-only analyzers), the DuckDB query engine, the GitHub Releases workflow for publication bundles, and DuckDB-WASM integration in fieldnotes. The data half of the v1.0 publication story.
7. **REQ_103** as the gate. Publication readiness review + v1.0 mint — REQ_106 grep tests passing, REQ_107 registry browsable, REQ_110 publication bundle workflow operational with at least one bundle queryable via DuckDB-WASM from a fieldnotes article.

**Asymmetry across the three workstreams.** REQ_109 is a primitive-extraction refactor with bounded blast radius. REQ_111 is a research-active *expand-then-collapse* phase: build new analyzers in parallel, let them and the old ones coexist on disk, validate, then collapse back via REQ_102. REQ_110 is an architecture extension — adding a new persisted output format, a new query layer, and a new publication workflow on top. Sequencing 109 → 111 → 110 lets each rest on the previous one's stabilization, with REQ_102 cleanup interleaved as parity validation outcomes accumulate.

---

## Discovery code (already in place on this branch)

These files implement the locked-in spec for testing. They are *additive
only* — no existing code is touched. Migration of analyzers / views happens
under the relevant REQs.

### Core layer (vocabulary)

```
src/miscope/core/
  __init__.py        # re-exports
  sites.py           # WeightSite, ActivationSite enums
  frequencies.py     # CommitmentMethod, THRESHOLDS, FrequencySpectrum, FrequencySet
  pca.py             # PCAResult dataclass
  groups.py          # WEIGHT_MATRIX_NAMES, COMPONENT_GROUPS, ATTENTION_MATRICES, ARCH_WEIGHT_NAMES
```

`miscope.core` contains only types, enums, and constants. No compute, no
I/O, no heavy dependencies. Importable from any layer (renderers, dashboard,
notebooks).

### Library layer (operations)

```
src/miscope/analysis/library/
  pca.py             # REPLACED: pca() primitive + 4 convenience builders + merge_columns
  frequency.py       # NEW: site_spectrum, learned_frequencies, weights_by_frequency, trajectory helpers
  storage.py         # NEW: ArtifactSource, LocalFSSource, HFHubSource (stub), @cached_artifact
```

### Implementation depth in discovery

| Component | Status |
|-----------|--------|
| `core/*` | Fully implemented |
| `library/pca.py::pca()` + builders | Fully implemented |
| `library/storage.py::LocalFSSource` + `@cached_artifact` | Fully implemented |
| `library/storage.py::HFHubSource` | Stubbed (REQ_100) |
| `library/frequency.py::site_spectrum` (WeightSite, period-axis sites) | Fully implemented |
| `library/frequency.py::site_spectrum` (composed weight sites: MLP, ATTN) | Raises with guidance (REQ_109) |
| `library/frequency.py::site_spectrum` (ActivationSite) | NotImplementedError (REQ_109) |
| `library/frequency.py::learned_frequencies` (SPECTRAL_THRESHOLD method) | Fully implemented |
| `library/frequency.py::learned_frequencies` (NEURON_DOMINANT, HEAD_DOMINANT) | NotImplementedError (REQ_109) |
| `library/frequency.py::weights_by_frequency` | NotImplementedError (REQ_109) |
| `library/frequency.py::transient_frequencies` (set algebra) | Fully implemented (works once underlying primitive does) |

---

## Known intentional duplications during discovery

1. **`COMPONENT_GROUPS`, `WEIGHT_MATRIX_NAMES`, etc.** are defined in BOTH
   `miscope/core/groups.py` (canonical) and
   `miscope/analysis/library/weights.py` (legacy). They MUST stay in sync.
   REQ_109 / REQ_102 collapses the duplication by making `weights.py` import
   from `core/groups.py`.

2. **PCA primitives** exist in both `analysis/library/pca.py` (canonical
   from this discovery) and the legacy paths in `trajectory.py` and
   `geometry.py`. REQ_109 retires the legacy paths.

3. **Frequency analysis** — the `dominant_frequencies` analyzer continues
   to write its artifact alongside any new analyzer that REQ_109 introduces.
   Old artifacts remain readable for the deprecation window per REQ_102.

---

## How to test the discovery code

```python
from miscope import load_family
from miscope.core import WeightSite, FrequencySpectrum, FrequencySet, PCAResult
from miscope.analysis.library.pca import (
    pca, pca_per_matrix, pca_flattened_snapshots, pca_centroids_global,
)
from miscope.analysis.library.frequency import (
    site_spectrum, learned_frequencies, spectrum_trajectory, transient_frequencies,
)
from miscope.analysis.library.storage import LocalFSSource, cached_artifact

family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=109, seed=485, data_seed=598)

# PCA primitive on a single weight matrix
snapshot = variant.artifacts.load_epoch("parameter_snapshot", 5000)
result: PCAResult = pca_per_matrix(snapshot, "W_E")
print(result.participation_ratio, result.explained_variance_ratio)

# Frequency primitive on the embedding (the original lissajous bug site)
spec: FrequencySpectrum = site_spectrum(variant, 5000, WeightSite.EMBEDDING)
print(spec.frequencies, spec.magnitudes, spec.derivation)

# Learned frequencies via spectral threshold
fs: FrequencySet = learned_frequencies(variant, 5000, WeightSite.EMBEDDING, threshold="canonical")
print(fs.frequencies, fs.method, fs.threshold)

# Storage: roundtrip an artifact via the source
source = LocalFSSource(str(variant.artifacts_dir))
print(source.exists("parameter_snapshot", {"epoch": 5000}))  # True
```

The lissajous-fix test case (`expressed_frequencies(variant, epoch,
ActivationSite.RESIDUAL_POST)`) intentionally raises `NotImplementedError`
in discovery — it requires both the activation-site Fourier *primitive*
(REQ_109) and the *analyzer* that consumes it (REQ_111). The type signature
being callable proves the API shape is right; the primitive body lands
under REQ_109, and the analyzer that wires it up to a real call site lands
under REQ_111.

---

## What's NOT in discovery (and which REQ owns it)

| What | Owning REQ |
|------|------------|
| Layering principles + grep-test acceptance criteria | REQ_106 |
| `Variant.result(name, epoch)` / `Variant.at(epoch).result(name)` / `Family.result(name)` access verb | REQ_106 |
| Per-epoch derivation context (wider analyze→summary handoff) | REQ_106 |
| Generalized analyzer dependencies (single Analyzer protocol, topological scheduling) | REQ_106 |
| DataView first-class status (versioning, declared dependencies, drift detection) | REQ_106 |
| Discoverability registry (`miscope.registry.*`) | REQ_107 |
| Composed weight site spectrum (W_E @ W_in for MLP, etc.) | REQ_109 |
| Pure-input forms of all measurement primitives (separate from variant-coupled wrappers) | REQ_109 |
| Activation-site Fourier *primitive* (pure-input projection on activations) | REQ_109 |
| Activation-site Fourier *analyzer + artifact* (variant-coupled, written to disk) | REQ_111 |
| `expressed_frequencies(variant, epoch, ActivationSite.*)` body — depends on the analyzer | REQ_111 |
| NEURON_DOMINANT / HEAD_DOMINANT commitment methods | REQ_109 |
| `weights_by_frequency` body | REQ_109 |
| Migration of existing analyzers to single PCA primitive (now via parallel construction + parity validation) | REQ_111 |
| Geometry function consolidation under shape-characterization category | REQ_109 |
| Shape characterization functions take typed inputs (`PCAResult.projections`), not raw matrices | REQ_109 |
| Lissajous, sigmoidality, saddle curvature, jerk extracted to library primitives | REQ_109 |
| Procrustes alignment extracted to library primitive | REQ_109 |
| Velocity / acceleration primitives (replacing inline `np.diff`) | REQ_109 |
| `ArtifactSource` Protocol move to `core/sources.py` (implementations stay in `library/storage.py`) | REQ_100 *(deferred)* |
| HuggingFace Hub source implementation | REQ_100 *(deferred)* |
| Schema versioning enforcement (`_format` validation) | REQ_103 |
| Long-format tabular schema with `group_type` / `operation_type` discriminators | REQ_110 |
| Long-format DataView population in `weight_pca.py`, `representation_pca.py` | REQ_110 |
| Parquet as persisted form for DataView outputs (coexisting with `.npz` for extract-only analyzers) | REQ_110 |
| DuckDB as cross-variant query engine | REQ_110 |
| Internal warehouse vs. published bundle distinction | REQ_110 |
| GitHub Releases workflow for publication bundles | REQ_110 |
| DuckDB-WASM integration in fieldnotes for inline queries | REQ_110 |
| `research/` directory restructure (research workbench separation) | Handled directly (tidying, not REQ-tracked); REQ_103 validates it |
| Fieldnotes "Platform" navigation section | REQ_103 |
| GitHub Action for PyPI publication on tag | REQ_103 |
| Migration of existing renderers to plot-only | REQ_099 |
| Retirement of `coarseness`, `fourier_nucleation`, `centroid_dmd`, `dominant_frequencies`, `effective_dimensionality` | REQ_102 |
| Layering audit of existing analyzers (migrate-vs-retire classification) | REQ_102 |
| `pyproject.toml` extras, public API audit, canonical templates | REQ_103 |

---

## Notes for future Claude

- **REQ_106 was authored after REQ_097–105 but logically precedes them.** The user's framing: research had to be prioritized over infrastructure during the project's first year, and REQ_106 encodes the lessons learned from that choice. The principles are not speculative — they encode what the codebase has shown should be true. Don't treat REQ_106 as theoretical; treat it as the precedent the other REQs were independently moving toward.
- **REQ_107 (Discoverability Registry) is the cultural complement to REQ_106.** Architecture removes friction; the registry removes ignorance. Without both, even clean layering accumulates re-derivation through normal forgetting.
- **REQ_109 consolidates 097/098/104.** The post-discovery user audit framed this as "if I were a researcher and saw scattered implementations of PCA, I'd likely stop review." Six measurement-primitive categories (PCA, Fourier, clustering, shape characterization, shape comparison, velocity) with pure-input tensor-friendly contracts. The `fit` → `characterize` rename came from the same audit — these are descriptors, not parameter learners. Existing scattered code locations (e.g., `repr_geometry.py`, `manifold_geometry.py`, `sketch_lissajous_fit.py`, `parameter_trajectory_pca.ipynb`) are catalogued in REQ_111 as the parallel-build manifest (originally written into REQ_109; carved out post-design-pass).
- **REQ_111 carves analyzer integration out of REQ_109.** During REQ_109 phase 2c, design passes for Fourier and clustering primitives surfaced multiple latent bugs in pre-REQ_109 code (arctan vs arctan2; DC fictitious row; normalization mismatch; float32 accumulation regression in `compute_class_centroids`). Each was caught **before** in-place migration would have either silently fixed it (no audit trail) or silently encoded it forward. The analyzer-migration content originally in REQ_109's CoS was reframed: build new analyzers in parallel, validate against existing on a reference variant set, record the answer to *"did the old code do the right thing?"*, then hand off to REQ_102 for retirement. The user's framing: *"the platform might need to expand before collapsing back again."*
- **REQ_110 consolidates 101/108.** The lakehouse framing surfaced post-discovery: lean into Parquet + DuckDB for the tabular layer, with inline browser queries via DuckDB-WASM in fieldnotes. The `group_type` / `operation_type` discriminator-column design is the schema-side load-bearing decision. Coexistence with `.npz` is structural — extract-only analyzers (`parameter_snapshot`, activation captures) stay `.npz`; transform-emitting analyzers go Parquet. **DuckLake is held in reserve, not committed** — if registry/catalog work in REQ_107 starts to look like reinvention of DuckLake's surface, revisit.
- **Publication reframed away from HF Hub** (during the discussion that produced REQ_108, now folded into REQ_110). Two motivations drove the reframe: (1) reviewer-trust workflow — query the data underlying a finding directly from a fieldnotes article, no install; (2) **moat for platform evolution** — published Parquet bundles are immutable and self-contained, so internal storage can continue to evolve without breaking citations. REQ_100 (HFHubSource) is deferred until raw-artifact publication demand surfaces.
- **REQ_103 covers the package + repo + docs surfaces; REQ_110 covers the data surface.** They're a matched pair for v1.0; neither alone constitutes the publication story. REQ_103 also carries the first-impression / repo-presentation work (root README polish, `research/` directory restructure validation, fieldnotes Platform section). The user explicitly named first-impression cost as significant for an independent researcher without existing social capital.
- **Monorepo through v1.0.** The split decision (Manim / Manim Community Edition pattern was the precedent raised) was discussed and deferred. Neither PyPI publication nor Parquet publication actually forces multi-repo; the Manim split happened *after* a community formed organically. Triggering conditions for a future split are documented in REQ_103.
- This branch is `refactor-dataview`, treated as exploratory. The discovery
  code may be discarded in favor of a cleaner v1 branch once the spec is
  validated. The REQs are the durable artifact.
- The user's explicit framing: "If we go in a bad direction, we can capture
  that in the requirements for a more solid and committed direction in a
  new branch." — so iteration on this branch should feed back into the REQs.
- Phase-change threshold: the user observed that ~60–70% threshold settings
  on Multi-Stream views surface a regime change ("something starts to happen
  dynamically"). They linked it conceptually to CR3BP atoll-shaped no-travel
  zones and weight-decay-driven release into saddle/tube transit. The
  THRESHOLDS registry is intentionally open for a future `'phase_change'`
  entry once the data is clean enough to characterize the threshold value.
  See REQ_097 Notes (preserved for archaeology) and REQ_109's Fourier section.
- The lissajous fit bug is the test case for the whole consolidation. If
  the new `expressed_frequencies(variant, epoch, ActivationSite.RESIDUAL_POST)`
  + `characterize_lissajous(trajectory_2d)` wired up under REQ_109 fix the
  snags reported by the user (CR3BP linearization), the architecture has
  paid for itself.
