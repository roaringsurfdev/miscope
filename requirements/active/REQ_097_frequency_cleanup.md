# REQ_097: Frequency Cleanup (Site-Coupled Frequency Analysis)

**Status:** Superseded by REQ_109 (Measurement Primitives Library)
**Priority:** High
**Branch:** TBD (currently incubating on `refactor-dataview` for discovery)
**Dependencies:** REQ_106 (layering principle — frequency primitives must have a pure-input form distinct from variant-coupled convenience wrappers); REQ_098 (PCA strategy) is parallel; REQ_100 (storage) provides the caching seam.
**Attribution:** Engineering Claude

> **Superseded by REQ_109 (Measurement Primitives Library).** This REQ's scope (Fourier primitives, site-coupled frequency analysis, the `frequency_spectrum_per_site` analyzer, the lissajous-fix test case) was consolidated with REQ_098 (PCA) and REQ_104 (Geometry) into a single primitives-library REQ. Content preserved here for archaeology — the framing of "three meanings of learned frequencies," the site-coupling type rules, and the THRESHOLDS registry design all carry forward unchanged into REQ_109's Fourier section. Implementation tracks under REQ_109.

---

## Problem Statement

Three distinct meanings of "learned frequencies" coexist in the codebase, with no
canonical definition for any of them:

1. `dominant_frequencies` analyzer outputs continuous Fourier norms on `W_E` only —
   single-site, embedding-only, despite the generic name.
2. `fourier_frequency_quality` artifact contains a field literally named
   `dominant_frequencies` holding integer indices of dominant frequencies — a
   discrete set, different concept.
3. `neuron_dynamics.dominant_freq` is per-neuron dominant frequency, used by
   `variant_analysis_summary._get_learned_frequencies()` (MLP-only, threshold-derived).

The lissajous fit bug is a direct consequence: the fit operates on residual-stream
geometry but checks against `W_E` spectral content. The right comparison is against
the residual stream's own expressed frequencies. The bug class is *unbug-writeable*
once site coupling is enforced at the type level.

The deeper failure is that the codebase has no single answer to:
> "Which frequencies has this variant learned at this site at this epoch?"

The lissajous use case (linearization à la Shane Ross / CR3BP) needs a clean,
trustworthy answer to that question across embedding, attention, MLP, and residual
sites.

---

## Conditions of Satisfaction

### Library API (additive — discovery is in place)

- [ ] `miscope/core/sites.py` exposes `WeightSite` and `ActivationSite` enums.
- [ ] `miscope/core/frequencies.py` exposes `FrequencySpectrum`, `FrequencySet`,
  `CommitmentMethod`, and `THRESHOLDS` registry.
- [ ] `miscope/analysis/library/frequency.py` exposes:
  - `site_spectrum(variant, epoch, site) -> FrequencySpectrum` — fully implemented
    for all `WeightSite` values (including composed sites like `MLP_INPUT`,
    `MLP_OUTPUT`, and `ATTN_*`). Discovery handles only direct period-axis sites.
  - `learned_frequencies(variant, epoch, site, *, threshold, method) -> FrequencySet`
    — fully implemented for all three `CommitmentMethod` values.
  - `weights_by_frequency(variant, matrix, frequency, epoch) -> ndarray` — fully
    implemented for MLP and Attention; embedding deferred.
  - `expressed_frequencies(variant, epoch, activation_site, ...) -> FrequencySet`
    — new analyzer required (see Analyzer below). Wires through to the new
    artifact when present.
- [ ] `transient_frequencies` derives via set algebra, not as a bespoke analyzer.

### Layering compliance (REQ_106)

- [ ] Each frequency primitive has a pure-input form: `site_spectrum(matrix, basis, site)`, `learned_frequencies_from_spectrum(spectrum, threshold, method)`. The variant-coupled forms — `site_spectrum(variant, epoch, site)` etc. — are 1–3 line wrappers that load the matrix and call the pure form.
- [ ] Acceptance test: pure-form functions are unit-testable with `np.array(...)` literals. No frequency primitive in pure form imports from `miscope.families`. Convenience wrappers may import `Variant` and are clearly marked.
- [ ] Storage decoration (`@cached_artifact`) applies to the variant-coupled wrapper, not to the pure form. Caching is data-plane concern, not derivation concern.

### New analyzer

- [ ] New analyzer `frequency_spectrum_per_site` (name TBD, must avoid the
  overloaded `dominant_frequencies` name):
  - Per-epoch artifact.
  - For each `WeightSite` with a meaningful spectrum (composed where needed),
    stores a (n_freqs,) magnitude array.
  - For each `ActivationSite` (mlp_out, attn_out, resid_post, resid_pre, embed,
    logits), stores a (n_freqs,) magnitude array derived from running the
    canonical analysis probe and projecting the activations onto the Fourier basis.
  - Stores derivation provenance string per site (matches `FrequencySpectrum.derivation`).

### Migration of consumers (additive transition)

- [ ] `variant_analysis_summary._get_learned_frequencies()` migrates to call
  `learned_frequencies(variant, epoch, WeightSite.MLP_INPUT,
  method=CommitmentMethod.NEURON_DOMINANT)`. Numeric output must match within tolerance.
- [ ] `notebooks/sketch_lissajous_fit.py` updated to use
  `expressed_frequencies(variant, epoch, ActivationSite.RESIDUAL_POST)` instead of
  loading `dominant_frequencies` (which was always W_E).
- [ ] Existing dashboard view `activations.mlp.dominant_frequencies_over_time`
  registered against the new analyzer. Visual output unchanged within tolerance.
- [ ] `transient_frequency` analyzer rewired to consume `frequency_trajectory(...)`
  rather than its own ad-hoc derivation, OR documented as a separate
  interpretation layer with explicit dependency on the canonical primitive.

### Threshold registry

- [ ] `THRESHOLDS` is the single source of truth: `'canonical': 0.10`,
  `'transient': 0.05`. All other locations that hard-code these values cite
  the registry.
- [ ] Registry is intentionally open. Adding a new named threshold (e.g.,
  `'phase_change'`) requires a documented phenomenon — not a dumping ground.

---

## Constraints

**Must:**
- Preserve site coupling at the type level. No primitive accepts a site-less
  "what frequencies does this variant have?" query.
- Maintain numerical equivalence with the existing `_get_learned_frequencies()`
  path during migration (verified by characterization tests on at least 3
  reference variants: p109/s485/ds598, p113/s999/ds598, p101/s999/ds598).
- New analyzer artifacts do not overwrite or rename `dominant_frequencies`
  artifacts — additive only. Old artifacts remain readable until REQ_102
  (Analyzer Deprecation) retires them.

**Must avoid:**
- Reintroducing site-less frequency queries.
- Coupling the library functions to a specific family. Variants supply prime
  via `variant.prime`; non-modular families either raise informatively or skip.
- Hidden defaults. `CommitmentMethod` must be explicit on the FrequencySet.

**Flexible:**
- The exact name of the new analyzer (`frequency_spectrum`, `site_frequency`,
  `per_site_fourier` — not `dominant_frequencies`).
- Whether `expressed_frequencies` and `learned_frequencies` are separate functions
  or unified with type dispatch (current discovery has them planned as separate;
  decision deferred to implementation).
- Storage decoration (whether `site_spectrum` etc. use `@cached_artifact` directly
  or wrap a manual cache check). Default toward decoration once REQ_100 lands.

---

## Architecture Notes

**The site distinction is load-bearing.** WeightSite captures *where frequencies
are learned*; ActivationSite captures *where they are expressed*. They are paired
but not 1:1 — `ATTN_OUT` is downstream of `W_Q/K/V/O`. The verbs differ:

```python
learned_frequencies(variant, epoch, WeightSite.MLP_INPUT)        # learned
expressed_frequencies(variant, epoch, ActivationSite.RESIDUAL_POST)  # expressed
```

**Composed weight sites need composition.** `MLP_INPUT`'s spectrum requires
projecting `W_E @ W_in` (or equivalent), since `W_in` itself has no period axis.
The composed primitives are this REQ's scope.

**`weights_by_frequency` membership method.** For `MLP_*`, neuron-level
attribution from the new analyzer (or migrated `neuron_dynamics`) provides the
mapping. For `ATTN_*`, head-level attribution from a migrated `attention_freq`.
For embedding sites, the question is currently open — defer or restrict.

**Phase-change threshold.** During discovery work on Multi-Stream views, a
~60–70% threshold appears to mark a dynamic regime change (possibly weight-decay
crossing a release point that opens saddle/tube transit). The threshold is not
yet a named entry — needs cleaner data first. The registry is explicitly open
for future named entries; do not add `'phase_change'` until it's earned.

---

## Notes

- This REQ is the highest-leverage of the consolidation set. Everything else
  (PCA cleanup, dataframe support, publication) leans on having a single
  trustworthy frequency primitive.
- The lissajous fit fix is the proof-of-concept test. Once
  `expressed_frequencies(variant, epoch, ActivationSite.RESIDUAL_POST)` works,
  re-running the lissajous analysis with the right site comparison should
  resolve the snags reported by the user (CR3BP linearization).
- During discovery, `frequency.py` returns informative `NotImplementedError` for
  every code path that needs the new analyzer — those are the closure points
  for this REQ.
