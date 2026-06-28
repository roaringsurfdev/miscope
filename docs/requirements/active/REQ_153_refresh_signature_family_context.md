# REQ_153: Refresh signature must capture family composition-site changes

**Status:** Draft — not started.
**Priority:** Medium — correctness gap in incremental refresh. Not blocking (a forced
rebuild is the workaround), but it makes the data model's CHEAP "repoint a universal
instrument at a new operand" pattern silently no-op, which will bite every Layer 4
sibling buildout (REQ_152 line).
**Branch:** TBD (`feature/REQ_153_refresh_signature_family_context`).
**Attribution:** Engineering Claude (under user direction).
**Relates to:** REQ_145 (signature-based incremental refresh, *staging*), REQ_137
(mass variant refresh), REQ_152 (FullOVCircuit — where this surfaced).

---

## Problem Statement

REQ_145 made the incremental refresh skip an analyzer when its **provenance
signature** is unchanged. That signature captures the analyzer's code/spec version
and its inputs — but **not the family context** the analyzer consumes at run time.

Several universal, site-driven analyzers (`weight_basis_projection`,
`activation_frequency_norm`, and now `full_ov_circuit`) read their *operands* from
family-declared sites (`weight_basis_projection_sites`,
`activation_frequency_norm_sites`, `circuit_spectra_sites`). Adding a site to a
family — the data model's CHEAP repoint (point an existing universal instrument at a
new composed operand) — changes what the analyzer *should* produce, but does **not**
change its signature. So the refresh considers the analyzer fresh and **skips it**:
the new site's outputs never materialize until someone forces a full rebuild of that
analyzer.

Concretely (found in REQ_152): adding the `full_ov` site to
`weight_basis_projection_sites` produced **no** `full_ov_*` Fourier artifacts on the
baselines until `weight_basis_projection` was force-rebuilt. The genuinely-*new*
`full_ov_circuit` analyzer ran fine (it had no prior artifacts to look fresh), so the
gap is specifically **repointed existing instruments**, which is the common case for
every future Layer 4 sibling (OV / QK / FullQK / DirectPath each add a site).

This silent no-op is dangerous: it reads as "covered" when it isn't.

## Conditions of Satisfaction

- [ ] **Family site config is part of the refresh signature.** A change to a family's
  declared composition sites consumed by an analyzer (additions, removals, or a
  composer change) invalidates that analyzer's freshness for that family/variant, so
  the next refresh re-runs it — no manual force needed. A site-set hash (site names +
  period axes + a composer identity/version) folded into the signature is the
  expected shape; do not stand up a parallel versioning surface (REQ_145 is the home).
- [ ] **Scoped invalidation.** Only analyzers that *consume* the changed site set are
  invalidated — adding `circuit_spectra_sites` must not force a rebuild of unrelated
  analyzers. The signature must attribute each site set to the analyzer(s) that read it.
- [ ] **Tensor catalog included.** The re-run must also refresh the tensor catalog for
  the affected analyzer (the tensor materialize pass is currently separate from the
  columnar one — confirm both are driven by the same freshness decision, or document
  the two-step contract).
- [ ] **Regression test.** A test that declares a family site, materializes, adds a
  second site, runs the *incremental* (non-forced) refresh, and asserts the new site's
  outputs appear — the exact scenario that silently failed in REQ_152.

## Constraints

- **Fold into REQ_145's machinery, not a parallel surface** (Part VI #4 of the data
  model: do not stand up a second versioning/staleness system).
- **Composer identity is the hard part.** A composer is a Python callable; a robust
  signature needs a stable identity (a declared version on the site, or a source hash)
  rather than object identity. Prefer an explicit, declared composer/site version over
  introspection magic — readable and AI-maintainable over clever.
- **Storage internal to the API** (invariant 3): the signature/refresh logic stays
  behind the warehouse/refresh boundary; consumers see only correct freshness.

## Notes

- Workaround until built (carry in REQ_137 mass refresh): force-rebuild the repointed
  instrument — `plan_analysis(v, [spec], force=True)` → `AnalysisPipeline(v).run(force=True, …)`
  → `materialize_variant_columnar(v, force=True)` + `materialize_variant_tensors(v)`.
- Surfaced in REQ_152 (see its Notes → "Implementation findings"); memory
  `finding-site-addition-not-in-refresh-signature`.
- Open question: should removing a site also *delete* its now-orphaned artifacts /
  catalog rows, or just stop reading them? Deletion is cleaner but riskier; a
  warehouse-health orphan scan (already contemplated in the data model Part VI #7) may
  be the safer surface for that half.
