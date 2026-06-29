# REQ_154: Generalize to a `circuit_spectra` instrument; add FullQKCircuit

**Status:** Implemented on `feature/REQ_154_circuit_spectra_siblings` (awaiting merge).
**Priority:** Medium — proves the Layer 4 buildout pattern (one universal spectral
instrument, circuits as sites) and adds the second circuit object so QK binding can
be compared head-for-head against the OV copy/transform split.
**Branch:** `feature/REQ_154_circuit_spectra_siblings`.
**Attribution:** Engineering Claude (under user direction).
**Data model anchor:** `data_model_master.md` Layer 4 — `FullQKCircuit`
(`STABLE · empty · PLANNED`); follows REQ_152 (`FullOVCircuit`).

---

## Problem Statement

REQ_152 built `full_ov_circuit` as a fresh analyzer that was *already generic* over
family-declared circuit composition sites — the spectral measurement (SVD invariants
+ eigenspectrum copying score) is universal; a circuit is just an operand. Naming it
for its first operand (`full_ov_circuit`) obscured that. As the data model's Layer 4
buildout continues (FullQKCircuit, then OV / QK / DirectPath), each sibling is the
*same* measurement on a *different* composed operand.

Per the "Views are universal instruments" invariant and the data model's allowance
for "a logical object as a discriminator-keyed slice of a shared table," the right
shape is **one universal `circuit_spectra` instrument** with each circuit a `site`
row — not one analyzer per circuit. This requirement generalizes the instrument and
adds **FullQKCircuit** as the first sibling, which the family already has the
composition for: the existing `attn_qk` composer builds exactly
`W_E^T W_Q^T W_K W_E` (the square token-pair QK form).

## Conditions of Satisfaction

- [ ] **Rename `full_ov_circuit` → `circuit_spectra`** (analyzer, SPEC name, class
  `CircuitSpectraAnalyzer`, registration, decompose rule, generic group_type,
  family.json). Pre-v1.0.0, no back-compat shims; regenerate the affected artifacts
  under the new name.
- [ ] **Add a `full_qk` circuit site** to the family's `circuit_spectra_sites`,
  reusing the existing `_compose_attn_qk` composer (single composition source). The
  `circuit_spectra` table now carries `full_ov` and `full_qk` rows, keyed by `site`.
- [ ] **FullQKCircuit spectral invariants land**: `effective_rank`, `operator_norm`
  (the data model's FullQKCircuit attributes) + the composed matrix / eigenvalues
  tensors, materialized and queryable, keyed `(variant, epoch, site='full_qk', head)`.
  `copying_score` is computed uniformly but is OV-meaningful only — documented as
  not-interpreted for QK.
- [ ] **Validation on a baseline**: `full_qk` `operator_norm` / `effective_rank`
  match an independent SVD of the `(W_E W_Q)(W_E W_K)^T` composition; the `full_ov`
  numbers are unchanged by the rename (value-identical to REQ_152).
- [ ] **Tests** cover both composers (vs independent references) and both sites'
  analyzer output shapes/keys.

## Constraints

- **Universal instrument (invariant 1).** `circuit_spectra` measures any composed
  square circuit; circuits are family-declared operands (sites), never per-object
  analyzers. Adding a sibling = one site entry + (if needed) one composer.
- **Reuse composers / primitives (not analyzers).** `full_qk` reuses `_compose_attn_qk`;
  the spectral math stays in the `compute_svd` / participation-ratio primitives.
- **Storage internal to the API (invariant 3).** Same declared-schema → warehouse
  path as REQ_152; no path literals.
- **Small pass.** FullQKCircuit only; OVCircuit (`W_O W_V`), QKCircuit
  (`W_Q^T W_K`), and DirectPath (`W_U W_E`) remain one-line site additions for a
  later pass.

## Notes

- **`full_qk` ≡ the existing `attn_qk` matrix.** `_compose_attn_qk` returns
  `(W_E W_Q)(W_E W_K)^T` = `W_E W_Q W_K^T W_E^T`, the same square `(p, p)` token-pair
  form the data model names `W_E^T W_Q^T W_K W_E` (orientation differs by transpose;
  spectral invariants are transpose-invariant). So FullQKCircuit's
  `dominant_frequency` is already materialized by `weight_basis_projection`'s
  `attn_qk` site — the conformed FullQKCircuit claim joins `circuit_spectra[site=full_qk]`
  (spectra) with `weight_basis_projection[site=attn_qk]` (frequency). Worth folding a
  site-name alias note into that claim when written.
- **Research hook.** With both circuits instrumented, the open question from the
  REQ_152 finding (`finding-ov-copy-transform-head-split`) becomes testable: does QK
  *binding* structure (effective_rank / dominant frequency of `full_qk`) co-emerge on
  the same heads that specialize on the OV copy/transform axis? Natural next probe
  before the fieldnotes capture.
- **Depends on REQ_153** (refresh-signature gap) for the `full_qk` site to
  auto-materialize on future variants without a forced rebuild; until then the
  REQ_152 force-rebuild workaround applies.
