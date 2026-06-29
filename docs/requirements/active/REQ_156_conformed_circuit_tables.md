# REQ_156: Conformed Layer 4 circuit object table (drain the generic fallback)

**Status:** Draft — not started.
**Priority:** Medium — Store-realization dial-tilt for the Layer 4 circuit objects
(REQ_152/154). Moves `circuit_spectra`'s columnar metrics from the generic
analyzer-named fallback into a **conformed semantic table** (the data model's Part IV
realization), establishing the circuit object identity so future per-circuit
attributes union into it.
**Branch:** `feature/REQ_156_conformed_circuit_tables`.
**Attribution:** Engineering Claude (under user direction).
**Data model anchor:** `data_model_master.md` Part IV (Object scalar column →
`SemanticClaim` reshaper → conformed semantic table) + Layer 8 guard ("a scalar
stable enough to name belongs as a column on an object table via a claim, never a
generic-store row").

---

## Problem Statement

`circuit_spectra` (REQ_152/154) emits three stable, named columnar metrics —
`copying_score`, `effective_rank`, `operator_norm` — per `(variant, epoch, site,
head)` for the five Layer 4 circuits. Today they land in the **generic
analyzer-named fallback** table (no semantic claim), which the data model calls "the
backlog to drain": stable named scalars belong on a conformed object table via a
claim, not in the generic store.

Conforming them now (a) registers the circuit object schema as designed intent,
(b) drains the generic fallback for this analyzer, and (c) makes the table a
**union target** so future per-circuit attributes (e.g. the data model's
OVCircuit `logit_attribution`, QKCircuit `operand_symmetry`) claim into the *same*
object table keyed by the circuit `site` discriminator — the cross-analyzer
conformance the generic store can't provide.

The circuits are realized as a **discriminator-keyed slice of one shared table**
(`site` = the circuit identity; `group_type = weight_matrix`), per the data model's
"the logical split into two objects does not mandate two physical tables."

## Conditions of Satisfaction

- [ ] **Conformed `circuit_spectra` semantic table + claim.** Add a `SemanticClaim`
  (mapping_semantic) that pulls `copying_score`/`effective_rank`/`operator_norm` into
  a designed table keyed `(variant_id, epoch, site, head)`, stamping
  `group_type = weight_matrix`. Register its `NATURAL_WIDE_INDEX`. With all three
  columnar fields claimed, the generic `circuit_spectra` fallback is no longer
  produced (the writer routes claimed → semantic, unclaimed → generic).
- [ ] **Parity (no regression).** The conformed table reproduces the prior generic
  table's content **value-identically** on a baseline (p113/s999/ds598): same rows,
  same `(site, head, epoch)` keys, same metric values, same `group_type`. The
  switch is a routing/realization change, not a data change.
- [ ] **Tensors unchanged.** `circuit_matrix` / `eigenvalues` stay tensor-catalog
  entries (untouched); only the columnar plane is conformed.
- [ ] **Consumers unaffected.** `miscope.query` table name stays `circuit_spectra`;
  the REQ_155 views/dashboard read it unchanged.
- [ ] **Remove the now-dead generic discriminator hint** for `circuit_spectra` in
  `mapping.py` (the conformed reshaper stamps `group_type` itself).
- [ ] **Tests** for the reshaper (3 metrics → conformed rows, group_type stamped,
  keyed by site/head) following the `test_publish` / mapping-semantic patterns.

## Constraints

- **Discriminator, not N tables (Part II/IV).** One conformed table, `site` as the
  circuit discriminator — do not mint five physical tables.
- **Storage internal to the API (invariant 3).** Pure warehouse-layer change behind
  the materializer; consumers see the same table name/columns.
- **No measurement change.** This is realization only — `circuit_spectra` the
  analyzer is untouched; values are identical (REQ_154 closed, no reopen).

## Notes

- **`dominant_frequency` is not a column here.** All five circuits are 2D/residual,
  so their Fourier dominant frequency is a `(k_a, k_b)` **pair tensor**
  (`weight_basis_projection`'s `dominant_frequency_pair` for `full_ov`/`full_qk`),
  not a scalar column — it lives in the tensor catalog, not this table. The data
  model's scalar `dominant_frequency` attribute on a circuit object is, concretely,
  that pair. A future 1-D circuit could add a columnar freq via a wbp claim into
  this same table.
- **Near-parity by construction.** Because the generic fallback already had the
  right shape, the conformed table is near-identical in content; the value is
  architectural (designed schema + union target + drained fallback), proven by the
  parity check. Keep the change minimal and the parity test strict.
- **Sets up the sibling attributes.** When OVCircuit `logit_attribution` /
  QKCircuit `operand_symmetry` (data model NEW fields) are built, they claim into
  this table — that's the payoff this REQ enables.
