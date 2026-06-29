# REQ_158: Architecture as a first-class object (the model-side type)

**Status:** Draft (problem-first; for review). Entry slice of the Layer 1
context-provider cluster — do before REQ_159 (TaskType/Task).
**Priority:** High — the data model marks Architecture "✅ commit now — many-to-many
already real" (Part V). Separating the structural factor is the smaller, FK-stable
half of the Family decomposition and unblocks the TaskType/Task slice.
**Branch:** TBD (`feature/REQ_158_architecture_object`).
**Attribution:** Engineering Claude (under user direction).
**Depends on:** none. **Blocks:** REQ_159 (TaskType/Task — the semantic half).
**Data model:** `data_model_master.md` Part II (Architecture · `STABLE`), Part V scan
row "Architecture … ✅ commit now".

---

## Problem Statement

The data model splits a **Family** into two independent factors — a model-side
**Architecture** (`STABLE`, the task-independent transformer spec) and a task-side
**TaskType** (`STABLE`, the semantic-context provider). Today neither is a
first-class object: the structural spec is **embedded** as an `architecture` block
inside each `data/{family}/family.json`:

```json
"architecture": { "n_layers": 1, "n_heads": 4, "d_model": 128, "d_head": 32,
                  "d_mlp": 512, "act_fn": "relu", "n_ctx": 3 }
```

Consequences:

- **No identity, no sharing.** The same spec (`1L_4H_d128`) is duplicated wherever a
  family uses it; the "many-to-many already real" relationship (one architecture
  across families — already true between the modular-addition and 2L-MLP lines) has
  no representation. You cannot ask "every variant on this architecture, across
  families."
- **The Family's two factors aren't separated in the Store.** Structural config is
  reached by digging into a family-config literal, not through an Architecture
  accessor — straining invariant 3 (storage internal to the API) and blocking the
  clean TaskType/Task split (REQ_159), which needs the structural half already lifted
  out.

This requirement commits **Architecture** as a first-class object and repoints
`Family → architecture_id`, leaving `family.json` to *reference* an architecture
rather than inline it.

## Conditions of Satisfaction

- [ ] **Architecture object** with the Part II fields — `architecture_id` (PK),
  `name` (e.g. `1L_4H_d128`), `n_layers`, `n_ctx`, `d_model`, `n_heads`, `d_mlp`
  (and the existing `d_head`/`act_fn`/`normalization_type` the runtime needs) —
  reachable through the API (an accessor, not a `family.json` literal).
- [ ] **`d_vocab` stays task-coupled — NOT an Architecture field.** It equals the
  Task's token count (`prime` + special tokens), fixed where Architecture meets Task
  (Part II note). Record the boundary; do not migrate `d_vocab` onto Architecture.
- [ ] **Family references Architecture by FK.** `Family.architecture_id` resolves to
  the Architecture; a Variant's structural config comes from `Family → Architecture`,
  not an embedded block. The `family.json` `architecture` block is replaced by an
  architecture reference (or normalized out).
- [ ] **Many-to-many representable.** Two families can share one Architecture
  identity (same `architecture_id`), and a query can enumerate variants by
  architecture across families.
- [ ] **Consumers read through the accessor.** Every reader of structural config
  (the REQ_136 `n_heads` head-axis coord, `d_model`, `d_mlp`, `n_ctx`, …) reaches it
  via the Architecture accessor, not the raw `family.json` block (invariant 3).
- [ ] **Migration preserves baselines.** Existing `family.json` architecture blocks
  map to Architecture identities with no behavioral change on the three baselines
  (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598) — analyzers and views produce
  identical output.

## Constraints

- **Storage internal to the API (invariant 3).** Consumers reach Architecture via an
  accessor (mirroring `variant.warehouse` / `variant.tensor_catalog`); no
  config-literal digging in library or app code.
- **Don't stand up a parallel store.** Realize Architecture on the existing
  warehouse/provenance surface (the REQ_156 "conform into the Store" dial), not a new
  registry system.
- **Pre-v1.0.0 — no back-compat shims.** Nothing is in production; migrate the
  config forward rather than carrying both shapes (memory `v1-cutoff-no-backcompat`).
  Migration must still preserve the *baselines'* data/behaviour.

## Open Questions (resolve in review before implementing)

1. **Realization surface.** Where does Architecture live — a warehouse table, a
   provenance/registry object, or a normalized top-level config (`data/architectures/`)
   referenced by `family.json`? This is the same Store-realization dial REQ_156 turned
   for circuits; recommend matching whatever that established.
2. **Identity scheme.** Canonical `name` (`1L_4H_d128`) as the natural key vs a
   surrogate `architecture_id`. Naming convention for the canonical name.
3. **Scope.** Both current families (`modulo_addition_1layer`, the 2L-MLP line) in
   this REQ, or just the modular-addition baselines first with the second family as a
   follow-up that proves the many-to-many?

## Notes

- This is deliberately the *small* half: the Architecture FK is structural and
  FK-stable for downstream tables (no `prime`-relocation churn). REQ_159 carries the
  harder semantic half (prime moves off Variant onto Task; the irrep-basis /
  master-dataset ownership moves off the Family implementation class).
- The 2L-MLP family (`TwoLayerMLPFamily`, REQ_088) is the concrete second user of the
  many-to-many — a different architecture on a related task line — and the cheapest
  proof that the relationship is real, not hypothetical.
