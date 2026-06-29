# REQ_158: Architecture as a first-class object (the model-side type)

**Status:** Specified (design settled in review; ready to implement). Entry slice of
the Layer 1 context-provider cluster — do before REQ_159 (TaskType/Task).
**Priority:** High — the data model marks Architecture "✅ commit now — many-to-many
already real" (Part V). Separating the structural factor is the smaller, FK-stable
half of the Family decomposition and unblocks the TaskType/Task slice.
**Branch:** TBD (`feature/REQ_158_architecture_object`).
**Attribution:** Engineering Claude (under user direction).
**Depends on:** none. **Blocks:** REQ_159 (TaskType/Task — the semantic half), and
the seed/bootstrap REQ (sequenced after 159).
**Data model:** `data_model_master.md` Part II (Architecture · `STABLE`), Part V scan
row "Architecture … ✅ commit now", and the Layer-1 build/realization decision note.

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

- **No identity, no sharing.** The same spec (`1L4Hd128`) is duplicated wherever a
  family uses it; the "many-to-many already real" relationship (one architecture
  across families) has no representation. You cannot ask "every variant on this
  architecture, across families."
- **The Family's two factors aren't separated in the Store.** Structural config is
  reached by digging into a family-config literal, not through an Architecture
  accessor — straining invariant 3 (storage internal to the API) and blocking the
  clean TaskType/Task split (REQ_159), which needs the structural half already lifted
  out.

This requirement commits **Architecture** as a first-class object and repoints
`Family → architecture_id`, leaving `family.json` to *reference* an architecture
rather than inline it.

## Resolved Decisions (from design review)

1. **Realization surface — a seed-backed warehouse dimension table.** Architecture
   lives in the **warehouse**, co-located with where `MethodRecipe` will, as a
   registry/dimension table with a **CRUD** accessor (mirroring `variant.warehouse` /
   `variant.tensor_catalog`). This is the warehouse's **first system-of-record
   table** — *authored* (given-real), not derived — which means two new contracts vs.
   the derived tables: (a) `materialize()` must **never wipe it blind**, and (b) it is
   **backed by a tracked, version-controlled seed** (the normalized authored registry
   that replaces the `family.json` blocks), so authored changes are reviewable in git
   and survive a clean checkout. The seed→table projection is the *same* two-layer
   write/read split the doc already endorses (Part 0), with an authored seed in the
   write slot instead of an analyzer artifact. *(The seed mechanism itself is a
   separate REQ, sequenced after 158/159; this REQ defines the table + accessor and
   consumes the seed contract.)*
2. **Naming — short natural key + derived Family name.** Architecture carries a
   short `name` (the natural key, e.g. `1L4Hd128`) and a verbose `display_name`. A
   `Family`'s name is the **composition** of its factors' short names
   (`{tasktype}_{architecture}`) and is **derived**, not stored independently (no
   drift), with an optional explicit-override field. The current
   `modulo_addition_1layer` already *is* this composition (task `modulo_addition` ×
   arch `1layer`).
3. **Scope — modular-addition baselines; many-to-many proven-in-schema now.** Build
   Architecture against the modular-addition family; the M:N relationship is real in
   the *schema* with one architecture populated. Prove it *with data* later, when a
   genuinely different architecture is warranted. **Leave the partial 2L-MLP family
   untouched** — and note that the "attention-free model" goal it was meant to serve
   is likely better met by *frozen pass-through attention* on the existing 1L family
   (a Model Parameter / intervention — REQ_159 / a research thread — **not** a new
   Architecture), so 2L-MLP may not need rebuilding at all.

## Conditions of Satisfaction

- [ ] **Architecture object** with the Part II fields — `architecture_id` (PK),
  `name` (short natural key), `display_name`, `n_layers`, `n_ctx`, `d_model`,
  `n_heads`, `d_mlp` (and the runtime's `d_head`/`act_fn`/`normalization_type`) —
  reachable through a CRUD accessor, not a `family.json` literal.
- [ ] **Realized as a seed-backed warehouse dimension table:** `materialize()` does
  not wipe it; it loads/projects from the tracked seed; an authored change to an
  Architecture is a reviewable git diff.
- [ ] **`d_vocab` stays task-coupled — NOT an Architecture field.** It equals the
  Task's token count (`prime` + special tokens), fixed where Architecture meets Task.
  Record the boundary; do not migrate `d_vocab` onto Architecture.
- [ ] **Family references Architecture by FK.** `Family.architecture_id` resolves to
  the Architecture; a Variant's structural config comes from `Family → Architecture`,
  not an embedded block. The `family.json` `architecture` block is replaced by a
  reference. `Family.name` is derived from the factor short names.
- [ ] **Many-to-many representable.** Two families can share one `architecture_id`,
  and a query can enumerate variants by architecture across families.
- [ ] **Consumers read through the accessor.** Every reader of structural config (the
  REQ_136 `n_heads` head-axis coord, `d_model`, `d_mlp`, `n_ctx`, …) reaches it via
  the Architecture accessor, not the raw `family.json` block (invariant 3).
- [ ] **Migration preserves baselines.** Existing architecture blocks map to
  Architecture identities with no behavioral change on the three baselines
  (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598) — analyzers and views produce
  identical output.

## Constraints

- **Storage internal to the API (invariant 3).** Consumers reach Architecture via the
  accessor; no config-literal digging in library or app code.
- **Don't stand up a parallel store.** Realize Architecture on the existing
  warehouse/provenance surface (the REQ_156 "conform into the Store" dial).
- **Pre-v1.0.0 — no back-compat shims.** Migrate the config forward rather than
  carrying both shapes (memory `v1-cutoff-no-backcompat`); preserve the baselines.

## Notes

- This is deliberately the *small* half: the Architecture FK is structural and
  FK-stable for downstream tables (no `prime`-relocation churn). REQ_159 carries the
  harder semantic half (prime moves off Variant onto Task; the irrep-basis /
  master-dataset ownership moves off the Family implementation class).
- The seed-backed-dimension-table contract introduced here is reused verbatim by
  REQ_159 for TaskType/Task and by the later seed/bootstrap REQ.
