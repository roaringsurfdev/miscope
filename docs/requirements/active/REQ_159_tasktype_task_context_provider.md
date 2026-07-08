# REQ_159: TaskType / Task as the semantic-context provider (drain the Family leak)

**Status:** Specified (design settled in review; ready to implement). The semantic
half of the Layer 1 context-provider cluster — follows REQ_158 (Architecture).
**Priority:** High — realizes invariant #2 (semantic context belongs to
TaskType/Task, not Family), the most recent load-bearing data-model decision
(2026-06-28). Unblocks the right home for IrrepBasis (build-queue item 5), REQ_157's
task-conditional `dominant_frequency`, and per-task performance metrics.
**Branch:** TBD (`feature/REQ_159_tasktype_task`).
**Attribution:** Engineering Claude (under user direction).
**Depends on:** REQ_158 (Architecture lifted out + the seed-backed dimension-table
contract). **Blocks / informs:** REQ_157 (per-task frequency home), the
seed/bootstrap REQ (sequenced after this), IrrepBasis naming.
**Data model:** `data_model_master.md` Part II (TaskType / Task / Family + Variant
repoint, integrity constraint) and the Layer-1 build/realization decision note.

---

## Problem Statement

A **Family** is meant to *pair* an Architecture with a TaskType and own **no
semantics of its own** (invariant #2). Today it owns plenty:

- **`family.json` `domain_parameters` conflates two parameter spaces.** `prime` (a
  **Task** parameter — *which task*) sits beside `seed` / `data_seed` (**Model**
  parameters — *which training run*) in one undifferentiated block.
- **`prime` is denormalized onto the Variant.** It is expanded into `variant_cols`
  on every warehouse row and baked into the variant directory name (`p{prime}_…`), so
  cross-task comparison joins on a *Variant* attribute rather than a **Task** spine.
  `prime` identifies the Task, not the run — the seeds are what vary per Variant.
- **The semantic logic lives in the Family implementation class.** The irrep/Fourier
  **basis** construction and the **master dataset** (train/test/probes) are produced
  by `ModuloAddition1LayerFamily` — owned by the Family, exactly the ownership
  invariant #2 forbids. They belong to the **TaskType** (logic) / **Task** (the
  resolved instance: Z/109 ≠ Z/113).

This requirement commits **TaskType** and **Task** as objects, relocates `prime`'s
**ownership** to the Task, repoints `Family → tasktype_id` and `Variant → task_id`,
moves basis/master-dataset ownership to the TaskType/Task, and reworks **Variant
identity** so any parameter that varies a run is reflected in the run's identity.

## Resolved Decisions (from design review)

1. **Commit the *objects*; defer the parameter-*value* tables.** Build `TaskType` and
   `Task` (seed-backed dimension tables, per REQ_158's contract). **Defer both EAV
   value tables** — `Task_Parameter_Value` *and* the symmetric `Model_Parameter_Value`
   I had reached for. Rationale (symmetry, surfaced in review): a value/EAV table only
   earns its place when params go **heterogeneous across many types** — multiple
   TaskTypes with different params (`prime` vs `list_length` vs `n_bits`), or many
   varying hyperparameters. At today's scale (one TaskType, two universal seeds) typed
   attributes + identity encoding suffice. This is the doc's own
   **commit-object-defer-table** pattern (cf. `MethodRecipe`: "committed now, built
   later"). Build the EAV tables when a second TaskType (or a fleet of varying
   hyperparameters) forces it.
2. **Keep the parameter *declaration* — it is not the value table.** Review sharpened
   a distinction I had blurred: the **declaration** (`name`, `datatype`, **required /
   default**) is the contract the Training flow needs to *force population*; the
   **resolved value** (`prime=109`) is what we defer normalizing. The declaration
   stays as **authored config — the TaskType's declared parameter set, seed-tracked**
   (the evolution of `family.json`'s `domain_parameters`), not a warehouse table.
3. **Required = "no default."** A parameter with no default is required and must be
   supplied to create a Variant (an explicit `required: true` is an escape hatch for
   "has a default but must still be confirmed"). Concretely: **`prime` loses its
   `default: 113` → required** (you choose *which* mod-p; you don't train "the default
   task"); **`seed` / `data_seed` keep defaults**.
4. **Variant identity = resolved Model Parameters, serialized into the directory
   name, override-only.** A parameter is pushed into the name **only when supplied /
   non-default**, so introducing or touching a hyperparameter does **not** rename every
   existing Variant. This makes `variant_pattern` a *function that joins the present
   identity-bearing params*, not a fixed template. It closes a current latent bug: the
   Training page's **train/test split** knob can today create untracked divergence
   (same name, different split); as a declared Model Parameter it differentiates the
   Variant the moment it is changed.
5. **Name carries overrides; provenance carries the *full* resolved set.** Because the
   name suppresses defaults, the defaults would otherwise "have nowhere to surface" — a
   future default change would make implicit-default Variants ambiguous. So each
   Variant's **complete** resolved parameter set (defaults included) is **pinned in the
   seed/provenance**; the directory name is the derived, override-only collision key.
   *(This is primarily a constraint on the seed REQ; recorded here because REQ_159
   surfaced it.)*

## Conditions of Satisfaction

- [ ] **TaskType object** (`tasktype_id`, `name` e.g. `modular_addition`,
  `display_name`, `structure` e.g. `cyclic_group`, `description`) as a seed-backed
  dimension table, reachable through the API.
- [ ] **TaskType parameter declaration** (`name`, `datatype`, `default?`, → required
  when no default) carried as the TaskType's authored/seed-tracked declared set — not
  an EAV warehouse table. `prime` is declared **required** (no default).
- [ ] **Task object** (`task_id`, `tasktype_id`, the resolved `prime` as a typed
  attribute + encoded in the Task's name/identity, e.g. `modadd_109`), unique on
  `(tasktype_id, resolved-params)`. The **resolved-value EAV table is deferred**
  (recorded, not built).
- [ ] **`prime` ownership relocated to the Task.** The Task owns `prime`; the Variant
  no longer declares it. The cross-model comparison spine works: `WHERE prime=109` →
  `Task` → `Variant` enumerates models on the *same Task*. *(Denormalizing `prime`
  back onto variant rows for query convenience is allowed as a projection — ownership,
  not physical location, is the invariant-#2 fix.)*
- [ ] **FK repoint + integrity.** `Family.tasktype_id` and `Variant.task_id` FKs;
  enforce `Variant.family.tasktype_id == Variant.task.tasktype_id`.
- [ ] **Variant identity rework.** Model Parameters (seeds + conditionals) serialize
  into the directory name **only when non-default**; `prime` leaves the variant name
  (it is Task identity now, sourced from the Task — may still appear for readability,
  resolved via the Task). No existing baseline is renamed by introducing the new
  scheme (the three baselines keep `p113_seed999_dseed598`-equivalent identities).
- [ ] **Semantic ownership moves to TaskType/Task.** The irrep/Fourier basis and the
  master dataset (and any task-specific performance metric) are provided by the
  TaskType (logic) / Task (resolved artifacts), reachable through the API — the Family
  no longer constructs or owns them. The `Probe` derives from the Task's master
  dataset; the probe invariant (same Variant + same Probe across checkpoints) holds.
- [ ] **Migration preserves baselines.** The three baselines and the existing
  cross-data-seed / cross-model-seed comparisons (peer comparison, joint-PCA work)
  resolve identically; `prime`-keyed reads return the same variants.

## Constraints

- **Invariant #2 — context provider.** After this, semantic context (group structure,
  irrep basis, `prime`, master dataset) lives on TaskType/Task only. Views remain
  universal instruments; Families/TaskTypes register no views.
- **Invariant #3 — storage internal to the API.** `prime`, the basis, and the master
  dataset are reached through Task/TaskType accessors, not `family.json` literals or
  per-Variant columns.
- **Don't stand up a parallel store** (REQ_156 dial); **pre-v1.0.0 — no shims**
  (migrate forward; preserve baselines).

## Open Questions (resolved 2026-06-29 by user)

1. **Basis / master-dataset ownership move — scope the blast radius.** **Resolved:
   separate REQ.** This REQ commits the *objects + ownership* (TaskType/Task,
   `prime` ownership, FK repoint, identity rework); the physical relocation of the
   basis / master-dataset construction code out of `ModuloAddition1LayerFamily` is a
   distinct follow-up requirement. Keep REQ_159 scoped to the Store-level objects.
2. **`prime` read-path migration.** **Resolved: likely no blast radius.** The current
   aggregate accessor (`variant.params["prime"]`) is the right model+task parameter
   surface and may not need changing — the Task becomes the *owner*, but the existing
   aggregate read surface can keep resolving `prime` (sourced from the Task). Confirm
   the accessor name against the code during implementation; if it already aggregates
   model+task params, the downstream blast radius does not exist.
3. **IrrepBasis (build-queue item 5).** **Resolved: separate, thin follow-up.** Not
   every Task has an IrrepBasis (only Tasks with a clearly associated Group), so it is
   *not* a requirement for standing up the Task infrastructure. Build Task first;
   name the basis as a STABLE Task instrument afterward.

## Notes

- This is the **larger** slice (the `prime` read-path is broad). Recommend phases —
  objects + declaration + `prime` ownership + identity rework first; basis/dataset code
  move second — each independently verifiable against the baselines.
- Resolves Part VI #2 ("family object-type ownership" tension).
- **Frozen-attention** (Clock-and-Pizza-style pass-through attention to study
  attention-free behaviour) lands naturally here as a **conditional Model Parameter**
  (`freeze_attention`, default false → suppressed from the name; set → into identity) —
  *not* a new Architecture (same weights, frozen). Worth its own short research REQ
  later; the identity mechanism here is what makes it cleanly trackable.
- Once Task exists, REQ_157's `dominant_frequency` ("task-conditional" in the Layer 4
  tables) and IrrepBasis have an unambiguous home — the reason to do this before them.
