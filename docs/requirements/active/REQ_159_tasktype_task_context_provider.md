# REQ_159: TaskType / Task as the semantic-context provider (drain the Family leak)

**Status:** Draft (problem-first; for review). The larger, semantic half of the
Layer 1 context-provider cluster — follows REQ_158 (Architecture).
**Priority:** High — realizes invariant #2 (semantic context belongs to
TaskType/Task, not Family), the most recent load-bearing data-model decision
(2026-06-28). Unblocks the right home for IrrepBasis (build-queue item 5), REQ_157's
task-conditional `dominant_frequency`, and per-task performance metrics.
**Branch:** TBD (`feature/REQ_159_tasktype_task`).
**Attribution:** Engineering Claude (under user direction).
**Depends on:** REQ_158 (Architecture lifted out first). **Blocks / informs:**
REQ_157 (per-task frequency home), IrrepBasis naming.
**Data model:** `data_model_master.md` Part II (TaskType / TaskType_Parameter / Task
/ Task_Parameter_Value / the Family + Variant repoint, integrity constraint).

---

## Problem Statement

A **Family** is meant to *pair* an Architecture with a TaskType and own **no
semantics of its own** (invariant #2). Today it owns plenty:

- **`family.json` `domain_parameters` conflates two parameter spaces.** `prime` (a
  **Task** parameter — *which task*) sits beside `seed` / `data_seed` (**Model**
  parameters — *which training run*) in one undifferentiated block. The old name
  `Family_Parameter` "already read 'per-family **task** parameters' — the leak named
  in plain sight."
- **`prime` is denormalized onto the Variant.** It is expanded into `variant_cols`
  on every warehouse row, so cross-task comparison (`WHERE prime=109`) joins on a
  *Variant* column rather than a **Task** spine. `prime` identifies the Task, not the
  run — the seeds are what vary per Variant.
- **The semantic logic lives in the Family implementation class.** The irrep/Fourier
  **basis** construction and the **master dataset** (train/test/probes) are produced
  by `ModuloAddition1LayerFamily`, i.e. owned by the Family — exactly the ownership
  invariant #2 forbids. They belong to the **TaskType** (logic) / **Task** (the
  resolved instance: Z/109 ≠ Z/113).

This requirement commits **TaskType**, **TaskType_Parameter**, **Task**, and
**Task_Parameter_Value**; relocates `prime` off the Variant onto the Task; repoints
`Family → tasktype_id` and `Variant → task_id`; and moves basis/master-dataset
ownership to the TaskType/Task.

## Conditions of Satisfaction

- [ ] **TaskType object** (`tasktype_id`, `name` e.g. `modular_addition`,
  `structure` e.g. `cyclic_group`, `description`) + **TaskType_Parameter**
  (`prime` declared here; renamed from `Family_Parameter`). Reachable through the API.
- [ ] **Task object** (`task_id`, `tasktype_id`) + **Task_Parameter_Value** (`prime`
  resolved here, e.g. `prime=109`; renamed from `Variant_Parameter`), unique on
  `(tasktype_id, resolved-params)`.
- [ ] **`prime` relocated off the Variant onto the Task.** `seed` / `data_seed`
  remain Variant **Model Parameters** (standing columns). The cross-model comparison
  spine works: `WHERE prime=109` → `Task` → `Variant` enumerates models on the *same
  Task* (no longer a per-Variant column).
- [ ] **FK repoint + integrity.** `Family.tasktype_id` and `Variant.task_id` FKs;
  enforce `Variant.family.tasktype_id == Variant.task.tasktype_id` (the template's
  TaskType and the bound Task's TaskType agree).
- [ ] **Semantic ownership moves to TaskType/Task.** The irrep/Fourier basis and the
  master dataset (and any task-specific performance metric) are provided by the
  TaskType (logic) / Task (resolved artifacts), reachable through the API — the
  Family no longer constructs or owns them. The `Probe` derives from the Task's
  master dataset (Part II).
- [ ] **Migration preserves baselines.** The three baselines and the existing
  cross-data-seed / cross-model-seed comparisons (peer comparison, joint-PCA work)
  resolve identically; `prime`-keyed reads return the same variants.

## Constraints

- **Invariant #2 — context provider.** After this, semantic context (group
  structure, irrep basis, `prime`, master dataset) lives on TaskType/Task only.
  Views remain universal instruments; Families/TaskTypes register no views.
- **Invariant #3 — storage internal to the API.** `prime`, the basis, and the master
  dataset are reached through Task/TaskType accessors, not `family.json` literals or
  per-Variant columns.
- **Don't stand up a parallel store** (REQ_156 dial); **pre-v1.0.0 — no shims**
  (migrate forward; preserve baselines).
- **Probe invariant holds.** Same Variant + same Probe across all checkpoints; the
  Probe now derives from the Task's master dataset without changing that invariant.

## Open Questions (resolve in review before implementing)

1. **Realization surface** — same dial as REQ_158/156: warehouse tables vs
   provenance objects vs normalized config for TaskType/Task/parameter tables.
2. **Basis / master-dataset ownership move** — does this REQ physically relocate the
   construction code out of `ModuloAddition1LayerFamily` into a TaskType-owned
   surface, or commit the *objects + provenance* now and relocate the code in a
   follow-up? (Scoping the blast radius — the implementation class is load-bearing.)
3. **`prime` read-path migration** — every consumer reading `variant.model_config["prime"]`
   / the `prime` warehouse column repoints to the Task. Enumerate the surface (it is
   broad: analyzers, views, cross_variant) and decide big-bang vs phased.
4. **IrrepBasis (build-queue item 5)** — fold the "name the basis as a STABLE Task
   instrument" move into this REQ (it belongs under TaskType/FrequencyMode), or keep
   it a thin follow-up once Task exists?

## Notes

- This is the **larger** slice and the higher blast radius (the `prime` read-path is
  broad). Recommend landing REQ_158 first, then sequencing this with explicit
  phases — objects + provenance + `prime` relocation before the basis/master-dataset
  ownership move — so each phase is independently verifiable against the baselines.
- Resolves Part VI #2 ("family object-type ownership" tension) in the data model.
- Once Task exists, REQ_157's `dominant_frequency` ("task-conditional" in the Layer 4
  tables) and IrrepBasis have an unambiguous home — the reason to do this before them.
