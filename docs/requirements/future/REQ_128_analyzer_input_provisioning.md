# REQ_128: Analyzer Input Provisioning — Lazy Accessor vs. Eager Materialization

**Status:** Draft — *Problem statement only.* Conditions of satisfaction, constraints, and scoping decisions to be developed in a dedicated session.
**Priority:** Medium — architectural; not blocking, but the eager-materialization pattern is a standing memory-pressure source that compounds as analyzers chain and as artifacts grow more granular.
**Branch:** TBD (the design exploration originated on `feature/generic_analyzer`).
**Dependencies:**
- REQ_106 (Analysis Layer Architecture — defines the declared-dependencies discipline this REQ would extend to the *loading* of those dependencies).
- REQ_109 (primitive layer — the granular analyzers that make whole-artifact loads expensive consume it).
- REQ_122 (storage-encapsulation invariant — constrains the solution space: analyzers must reach data through accessors, never raw paths).

**Relationships (not hard dependencies):**
- REQ_127 (Downstream Visualization Migration — *complete*). The **consumer-side analog** of this problem: REQ_127 fixed an OOM where view loaders stacked whole granular artifacts via `load_epochs(...)` and resolved it with `fields=` selective loading. This REQ is the **producer/chaining-side** version of the same tension.
- The Plan/Planner / Freshness infrastructure (from the ETL refactor) is the adjacent system that decides *what* to compute; this REQ is about *how the data flows in* once computation is scheduled.

**Attribution:** Engineering Claude (drafted 2026-05-27, surfaced during REQ_127's memory investigation and recalling the `feature/generic_analyzer` ETL exploration).

---

## Problem Statement

The analysis pipeline materializes analyzer inputs **eagerly**. For each per-epoch analyzer, `pipeline._materialize_per_epoch_inputs` builds a `ResolvedInputs` holding `model`, `cache`, `logits`, `probe`, and — for every `ArtifactInput` the analyzer's Spec declares — the **fully loaded** upstream artifact (`loader.load_epoch(dep, epoch)` / `load_summary` / `load`, with no field filter). A chained analyzer (e.g. `centroid_fourier_alignment` depending on `repr_geometry`) therefore receives the entire upstream artifact dict in memory, whether or not it reads all of it.

This couples two things that should be separable:

1. **What an analyzer depends on** (declared, static — good).
2. **When and how much of that dependency is resident in memory** (currently: all of it, for the lifetime of the `ResolvedInputs` object).

Two consequences follow:

- **Whole-artifact loads even for partial reads.** Post-REQ_126 granular analyzers emit large per-site cubes (e.g. `mlp_out_power` at `(d_mlp, n_freq, n_freq)`). A chained analyzer that needs one small key still triggers a load of the whole artifact. This is the *same* failure mode REQ_127 hit on the consumer side and fixed with `load_epochs(fields=...)` — but the producer/chaining side has no equivalent lever today.
- **Memory lifetime tied to an object, with no clean release.** REQ_127 Phase A.2f added a defensive fix in `_run_single_epoch` (null `inputs`/`result` before `del model, cache, ...` so `torch.cuda.empty_cache()` can reclaim). That treats the *symptom* (the last analyzer's `ResolvedInputs` keeps model/cache references alive past the explicit cleanup) rather than the *structure* (eager materialization makes the resident set as large as everything declared, released only when the inputs object dies).

The `feature/generic_analyzer` ETL exploration surfaced the core design question directly: **how should data be passed to analyzers?** If everything is pre-loaded and handed to the input object, it can be a lot of data with no clean release path — especially for chained analyzers. One option considered there was to pass **file paths** that each analyzer loads on demand.

The opportunity: pick a provisioning model that lets analyzers load **only what they need, when they need it, in a releasable scope**, without eroding the declared-dependency discipline or the storage-encapsulation invariant.

---

## Context: the design axes (framing, not decisions)

Three provisioning models, with the tension between ergonomics and control:

1. **Eager dict (current).** The pipeline pre-loads each declared dependency and hands the analyzer a ready dict (`inputs.artifacts["repr_geometry"]`). Dead simple for analyzer authors; centralized loading. But loads everything declared, and memory lifetime is the inputs object's lifetime.

2. **Raw file paths.** The analyzer receives paths and loads on demand. Gets on-demand release and selective reads — but **violates the storage-encapsulation invariant** (PROJECT.md constraint 3: only storage primitives compose paths; everything else uses accessors). An analyzer composing paths reaches past the API. Likely off the table for this reason, but recorded because it was the original idea on `feature/generic_analyzer`.

3. **Lazy accessor handle (recommended starting point).** The analyzer receives a scoped `ArtifactLoader` (or a thinner accessor) and calls `load_epoch(dep, epoch, fields=[...])` on demand. This is the invariant-preserving version of the path-passing idea: it gives the same on-demand, selective, releasable loading, but the analyzer still reaches data only through the accessor. The cost is ergonomics — a `.load()` call instead of a ready dict — and the need for a convention (or Spec-level validation) that an analyzer only loads what its Spec declares, so the declared-dependency discipline doesn't erode into ad-hoc reads.

Open scoping questions (for the dedicated session):

- **Selective fields at the dependency level.** Should an `ArtifactInput` be able to declare *which fields* it needs, so the pipeline can load selectively even in an eager model — a smaller change than full lazy provisioning? (Mirrors REQ_127's `fields=` exactly.)
- **Declared-dependency enforcement.** If analyzers get a loader handle, how is "only load what you declared" enforced — convention, a wrapping accessor scoped to the declared deps, or a post-hoc audit/grep test (REQ_106 style)?
- **Interaction with the per-epoch cleanup.** A lazy/scoped model should make the REQ_127 Phase A.2f reference-nulling band-aid unnecessary. Confirm, and remove the band-aid if so.
- **Cross-epoch and summary scopes.** The same question applies to `scope="all_epochs"` and `scope="summary"` inputs, which can be the largest loads of all.
- **Migration cost.** How many analyzers read `inputs.artifacts[...]` today, and what's the blast radius of changing the contract? (Bounded by the Spec-declared `ArtifactInput` set.)

---

## Conditions of Satisfaction

*(Deferred to a dedicated scoping session — this is a problem-statement draft.)*

---

## Notes

- **This is the producer-side twin of REQ_127.** That REQ's `feedback_granular_analyzer_load_epochs_fields` lesson (consumer loaders must use `fields=`) is the same insight applied to view loaders; REQ_128 asks whether the *analyzer input* path deserves the same treatment structurally rather than per-call.
- The minimal viable version might be just "let `ArtifactInput` declare fields, and have `_materialize_per_epoch_inputs` honor them" — a small, low-risk change that captures most of the memory win without re-architecting to lazy handles. The fuller lazy-accessor model is the more complete answer but a bigger contract change. The scoping session should weigh minimal-fields vs. full-lazy.
- Revisit the REQ_127 Phase A.2f pipeline band-aid (`inputs = result = summary = None` before `del`) here — a cleaner provisioning model likely subsumes it.
