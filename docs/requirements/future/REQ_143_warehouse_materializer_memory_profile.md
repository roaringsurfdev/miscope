# REQ_143: Profile the Warehouse Materializer's Memory Scaling

**Status:** Future (parking lot — captured 2026-06-06 from the run_with_cache memory investigation).
**Priority:** Medium — scalability characterization ahead of publication; not a defect.
**Branch:** TBD (`feature/REQ_143_materializer_memory_profile`).
**Dependencies:** REQ_110A (`warehouse` columnar tables), REQ_110B (tensor catalog), REQ_140 (materializer scope/isolation — the "~20 GB steady-state" observation originates there). Related to REQ_103 (publication gate).
**Attribution:** Engineering Claude (under user direction).

---

## Problem Statement

The 2026-06-06 memory investigation
(`docs/notes/infra_discovery/memory_investigations.md`) cleared the analysis path:
process RSS is flat for `run_with_cache`, per-epoch model recreation, and the full
`AnalysisPipeline`. It also established that `AnalysisPipeline.run` **never invokes
the warehouse materializer** — they are separate code paths.

That leaves one genuinely RSS-bound stream uncharacterized: the **warehouse
materializer**, which the REQ_140 notes record as carrying a ~20 GB steady-state
footprint. Unlike the analysis pipeline (small, flat, per-epoch-streamed), the
materializer's memory is real process RSS and is the surface most likely to gate
scalability as the number of variants, epochs, and tensor fields grows — exactly
the dimension a publication exposes to researchers with larger or smaller machines
than ours.

Today that ~20 GB is an anecdote, not a curve. Before publication we should know:
how does materializer peak RSS scale with (variants × epochs × fields), where is
the high-water mark within a build, and is the footprint inherent (a whole-table
load before write) or incidental (an avoidable buffering choice). The answer
decides whether anything beyond documentation is needed.

## Conditions of Satisfaction

- [ ] A reproducible profiling harness measures the materializer's **peak process
  RSS** (not total system used — see the measurement protocol in the notes) for a
  warehouse build, parameterized by input size (variant count, epoch count,
  and/or which analyzers/tables are emitted).
- [ ] A short results write-up in `docs/notes/infra_discovery/memory_investigations.md`
  reports: peak RSS at representative scales, the dominant contributor (which
  table/step holds the high-water mark), and whether peak scales linearly /
  super-linearly with input size.
- [ ] A verdict is recorded: peak RSS is either (a) acceptable and inherent →
  document the expected footprint as a system requirement; or (b) reducible →
  file a focused follow-up REQ for the specific fix (e.g. stream-by-coord-signature
  instead of whole-table buffering), with the profile as its evidence.
- [ ] Profiling uses a throwaway/representative data root and does not mutate the
  pinned baselines.

## Constraints

**Must:**
- Measure process RSS, per the investigation's measurement protocol; do not
  diagnose from Windows Task Manager "total used."
- Treat this as characterization first — do not pre-commit to a rewrite. The fix
  (if any) is a separate REQ scoped by what the profile actually shows.

**Must avoid:**
- Conflating the materializer footprint with the (already-cleared) analysis-path
  memory or with reclaimable page cache.
- Scope creep into a general warehouse perf overhaul. This REQ answers one
  question: how does materializer RSS scale, and is it a problem.

## Notes

- The "~20 GB steady-state" figure lives in the REQ_140 memory note as a deferred
  stream; this REQ is where it gets a real curve.
- Pairs with REQ_142 (WSL2 setup guidance) for a complete publication memory
  story: analysis flat, cache reclaimable + documented, materializer characterized.
- If the profile shows linear, modest scaling, the outcome may be purely a
  documented system requirement — which is itself a valid and valuable result.
