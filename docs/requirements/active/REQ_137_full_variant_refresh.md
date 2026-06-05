# REQ_137: Full Variant Refresh (dense checkpointing + artifact regeneration)

**Status:** Draft (stub — for review)
**Priority:** Medium — unblocks trustworthy whole-corpus verification; deferred because the churn is large and slow.
**Branch:** TBD
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Only three variants are currently carried forward through each refactor and kept
as trustworthy ground truth:

- **p113/s999/ds598** (canon)
- **p109/s485/ds598**
- **p101/s999/ds598**

Every other on-disk variant is **pre-refresh**: it may carry stale analyzer
artifacts, old array shapes (pre-REQ_136 head axis), deprecated analyzer outputs,
and dtype/shape drift relative to the current REQ_107 declarations. These stale
variants are actively misleading during validation — REQ_110B's verification pass
hit exactly this: a `parameter_trajectory.projections` "dtype drift finding"
observed only on `p109/s485/ds42` (a non-baseline variant) turned out to be a
staleness artifact, absent on every refreshed baseline.

The refresh is held off deliberately because doing it across the whole corpus is
large and slow. This REQ tracks that work so it is scheduled, not forgotten, and
so the **verification-scope rule** (below) has an owner and an end date.

## Verification-Scope Rule (in force until this REQ completes)

Until the full refresh lands, **validate refactors only against the three
baselines above.** A result, regression, or "finding" observed solely on a
non-baseline variant is presumed a staleness artifact, not a real finding, until
reproduced on a baseline. (Mirrored into agent memory so it survives across
sessions.)

## Refresh procedure (per variant)

1. **Re-train with dense checkpointing.** Seed lockdown makes training
   deterministic, so results should be identical to the existing run; the only
   change is denser checkpoint coverage. (Confirm determinism on the first
   re-trained variant before fanning out.)
2. **Purge deprecated analyzer artifacts.** Remove artifacts for retired
   analyzers so nothing stale survives the refresh (cf. `scripts/prune_deprecated_artifacts.py`).
3. **Re-analyze** to regenerate fresh `.npz` artifacts, the columnar warehouse
   (REQ_110A), and the tensor catalog (REQ_110B).

## Conditions of Satisfaction (to flesh out at scoping)

- [ ] Every tracked variant re-trained with dense checkpointing; determinism
      confirmed against the prior run on at least one variant.
- [ ] Deprecated-analyzer artifacts purged corpus-wide; no retired-analyzer
      directories remain.
- [ ] Fresh artifacts + warehouse + tensor catalog regenerated for every variant.
- [ ] `variant_registry.json` / `variant_summary.json` recompiled.
- [ ] Verification-scope rule retired (memory + this REQ) once the corpus is
      uniformly fresh; the baseline-only restriction is lifted.

## Other relevant tasks to sweep in (candidates)

- ~~Fix the **`dominant_frequency_pair` dtype declaration** (`float64` → `int32`)
  in `weight_basis_projection` and `activation_basis_projection`.~~ **DONE on the
  REQ_110 branch** (both analyzers bumped to `version=2`; field pulled out of the
  float64 tensor-field loop and declared `int32`). The on-disk bytes were already
  `int32`, so existing baseline artifacts now match the declaration — re-analysis
  is not required to clear it, but the refresh will regenerate them regardless.
- Reconcile the Atlas vs `family.json` analyzer ground truth during the
  re-analyze pass (see memory: *Atlas vs family.json ground truth*).

## Notes

- Determinism caveat: if dense checkpointing changes the checkpoint *cadence* in
  a way that touches RNG draw order (e.g. extra eval passes consuming the RNG),
  confirm the loss curve still matches before trusting "identical results."
- This is a stub for review — the user decides when the churn is worth paying.
