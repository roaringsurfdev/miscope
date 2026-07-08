# REQ_135: Retire the per-variant `manifest.json`

**Status:** Completed (2026-06-06) — implemented on `feature/REQ_135_manifest_retirement`.
**Priority:** Low — cleanup; removes a drift surface, no feature impact.
**Branch:** `feature/REQ_135_manifest_retirement`
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Every analyzed variant carries a `manifest.json` under its `artifacts/` directory
(47 on disk as of 2026-06-04). The pipeline writes and merges it on every run
(`_update_manifest` / `_save_manifest`), recording per-analyzer
`epochs_completed` / `shapes` / `dtypes`, plus `variant_params`, `family_name`,
and `model_config`.

Nothing in production reads it. The only callers of the reader surface
(`ArtifactLoader.manifest`, `.get_metadata`, `.get_model_config`) are
`packages/miscope/tests/test_artifact_loader.py`. Every value it holds is now
authoritative elsewhere:

- `shapes` / `dtypes` / `epochs_completed` — derivable from the npz files
  themselves, and declared canonically on `AnalyzerSpec.outputs` (REQ_107) /
  surfaced by the columnar warehouse (REQ_110A).
- `model_config` — `config.json` (`variant.config_path`).
- `variant_params` / `family_name` — `variant.params` / `variant.family`.

So the manifest is a **write-only drift surface**: a cache of shapes/dtypes that
can silently diverge from the actual artifacts while no decision depends on it.
This conflicts with the discoverability-first / single-source-of-truth direction
REQ_107 established.

## Conditions of Satisfaction

- [x] **Stop writing it.** Removed `_update_manifest` / `_save_manifest` /
  `_load_manifest` and the `self._manifest` plumbing from
  `analysis/pipeline.py`; the pipeline no longer creates or merges `manifest.json`.
  (Now-unused `json` and `datetime` imports dropped too.)
- [x] **Remove the reader surface** from `ArtifactLoader` (`manifest` property,
  `get_metadata`, `get_model_config`, `_load_manifest`, `self._manifest`). No
  genuine consumer found — every caller was a test. (Unused `json` / `typing.Any`
  imports dropped.)
- [x] **Update / drop the manifest tests** in `test_artifact_loader.py` and
  `test_analysis_pipeline.py` accordingly (the only readers). Two pipeline tests
  that were named for the manifest but actually assert filesystem-based completion
  detection were kept and renamed (`test_completion_persists_between_sessions`,
  `test_completion_tracked_for_all_analyzers`).
- [x] **Delete the stale `manifest.json` files** under `data/.../artifacts/`
  (47 removed; regeneratable / now unused; gitignored already).
- [x] **No behavior change** elsewhere: dashboard and analyzers already use
  filesystem scans (`get_epochs`, `get_available_analyzers`) and `config.json`,
  not the manifest. Full miscope suite green (1543 passed; one pre-existing WSL2
  mtime flake in `test_secondary_force_recomputes`, unrelated — passes in isolation).

## Constraints

- Honor the storage-encapsulation invariant: the cleanup is internal to the
  storage primitive (`ArtifactLoader`) and the pipeline; no consumer should need
  to change because none reads the manifest today.
- If implementation uncovers a real consumer (not a test), **stop and flag** —
  the premise (write-only) would be wrong and the field should be repointed, not
  dropped.

## Notes

- Surfaced 2026-06-04 during REQ_110A review (the warehouse made the redundancy
  obvious — shapes/dtypes/epochs now come from declared schema + materialized
  Parquet).
- Small, self-contained; a good candidate to fold into the next v1.0 cleanup pass
  alongside REQ_129 (dead-code cull) if convenient.
