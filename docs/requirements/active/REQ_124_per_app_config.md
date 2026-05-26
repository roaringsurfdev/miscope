# REQ_124: Per-App Configuration Files

**Status:** Drafted (stub — Problem Statement only; CoS to be filled in during REQ_123 wrap-up or shortly after).
**Priority:** Medium — independent of any in-flight work; sequenced behind REQ_123 (Unified Data Root) so the per-app schema reflects the final data-root surface.
**Branch:** TBD
**Dependencies:** REQ_123 (Unified Data Root) should land first so the config schema lines up with the unified layout. No hard blockers otherwise.
**Coordinates with:**
- *REQ_123 — Unified Data Root.* REQ_123 introduces the `data/` root and an updated config surface; REQ_124 builds on whatever schema REQ_123 settles on.
- *REQ_125 — Research Notebook & Sketch Retrofit.* The path-abstraction policy that REQ_123 codifies is enforced uniformly by REQ_125 in research code and by REQ_124 in app code.
**Attribution:** Engineering Claude (drafted as a follow-up captured during REQ_122 review on 2026-05-26).

---

## Problem Statement

Each application in the monorepo (`apps/dashboard/`, `apps/fieldnotes/`, `apps/research/`, plus the operational scripts under `scripts/`) has its own deployment surface and may need to know its own data paths, ports, and environment-specific values. Today `miscope.config.AppConfig` is a single shared schema configured via env vars (`MISCOPE_PROJECT_ROOT`, `MISCOPE_RESULTS_DIR`, `MISCOPE_MODEL_FAMILIES_DIR`). This conflates "API defaults" with "app deployment configuration":

- `apps/dashboard/src/dashboard/state.py` hardcodes `Path("model_families")` and `Path("results")` when constructing the family map. The same literals appear in `apps/dashboard/src/dashboard/pages/initialization_sweep.py` and `viability_certificate.py`. These are deployment-time decisions baked into code at import time.
- Scripts re-derive paths from `PROJECT_ROOT` constants or env vars without a clear contract.
- There is no place for an app to declare deployment-specific values (e.g., dashboard host/port, fieldnotes build path) that are unrelated to the library's data root.

The expectation that emerged during REQ_122 review: each app declares its configuration **once, at the app root**, in a way that supports deployment to multiple environments (DEV / STAGING / PROD) without code changes. App startup loads its config and supplies the relevant slice to the API via the existing `AppConfig` override mechanism. The library config becomes the *default* (or, equivalently, the source-of-truth schema) and the per-app file becomes the configured deployment value. No file paths live in app code outside this config.

The pythonic shape is open at implementation time — JSON files at app root, TOML in `pyproject.toml`, or pydantic-validated config classes are all plausible. The constraint is *not* the file format but the rule: **explicit file paths and deployment values do not live in code outside configuration.**

---

## Notes

- This REQ codifies a policy the project intends to add to `PROJECT.md` (alongside "views are universal" / "families are context providers"). The policy itself — "no file paths outside config; storage layout is internal to the API" — lands as part of REQ_123 so it informs the unified data root work; this REQ is the application-side enforcement.
- Once landed, the dashboard / fieldnotes / research apps each have a `config.{json,toml}` at their root, validated against the shared schema, with environment-specific overrides (e.g. `config.staging.json`, env-var overlay) as a follow-up if needed.
- Adjacent but separate: REQ_125 covers the research notebook / sketch retrofit so existing examples stop re-introducing path literals.

---

## Out of Scope

- The data-root layout itself (REQ_123).
- The notebook / sketch retrofit (REQ_125).
- CI/CD or multi-environment deployment infrastructure beyond making the code *capable* of being deployed to different environments without changes.
