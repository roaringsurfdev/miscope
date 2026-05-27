# REQ_124: Per-App Configuration Files & Storage-Encapsulation Enforcement

**Status:** CoS drafted; implementation queued behind REQ_123 (now landed on `develop`). Ready to pick up.
**Priority:** Medium — the path-literal enforcement closes a structural gap that REQ_123's data-root unification opened; the per-app config piece is small but unlocks deployable dashboard configuration.
**Branch:** `feature/req-124-per-app-config` (to be created at start of implementation).
**Dependencies:** REQ_123 (Unified Data Root) — landed. This REQ assumes `cfg.data_root`, `family.family_dir`, `family.variants_dir` exist.
**Coordinates with:**
- *REQ_125 — Research Notebook & Sketch Retrofit.* REQ_124 adds the API accessors (`variant.summary`, `family.variant_registry`) that REQ_125's research retrofit will lean on. REQ_125's remaining scope (`ArtifactLoader` privacy + template polish + scripts/ audit) is independent of this REQ.
- *REQ_103 / REQ_108 — Publication hardening / surface.* Both consume a clean app-config + storage-encapsulation surface; doing REQ_124 first makes those REQs easier.
**Attribution:** Engineering Claude (drafted as a follow-up captured during REQ_122 review on 2026-05-26; CoS filled in during REQ_123 wrap-up on 2026-05-26).

---

## Problem Statement

Two related gaps that REQ_123's data-root unification did not close:

1. **Deployment-specific values are baked into app code.** `apps/dashboard/src/dashboard/app.py:86` hardcodes `host="0.0.0.0"`, `port=8060`, `debug=False`. There is no place for an app to declare deployment-time values without code changes. The library's `AppConfig` (which now carries `data_root`) is the wrong place for these — it describes library data paths, not app deployment values.

2. **Storage layout still leaks past the API in many sites.** REQ_123 codified the third architectural invariant ("storage layout is internal to the API") but the enforcement pass closed only the obvious `Path("results/...")` literals. The `variant.variant_dir / "variant_summary.json"` pattern persists in:
   - 5 dashboard pages/components (including one `Path("results/...")` regression in `viability_certificate.py:31` that REQ_123 missed).
   - 8+ library sites (`miscope.views.universal`, `miscope.analysis.saddle_transport`, `miscope.analysis.freshness`, `miscope.analysis.analyzers.gradient_site`, plus the variant_summary modules themselves).
   - `apps/dashboard/src/dashboard/pages/initialization_sweep.py` composes `data_root / family / "variant_registry.json"` via a private `_registry_path()` helper.

   The pattern leaks because no `variant.summary` or `family.variant_registry` accessor exists. The invariant says: if the accessor isn't there, add it — don't reach past the API.

### What this REQ does

Two coordinated moves:

- **Adds the missing accessors** — `variant.summary` (parsed `variant_summary.json`) and `family.variant_registry` (parsed `variant_registry.json`). Both are simple JSON-loading properties; both raise `FileNotFoundError` when absent.
- **Establishes per-app config** — `apps/dashboard/config.toml` becomes the single source for dashboard deployment values. A small loader (stdlib `tomllib`, no new dep) returns a frozen dataclass. Dashboard startup reads it; the rest of the app stays unaware.

Once both are in place, the enforcement pass replaces every `variant.variant_dir / "variant_summary.json"` and every registry-path composition with the new accessors, and removes the hardcoded host/port from `app.py`.

---

## Conditions of Satisfaction

### API additions (prerequisite for enforcement)

- [ ] **`Variant.summary` property** — returns the parsed dict from `{variant_dir}/variant_summary.json`. Raises `FileNotFoundError` if the file is missing, with a message naming the variant.
- [ ] **`BaseModelFamily.variant_registry` property** — returns the parsed list from `{family_dir}/variant_registry.json`. Raises `FileNotFoundError` if absent, with a message naming the family.
- [ ] Both accessors read on each access (no caching); callers that need caching assign to a variable. Same pattern as the existing `variant.metadata` / `variant.model_config` properties.
- [ ] **Protocol updated** — `ModelFamily` protocol in `families/protocols.py` declares `variant_registry`.

### Per-app config (dashboard only)

- [ ] **`apps/dashboard/config.toml`** — tracked in git; declares dashboard deployment values:
  ```toml
  [server]
  host = "0.0.0.0"
  port = 8060
  debug = false
  ```
- [ ] **`apps/dashboard/src/dashboard/config.py`** — defines `DashboardConfig` (frozen dataclass) and `load_dashboard_config(path: Path | None = None) -> DashboardConfig`. Uses stdlib `tomllib`. Validates required keys; raises a clear error on malformed or missing file.
- [ ] **Loader behavior:**
  - Default path: `apps/dashboard/config.toml` (resolved relative to the dashboard package).
  - Missing required keys → `ValueError` with the missing-key list.
  - Unknown keys → silently ignored (forward-compatible); document this.
- [ ] **Scope explicitly:** only the dashboard gets a `config.toml` in this REQ. Fieldnotes has `astro.config.mjs`; research is exploratory; scripts use CLI args. Other apps adopt the same pattern only if and when they grow deployment surface.

### Enforce in app code

- [ ] **`apps/dashboard/src/dashboard/app.py`** — `app.run(host, port, debug=...)` reads from the loaded `DashboardConfig`. No hardcoded values remain.
- [ ] **`apps/dashboard/src/dashboard/pages/viability_certificate.py:31`** — `_REGISTRY_PATH = Path("results/...")` removed (REQ_123 oversight). Replaced with `family.variant_registry` at the call site.
- [ ] **`apps/dashboard/src/dashboard/pages/initialization_sweep.py`** — `_registry_path()` helper deleted. `_get_canonical_frequencies` reads via `family.variant_registry`.
- [ ] **`apps/dashboard/src/dashboard/components/variant_context_bar.py:66`** — `path = variant.variant_dir / "variant_summary.json"` replaced with `variant.summary`.
- [ ] **`apps/dashboard/src/dashboard/pages/transient_frequency.py:21`** — same replacement.
- [ ] **`apps/dashboard/src/dashboard/pages/analysis_run.py:276`** — comment-string mention is fine; verify no path composition remains.

### Library-side enforcement (storage invariant applies to library code too)

- [ ] **`packages/miscope/src/miscope/views/universal.py`** — 3 sites composing `variant.variant_dir / "variant_summary.json"` (lines 506, 1435, 1473) replaced with `variant.summary`.
- [ ] **`packages/miscope/src/miscope/analysis/saddle_transport.py:358`** — same.
- [ ] **`packages/miscope/src/miscope/analysis/freshness.py`** — same (line 390 reads, line 136 only formats display text and stays).
- [ ] **`packages/miscope/src/miscope/analysis/analyzers/gradient_site.py:193`** — same.
- [ ] **Writers stay path-based.** `variant_summary.py:write_variant_summary` and `variant_analysis_summary.py:_write_summary` / `build_variant_registry` are the API implementations themselves; they own the write side. The accessor implementations (`variant.summary` getter, `family.variant_registry` getter) likewise own the read side. The invariant applies to *callers*, not to the API itself.

### PROJECT.md / CLAUDE.md

- [ ] **Clarify scope of invariant #3.** The current text ("storage layout is internal to the API") is correct but the post-REQ_123 implementation pass treated app code as the only target. Add one line confirming the invariant binds library code too — callers in `miscope.views`, `miscope.analysis`, etc. use the accessors, only the storage primitives themselves compose paths.
- [ ] **No new invariant.** The policy was already captured in REQ_123's third invariant addition.

### Audit / re-grep

- [ ] **Pattern sweep:** `grep -rn 'variant_summary.json\|variant_registry.json' apps/dashboard/src packages/miscope/src` returns only:
  - Docstrings / comments
  - Write-side implementations (`write_variant_summary`, `build_variant_registry`)
  - Accessor read implementations (`variant.summary`, `family.variant_registry`)
- [ ] **Path-literal sweep:** `grep -rn 'Path("results\|Path("model_families' apps/ scripts/ packages/` returns nothing.
- [ ] **Hardcoded server value sweep:** `grep -rn 'host=\|port=' apps/dashboard/src` returns no string-literal hardcoded values outside `apps/dashboard/config.toml`.

### Tests

- [ ] **`tests/test_variant_summary_accessor.py`** (or extend `test_notebook_api.py`) — `variant.summary` returns parsed JSON; raises `FileNotFoundError` cleanly when absent.
- [ ] **`tests/test_families.py`** — extended with `family.variant_registry` returning the parsed list; raising `FileNotFoundError` when absent.
- [ ] **`apps/dashboard/tests/test_dashboard_config.py`** — `load_dashboard_config` returns the expected dataclass; malformed TOML / missing keys raise with informative message; loading from an explicit path works.
- [ ] Existing dashboard tests pass after the enforcement-pass edits (no behavior change, just call-site swaps).

---

## Constraints

**Must:**
- Add accessors (`variant.summary`, `family.variant_registry`) before any caller is refactored. Two-step: accessor + tests; then enforce.
- TOML for the per-app config (stdlib `tomllib`, no new runtime dependency).
- Keep `apps/dashboard/config.toml` tracked so the default deployment values are reproducible across checkouts.
- Use a frozen dataclass for the loaded config (immutable; mirrors `AppConfig`).
- Validate at load time — clear error on malformed file or missing required keys, not at use site.

**May:**
- Other apps can opt into a `config.toml` later; not required if they have no values to put in it.
- The dashboard schema can grow over time (logging, feature flags, etc.). Only `[server]` is needed today.
- Env-var overlay on top of `config.toml` (e.g., `DASHBOARD_PORT=…`) is allowed if convenient, but optional — defer until a deployment scenario actually demands it.

**Must Not:**
- Introduce a runtime dependency on a config library (pydantic, etc.). Stdlib only.
- Move data-root resolution into per-app config. `MISCOPE_DATA_ROOT` env var (and the `cfg.data_root` it produces) remains the canonical mechanism — per-app config is for app deployment values, not library data paths.
- Cache accessor reads inside the property. Callers that need caching assign to a variable, same as `variant.metadata`.
- Retroactively rewrite archived requirement docs to reflect the new accessors.

---

## Notes

- The dashboard's `app.run(...)` call is the only persistent-server entry point in the monorepo today. The "per-app config" framing in REQ_124's original stub generalized prematurely; in practice only dashboard needs this file. The mechanism is generalizable, but we're not seeding empty config files in other apps just to look symmetric.
- The library-side cleanup is genuinely mechanical once the accessors exist. The sites surfaced by the audit (3 in `views/universal`, plus saddle_transport / freshness / gradient_site) are all reads of the same JSON file with the same `variant.variant_dir / "variant_summary.json"` shape.
- `viability_certificate.py:31` is a regression introduced before REQ_123 and not caught by the path-literal sweep — `Path("results/modulo_addition_1layer/variant_registry.json")` pointed at the old layout. REQ_124's enforcement pass replaces it with `family.variant_registry`, fixing the regression as a side effect.
- The accessor naming follows the existing Variant API: `variant.metadata`, `variant.model_config`, `variant.params`, `variant.summary`. Each is a property that reads on access. Family additions: `family.variant_registry` slots alongside `family.variants`, `family.variant_parameters`.
- Adjacent but separate: REQ_125's remaining scope is `ArtifactLoader` privacy + template polish + scripts/ audit. The notebook retrofit was deprioritized — research notebooks are exploratory drafts whose patterns propagate through the notebook template (a small artifact under separate control) and through `ArtifactLoader` privacy (the strong nudge for future me to use `variant.artifacts.*`).

---

## Out of Scope

- Bulk retrofit of research notebooks (REQ_125, deprioritized per follow-up discussion).
- `ArtifactLoader` privacy / non-public marker (REQ_125).
- Multi-environment overrides (`config.staging.toml` etc.) — make capable, don't implement.
- A new architectural invariant in PROJECT.md (the policy is already captured under invariant #3; REQ_124 only clarifies that it binds library code too).
- Restructuring `AppConfig` itself. The data-root config keeps its current shape; the per-app config is a separate object.
- Moving `MISCOPE_DATA_ROOT` resolution into TOML. The library reads it via env var; that contract is fine.
