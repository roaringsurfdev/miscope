# REQ_129: v1.0.0 Dead-Code Cull (Scaffolding Retirement)

**Status:** Completed — merged to `develop` 2026-05-30 (merge `dbab5b9`, `--no-ff`). Net −270 lines; miscope 1466 passed / 29 skipped, dashboard 45 passed; user smoke-test (re-analysis with new checkpoints + dashboard views) clean. Two API-surface follow-ups deferred to REQ_132.
**Priority:** Medium — close-out track for v1.0.0. Not blocking; pairs with REQ_103.
**Branch:** `feature/req-129-v1-dead-code-cull`
**Dependencies:**
- REQ_102 (Analyzer Deprecation — retires superseded *analyzers*; this REQ retires superseded *scaffolding*).
- REQ_128 (Analyzer Input Provisioning — removes dead code **on the input path**; this REQ catches the rest).

**Relationships:**
- REQ_103 (PyPI Publication Hardening). The publishable library should not ship dead coexistence scaffolding. This REQ feeds REQ_103's release notes.

**Attribution:** Engineering Claude (stubbed 2026-05-28 from findings surfaced during the REQ_128 design session).

---

## Motivation

The work approaching v1.0.0 (first PyPI release) is the forcing function. The user has confirmed **no external back-compat needs** and **tolerance for system instability** while converging on a cleaner base. Several REQ-era coexistence layers (REQ_120/121 spec/IO unification, the ETL/planner introduction) left scaffolding that is now dead or near-dead. Carrying it into the published library raises maintenance burden and confuses external readers about which paths are canonical.

This REQ is the **ruthless cull** track the user flagged — deliberately separated from REQ_128 so that REQ_128 stays atomic ("the input contract changed") and this stays atomic ("dead scaffolding removed"). REQ_128 removes only the dead code co-located on the input-provisioning path; everything else lands here.

## Findings parked (from the 2026-05-28 REQ_128 audit)

These are *candidates*, not confirmed scope — each needs a use-check before removal (the REQ_102 "audit before deleting" discipline applies):

- **Legacy `category`-authoring scaffolding — now the WHOLE coexistence layer (confirmed scope, 2026-05-28).** REQ_128 implementation found this is *not* input-path-isolated, so the **entire** cull lands here, not split. The coupled unit:
  - `AnalyzerSpec` authored fields `category` / `requires` / `requires_model_weights` / `requires_activation_cache`; the `is_unified` discriminator; the `if self.is_unified:` branches inside the `effective_*` properties (collapse to pure derivation).
  - The legacy **registry synthesis**: `AnalyzerRegistry.register` / `register_secondary` / `register_cross_epoch` → `_legacy_register` (`registry.py`), which is the only thing that *produces* `category`-authored Specs. **No production caller** — all analyzers use `@register_analyzer`; only `test_spec_registry.py` and `test_secondary_analyzers.py` exercise these. Retiring them is what makes the `effective_*`/`is_unified` branches truly dead.
  - The pipeline's `if not spec.inputs` branch in `_materialize_per_epoch_inputs` (and `extra_allowed` for spec-less `depends_on` secondaries) — both exist only for legacy/spec-less analyzers; removable once the legacy registry path is gone.
  - Test impact: rewrite/remove the legacy-authoring + legacy-register tests in `test_spec_registry.py` (~577 lines, many `category=`/`requires=` fixtures) and the `TestSecondaryAnalyzerRegistry` block in `test_secondary_analyzers.py`.
  - Planner simplification (`planner.py` `effective_category` reads, the descriptor `category` field, primary/secondary/cross-epoch classification) follows once authoring is gone. Confirm nothing external classifies by authored `category`.
  - Also retire the `effective_` prefix once authored-vs-derived is no longer a distinction (rename `spec.effective_*` → `spec.*` across registry/planner/freshness/tests).
- **REQ_121 Phase 2C three-protocol dispatcher.** `spec.py` documents a legacy three-protocol dispatch path kept "until Phase 2C retires it." If no legacy-style Specs remain (they don't), the dispatcher is dead.
- **General v1.0.0 surface review.** A dedicated pass over `packages/miscope/` for re-export shims, deprecated aliases, and `# removed`/back-compat comments left by prior REQs (e.g. the `modadd_intervention.py` deprecated re-export shim noted in project memory; the `TDW_*` legacy env-var aliases).

## Audit results (2026-05-30)

Each parked candidate, resolved against a repo-wide consumer trace:

| Candidate | Verdict | Notes |
|---|---|---|
| Legacy `category`-authoring fields + `AnalyzerRegistry.register`/`register_secondary`/`register_cross_epoch`/`_legacy_register` | **Dead — remove.** | No production caller. Only `test_spec_registry.py` + `test_secondary_analyzers.py` exercise them. All 32 analyzers use `@register_analyzer`. **Trap:** `AnalysisPipeline.register*` (pipeline.py:79/91/110) shares the names but is **live** (`scripts/run_regression_check.py`) — out of scope, do not touch. |
| `is_unified` discriminator + `effective_*` branch collapse | **Collapse + rename.** | With authored `category` gone, `is_unified` is always true, so `effective_category/requires/requires_model_weights/requires_activation_cache` reduce to their `derive_*` bodies (`inputs.py:126/135/140/145`). Rename `effective_X` → `X` (now plain derived properties). Live readers to migrate: `registry.py` (11 sites), `planner.py` (lines ~350–363), `freshness.py` (lines 264/279), + tests. |
| Pipeline `if not spec.inputs` / `extra_allowed` spec-less branches | **Remove** once legacy registry path is gone. | Existed only for legacy/spec-less analyzers. |
| Three-protocol dispatcher | **Already collapsed** (REQ_121 Phase 2C). | Residue only: stale "until Phase 2C retires it" docstrings in `spec.py` (lines ~20–21), and the pure `= Analyzer` aliases `UnifiedAnalyzer`/`SecondaryAnalyzer`/`CrossEpochAnalyzer` (`protocols.py:110–112`). Remove the aliases + their `__init__.py` re-exports; retype `registry.py` `get_secondary`/`get_cross_epoch` returns to `Analyzer`; update test isinstance checks to `Analyzer`. Wider test churn — bounded but touches ~8 test files. |
| `modadd_intervention.py` re-export shim | **Already gone** — file does not exist. | Project memory was stale; dropped from scope. |
| `TDW_*` env-var aliases (`config.py`) | **Remove (user decision 2026-05-30).** | `MISCOPE_*` becomes the sole env contract. Removes the `or os.environ.get("TDW_*")` fallbacks and the `TDW_*` doc/`__all__` references. |

## Conditions of Satisfaction

### Authoring + registry cull
- [x] `AnalyzerSpec` authored fields `category` / `requires` / `requires_model_weights` / `requires_activation_cache` removed; `is_unified` removed.
- [x] `effective_*` properties collapsed to plain derived properties and renamed `effective_X` → `X` across `spec.py`, `registry.py`, `planner.py`, `freshness.py`, and tests.
- [x] `AnalyzerRegistry.register` / `register_secondary` / `register_cross_epoch` / `_legacy_register` removed. `AnalysisPipeline.register*` untouched.
- [x] Pipeline `if not spec.inputs` branch in `_materialize_per_epoch_inputs` collapsed (derived flags already OR over all `ModelInput`s — behavior-preserving). *See follow-up §1 re: the remaining `spec is None` / `extra_allowed` spec-less support.*
- [x] `test_spec_registry.py` and `TestSecondaryAnalyzerRegistry` (in `test_secondary_analyzers.py`) rewritten/removed to drop legacy `category=`/`requires=` fixtures.

### Protocol-alias cull
- [x] `UnifiedAnalyzer` / `SecondaryAnalyzer` / `CrossEpochAnalyzer` aliases removed from `protocols.py` and `analysis/__init__.py`; consumers retyped/isinstance-checked against `Analyzer`.
- [x] Stale "Phase 2C" dispatcher docstrings in `spec.py` corrected.

### Env-contract cull
- [x] `TDW_*` fallbacks and references removed from `config.py`; `MISCOPE_*` is the sole contract.

### Validation
- [x] Full `miscope` (1466 passed, 29 skipped) + `dashboard` (45 passed) test suites pass.
- [x] `grep` confirms zero remaining references to each removed symbol (`is_unified`, `effective_category`, `_legacy_register`, `TDW_`, the protocol aliases) outside this REQ doc, CHANGELOG, and the `test_config` test that asserts `TDW_*` is *ignored*.
- [x] No new dead-code introduced (no commented-out shims left behind — delete, don't comment).

## Follow-ups deferred (flagged during implementation)

These are *test-only / dead-in-production* surfaces deliberately left in place to keep this REQ a dead-**code** cull rather than an API-surface decision. **Tracked in REQ_132** (both share the pre-unification category-phased model as their root):

1. **Pipeline spec-less analyzer support.** The `spec is None` conservative branch in `_materialize_per_epoch_inputs` and the `extra_allowed`/`depends_on` widening in `_run_secondary_analyzers` only fire for analyzers with no registered Spec — which never happens in production (all 32 register via `@register_analyzer`), only for ad-hoc test analyzers. Removing it is coupled to retiring the pipeline's separate `_secondary_analyzers` execution phase and the `depends_on` attribute convention — a structural change, not a cull.
2. **Legacy class-based query API.** `AnalyzerRegistry.get` / `get_secondary` / `get_cross_epoch` / `get_for_family` / `get_secondary_for_family` / `get_cross_epoch_for_family` / `list_all` are now test-only (and three are fully dead). Removing them changes `AnalyzerRegistry`'s public surface (the spec-based vs. legacy split), which deserves an explicit decision rather than a ride-along here. They are retyped to `Analyzer` and still functional.

## Notes

- **Audit before deleting.** Mirror REQ_102's safeguard: grep for each candidate, confirm no consumer (view, notebook, downstream analyzer, app), confirm family configs and `__init__`/`registry` don't reference it, then remove. *(Audit table above, 2026-05-30.)*
- **Sequencing.** Best run *after* REQ_102 and REQ_128 so the remaining dead set is well-defined and not a moving target. *(Both landed on `develop` 2026-05-29; this REQ executed 2026-05-30.)*
