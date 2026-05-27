# REQ_123: Unified Data Root — `data/{family}/variants/{vid}/`

**Status:** Completed (merged to `develop`). Migration applied; legacy `model_families/` removed; `results/` renamed to `results_bk` as a safety net pending final removal after extended exercise.
**Priority:** Medium — biggest blast-radius change in the ETL forward plan but the lowest *urgency*. The current `model_families/` / `results/` split works; it's just structurally arbitrary and leaks the historical accident of how things were built. Step 4 of 5 in the ETL forward plan.
**Branch:** `feature/req-123-unified-data-root`
**Supersedes:** None. (Implicitly retires the `model_families/` and `results/` top-level directories and the `MISCOPE_RESULTS_DIR` / `MISCOPE_MODEL_FAMILIES_DIR` env-var pair, but neither is a standing requirement.)
**Dependencies:**
- **REQ_122 (LoadedFamily Retirement) — hard dependency.** This REQ anchors data-root resolution on the family object. REQ_122 collapses `LoadedFamily` and makes the family the natural single owner of `(family_dir, results_dir)`. Doing this REQ first would mean rewriting the family API twice.
**Coordinates with:**
- *Atlas + MEMORY updates.* The variant_pattern shortening changes every variant id referenced in [docs/variant_atlas.md](../../variant_atlas.md), [docs/analysis_atlas.md](../../analysis_atlas.md), MEMORY files, fieldnotes posts, sketches comments, and archived requirement docs. The bulk is mechanical (sed-style replace) but the inventory needs to be exhaustive.
- *Regression snapshot machinery* ([scripts/run_regression_check.py](../../../scripts/run_regression_check.py), [REQ_086_regression_snapshot_scaffold.md](../REQ_086_regression_snapshot_scaffold.md)). The regression flow reads from canonical paths; it gets updated alongside the migration.
**Downstream consumers:**
- *Future Store work* (REQ_100 external storage / REQ_101 dataframes). Once `data/` is the only root, swapping it for an external backend is one configuration change.
- *REQ_110 (Lakehouse Surface).* Tabular output sits naturally alongside `data/{family}/variants/{vid}/`.
**Attribution:** Engineering Claude (under user direction). Outcome of the design dialogue captured in `feature/generic_analyzer` [BRANCH_NOTES.md](../../../../BRANCH_NOTES.md). The placement intuition ("family.json belongs *with* the variants") was reconnoitered by the user via the now-deleted duplicate `family.json` under `results/modulo_addition_1layer/`; the REQ adopts that placement as the target layout.

---

## Problem Statement

The repository currently has two top-level directories that are both *instance data for a family*:

- **`model_families/`** — 4 tracked files (3 `family.json` + 1 `ideal_frequency_sets.json`). Describes families.
- **`results/`** — ~30 variants per family × per-variant `{checkpoints/, artifacts/, interventions/, config.json, metadata.json, variant_summary.json}`. Generated, gitignored.

The split is structural noise. Both trees are scoped to a family; both are indexed by family name; the only thing distinguishing them is the word "results." `family.json` describing the family lives in one tree, but `variant_summary.json` describing instances of that family lives in another — even though both are *data about* the family.

Three secondary consequences make the split more annoying than it would otherwise be:

1. **The `variant_pattern` is forced to include the family name.** Today: `"modulo_addition_1layer_p{prime}_seed{seed}_dseed{data_seed}"`. The `modulo_addition_1layer_` prefix exists because variants live in a *flat* `results/` tree alongside other families' variants. Once they're nested under `data/{family}/variants/`, the prefix becomes dead weight.
2. **`FamilyRegistry` (now `FamilyDiscovery` after REQ_122) carries *two* directory roots** — `model_families_dir` *and* `results_dir`. Wherever the family is constructed, callers have to pass both. Under a unified root, there's one.
3. **`MISCOPE_RESULTS_DIR` and `MISCOPE_MODEL_FAMILIES_DIR` are two env vars expressing one thing.** "Where does the data live?" — two answers, one for each tree, plus `MISCOPE_PROJECT_ROOT` to anchor both.

The motivation is *not* a bug. Nothing is broken; the split just doesn't earn its existence.

### What this REQ does

Collapses both trees into a single `data/` root:

```
data/
  modulo_addition_1layer/
    family.json
    ideal_frequency_sets.json
    variants/
      p113_seed999_dseed598/
        checkpoints/
        artifacts/
        interventions/
        config.json
        metadata.json
        variant_summary.json
      p109_seed485_dseed598/
        ...
    variant_registry.json
  modulo_addition_2layer_mlp/
    family.json
    variants/
      ...
```

Single-PR hard cutover. `variant_pattern` shortens to drop the family prefix. New `MISCOPE_DATA_DIR` env var replaces the two old ones (which become deprecated aliases for one release). A migration script handles the variant-directory renames. Atlas and MEMORY references to old-form variant IDs are grep-and-updated.

This is a **mechanical path migration** with one API-shape change (family takes one root instead of two). No analyzer changes. No view changes. No dashboard logic changes beyond path strings.

---

## Conditions of Satisfaction

### On-disk layout

- [ ] **`data/{family}/family.json`** — tracked in git (currently the only tracked content under `model_families/`).
- [ ] **`data/{family}/ideal_frequency_sets.json`** — tracked alongside `family.json` (currently exists only for `modulo_addition_1layer`).
- [ ] **`data/{family}/variants/{short_vid}/{checkpoints,artifacts,interventions,config.json,metadata.json,variant_summary.json}`** — generated, gitignored.
- [ ] **`data/{family}/variant_registry.json`** — compiled aggregate, gitignored. Stays at family root, not under `variants/`, because it's an index *of* variants, not a variant itself.
- [ ] **`.gitignore`** — `data/` body gitignored with whitelists: `data/**/family.json`, `data/**/ideal_frequency_sets.json` (and similar tracked-config patterns) tracked; everything else under `data/` (variant directories, registries, generated artifacts) ignored.

### Variant naming (`variant_pattern` shortens)

- [ ] **`family.json::variant_pattern` drops family prefix** for all three current families:
  - `modulo_addition_1layer_p{prime}_seed{seed}_dseed{data_seed}` → `p{prime}_seed{seed}_dseed{data_seed}`
  - `modulo_addition_2layer_mlp_p{prime}_seed{seed}_dseed{data_seed}` → `p{prime}_seed{seed}_dseed{data_seed}`
  - `modulo_addition_learned_emb_mlp_p{prime}_seed{seed}_dseed{data_seed}` → `p{prime}_seed{seed}_dseed{data_seed}`
- [ ] **Variant directories renamed on disk** by the migration script (see below).
- [ ] **`Variant.name` semantics preserved** — still derived from `family.variant_pattern.format(**params)`, just now without the prefix. Code that constructs paths from `variant.name` continues to work; only the resulting strings shorten.

### Family API anchors on one data root

- [ ] **`ModelFamily.__init__` takes `data_root` (single argument) instead of `(model_families_dir, results_dir)`.** Family's own directory is `data_root / family.name`; variants live under `data_root / family.name / "variants"`.
- [ ] **`Variant.__init__` takes the family object** (already does) and derives its directory from `family.variants_dir / self.name`. The standalone `results_dir` parameter is removed.
- [ ] **`FamilyDiscovery.list_family_dirs(data_root)`** — scans `data_root` directly (no separate model_families_dir). Pattern: `data/*/family.json` exists ⇒ family.
- [ ] **`FamilyDiscovery.load_family_from_dir(family_dir, data_root)`** — reads `family.json`, instantiates the class, passes `data_root`.
- [ ] **Family-implementation `from_json` helpers** ([`modulo_addition_1layer.py:390`](../../../packages/miscope/src/miscope/families/implementations/modulo_addition_1layer.py#L390), `modulo_addition_2l_mlp.py:291`, etc.) — updated to read from `data_root / family_name / family.json` instead of `model_families_dir / family_name / family.json`. The `model_families_dir` parameter renames to `data_root` (or is dropped if it can be derived from the cfg).

### Configuration

- [ ] **`AppConfig` collapses two fields into one.** `AppConfig.results_dir` and `AppConfig.model_families_dir` removed; `AppConfig.data_root` added.
- [ ] **`get_config()` env-var resolution:**
  - Primary: `MISCOPE_DATA_ROOT` (or `MISCOPE_DATA_DIR` — name TBD at implementation time; pick one and stick with it).
  - Deprecated: `MISCOPE_RESULTS_DIR` + `MISCOPE_MODEL_FAMILIES_DIR` — if either is set without `MISCOPE_DATA_ROOT`, emit a `DeprecationWarning` and fall back to a derived root (use the parent dir if they happen to share one; otherwise raise with a clear message pointing at `MISCOPE_DATA_ROOT`).
  - Legacy TDW_* aliases — `TDW_DATA_ROOT` added; `TDW_RESULTS_DIR` / `TDW_MODEL_FAMILIES_DIR` kept on the same deprecation timeline.
- [ ] **Default resolution** — if no env var is set, `data_root = project_root / "data"`.
- [ ] **Documentation in `config.py` docstring** updates to name `data_root` as canonical and mark the old pair as deprecated.

### Migration script

- [ ] **`scripts/migrate_to_data_root.py`** — one-shot script that:
  1. Creates `data/{family}/` for each family discovered under `model_families/`.
  2. Copies `model_families/{family}/family.json` and `ideal_frequency_sets.json` into `data/{family}/`.
  3. Creates `data/{family}/variants/`.
  4. For each variant directory under `results/{family}/`: renames `{family_prefix}_{rest}` → `{rest}` and moves it to `data/{family}/variants/{rest}/`.
  5. Moves `results/{family}/variant_registry.json` (if present) to `data/{family}/variant_registry.json`.
  6. Reports unmoved files in `results/{family}/` (e.g., the leftover duplicate `family.json` / `family_config.json` from REQ_122 should be gone by then; anything else surfaces a warning).
  7. Updates the three `family.json::variant_pattern` fields in-place (or in the new copy) to the shortened form.
- [ ] **The script is idempotent** — running it twice produces no change after the first run; running it against partial state (some variants migrated, some not) resumes cleanly.
- [ ] **The script does not delete `model_families/` or `results/`.** That's a final manual step after the user verifies the migration. The script reports what would be deleted.

### Codebase path-rewrites

- [ ] **`packages/miscope/src/miscope/config.py`** — see Configuration above.
- [ ] **`packages/miscope/src/miscope/families/__init__.py:16-17`** — default arguments update from `model_families_dir`/`results_dir` to `data_root`.
- [ ] **`packages/miscope/src/miscope/families/variant.py`** — `results_dir` parameter removed; path derivation goes through `family.variants_dir`.
- [ ] **`packages/miscope/src/miscope/families/intervention_variant.py:104`** — `results_dir=parent._results_dir` replaced with family-anchored path resolution.
- [ ] **`packages/miscope/src/miscope/families/implementations/*.py`** — `model_families_dir` parameter updates to `data_root` in all `from_json` helpers.
- [ ] **`apps/dashboard/src/dashboard/state.py:85-96`** — hardcoded `Path("model_families")` / `Path("results")` replaced with `cfg.data_root` reference.
- [ ] **`apps/dashboard/src/dashboard/pages/analysis_run.py:173`** — `variant.variant_dir.parent.parent` (currently the family results directory) updated to the equivalent expression under the new layout.
- [ ] **`apps/dashboard/src/dashboard/pages/initialization_sweep.py:44`** — hardcoded `Path("results") / "modulo_addition_1layer" / "variant_registry.json"` updated.
- [ ] **`scripts/run_analysis.py`, `run_analysis_regression.py`, `migrate_dseed.py`, `refresh_variant_summaries.py`, `viability_certificate_calibration.py:39`, `precompute_ideal_sets.py`, `train_2layer_mlp.py`, `train_learned_emb_mlp.py`, `export_fieldnotes_figures.py`, `generate_regression_checksums.py`, `run_regression_check.py`** — all updated to use `cfg.data_root`.
- [ ] **All path-construction code that does `results_dir / family.name` or `model_families_dir / family.name`** is replaced with `data_root / family.name` (single concatenation). The semantic equivalence is checked: every old `results_dir / family.name / variant.name` is now `data_root / family.name / "variants" / variant.name`.

### Documentation & memory updates

- [ ] **`docs/variant_atlas.md`** — references like `results/modulo_addition_1layer/{variant_dir}/variant_summary.json` updated to `data/modulo_addition_1layer/variants/{variant_dir}/variant_summary.json`. Variant IDs in the atlas table (e.g., `modulo_addition_1layer_p113_seed999_dseed598`) shortened to `p113_seed999_dseed598`.
- [ ] **`docs/analysis_atlas.md`** — same.
- [ ] **MEMORY files** (under user's `~/.claude/projects/.../memory/`) — any references to old-form paths or variant IDs grep-updated. These live outside the repo; the REQ surfaces an item to update them manually with the exact list of strings to replace.
- [ ] **`README.md`, `CLAUDE.md`, `CHANGELOG.md`** — path references in onboarding text updated.
- [ ] **`apps/fieldnotes/src/content/posts/*.mdx`** — variant ID references in published posts updated. The shortened form is more readable in prose; the rename is a bonus.
- [ ] **Archived `docs/requirements/archive/**.md` files are NOT retroactively updated.** Archived requirements freeze the historical state at release time; rewriting them would break archaeology. Flag this in CHANGELOG.

### Tests / validation

- [ ] **Migration parity test:** run the migration script against a copy of the live `results/` tree; verify the resulting `data/` tree contains every variant directory the source had, with shortened names, and that every file under each variant is byte-identical.
- [ ] **End-to-end smoke** — after migration: `load_family("modulo_addition_1layer").get_variant(prime=113, seed=999, data_seed=598).artifacts.load_epoch(...)` produces the same result as pre-migration.
- [ ] **Variant pattern roundtrip:** `family.variant_pattern.format(**variant.params) == variant.name` for every variant in every family. (Catches the case where the pattern shortens but a variant directory's name was missed by the migration.)
- [ ] **No remaining references to `results_dir` or `model_families_dir`** in non-archived code. `grep -rn 'results_dir\|model_families_dir' packages/ apps/ scripts/` returns nothing.
- [ ] **`MISCOPE_DATA_ROOT` honored end-to-end:** set the env var to a non-default location, run a small analysis, verify artifacts land under the override.
- [ ] **Deprecation warning surfaces** when `MISCOPE_RESULTS_DIR` or `MISCOPE_MODEL_FAMILIES_DIR` is set without the new var.

---

## Constraints

**Must:**
- Single-PR hard cutover. No two-phase coexistence period. The migration script is committed, the path code is updated, the directories are moved, all in one PR.
- REQ_122 lands first. This REQ assumes the family object owns variant lookup; the diff would be much messier without that substrate.
- Migration script is idempotent and reports rather than auto-deletes the old `model_families/` and `results/` trees. The user verifies before removing the originals.
- Old env vars (`MISCOPE_RESULTS_DIR`, `MISCOPE_MODEL_FAMILIES_DIR`, `TDW_RESULTS_DIR`, `TDW_MODEL_FAMILIES_DIR`) emit a `DeprecationWarning` for one release before they're hard-removed. The deprecation removal is a follow-up; this REQ keeps the deprecation alive.
- All three family implementations migrate together. No staged "1layer first, MLP families later" — the path code paths through one config function, splitting it would be more work than just doing all three.
- `.gitignore` patterns carefully scoped so newly-tracked `family.json` files don't accidentally pull in generated artifacts. Explicit whitelists, not negative patterns.

**May:**
- Naming of the env var (`MISCOPE_DATA_ROOT` vs `MISCOPE_DATA_DIR`) is open. Pick one and document it.
- Migration script can be a one-shot deletable after the migration, or kept in `scripts/` as a record. Implementer's call.
- `variant_pattern` shortened form can differ from the proposal (`p{prime}_seed{seed}_dseed{data_seed}`) if there's a reason to prefer something else (e.g., `p{prime}_s{seed}_d{data_seed}`). The constraint is "drop the family prefix"; the form of what remains is open.

**Must Not:**
- Introduce a `data/` resolver that falls back to `model_families/` / `results/`. Either we migrate, or we don't. No coexistence.
- Modify analyzer code, view code, or visualization code. The path-string changes thread through `cfg`, `family`, and `variant` — they don't reach the analyzer layer.
- Touch the `Variant.checkpoints_dir` / `Variant.artifacts_dir` / `Variant.metadata_path` / etc. property names. Only their *resolved* paths change.
- Retroactively rewrite archived requirement docs. Archive integrity matters more than path-consistency.
- Auto-delete `model_families/` or `results/`. The migration script reports; the user removes.

---

## Notes

- The user added duplicate `family.json` files under `results/{family}/` "while exploring refactoring … to get a feel for where the files might rightfully belong." This REQ adopts that placement (family.json lives next to its variants under `data/{family}/family.json`). The reconnaissance files themselves are deleted by REQ_122; only the layout decision survives.
- The "biggest blast radius" framing in BRANCH_NOTES is accurate but not as bad as it sounds. The actual mechanical surface is ~38 `results_dir`/`model_families_dir` assignment/usage sites in non-test code; the rest is documentation and string replacement.
- **Why hard cutover over two-phase:** a coexistence period doubles the path-resolution surface for an indefinite duration (the second phase always slips). Pre-1.0, internal-only project — a single-PR migration is cheaper than maintaining a fallback resolver that has to be reasoned about every time someone touches config.
- **The hard dependency on REQ_122:** under today's API, the family takes `(model_families_dir, results_dir)` from `FamilyRegistry`. Collapsing to one argument requires the family to be the owner of that argument, which is what REQ_122 establishes. Doing REQ_123 first would mean introducing a single-data-root family that still has to pretend it has two roots until REQ_122 lands — i.e., rewriting the family API twice.
- **Why `data/` is also gitignored** like `results/` is today: variants are generated, large, and ephemeral. Only the tracked-config files (`family.json`, `ideal_frequency_sets.json`) need whitelisting. Same tracking policy as today, just under a different root.
- **Archive immutability** is worth being explicit about. Archived requirements (e.g., REQ_021c, REQ_086, REQ_088) contain path references that will become stale relative to current code. That's *fine* — they describe the state at the time they shipped. Rewriting them to track current paths would erase historical context.
- **Atlas updates are mechanical but not trivial.** `variant_atlas.md` and `analysis_atlas.md` reference variants by full ID in dozens of places; a careful sed with manual review is needed. The MEMORY files under the user's `~/.claude/...` aren't in the repo, so the REQ surfaces them as an explicit external update item.
- **Fieldnotes posts** (under `apps/fieldnotes/src/content/posts/`) are published research notes. Variant ID references in those will benefit from the shortened form — easier to read in prose — so the migration is a small UX win there.
- **Downstream Store work** (REQ_100, REQ_101): once `data_root` is a single path, the Store abstraction's first job is "swap this for an external backend." That's a much cleaner seam than "swap *this pair* for an external backend." This REQ is doing groundwork for that abstraction without committing to it.

---

## Out of Scope

These belong elsewhere and explicitly do not land here:

- **`AbstractStore` abstraction.** Triggered by REQ_100 / REQ_101, not by this REQ.
- **Retiring the deprecated env-var aliases.** Their `DeprecationWarning` lifecycle is multi-release; the hard removal is a later, trivial PR.
- **Migrating archived requirement docs to new paths.** Archive immutability — keep historical state intact.
- **Renaming Variant API properties** (`checkpoints_dir`, `artifacts_dir`, etc.). Only their resolved paths change; the property names stay.
- **Cross-family aggregation surfaces.** A future REQ might want `data/cross_family/` or similar for cross-family analysis outputs; this REQ stays inside the per-family scope.
- **Re-tracking large data in git.** The current ignore policy (tracked configs, ignored artifacts) is preserved exactly under the new root.
