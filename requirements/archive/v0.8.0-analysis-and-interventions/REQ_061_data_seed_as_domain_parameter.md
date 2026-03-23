# REQ_061: Data Seed as Domain Parameter

**Status:** Active
**Priority:** High
**Milestone:** TBD

---

## Problem Statement

Training data composition is a candidate causal factor in frequency competition — which Fourier frequencies "win" during grokking may depend not only on the modulus (prime) and model initialization (seed), but on which (a, b) pairs the model sees during training. Currently, `data_seed` (the random seed controlling the train/test split) is a training-time parameter with a hardcoded default of `598`. It is invisible to the variant identity system: it does not appear in `family.json`, is not encoded in the variant directory name, and cannot be varied through normal variant creation workflows.

To investigate data seed as a causal variable, it must become a first-class domain parameter — part of variant identity, encoded in the filesystem layout, and discoverable by the registry and dashboard.

---

## Conditions of Satisfaction

### 1. `family.json` updated
- `data_seed` is added to `domain_parameters` with `type: "int"`, `default: 598`, and a clear description
- `variant_pattern` is updated to include `data_seed`: `modulo_addition_1layer_p{prime}_seed{seed}_dseed{data_seed}`

### 2. Existing variant directories migrated
- A one-time migration script renames all 17 existing variant directories from `modulo_addition_1layer_p{prime}_seed{seed}` to `modulo_addition_1layer_p{prime}_seed{seed}_dseed598`
- Migration is idempotent: re-running it against already-migrated directories is a no-op
- Migration script lives at `scripts/migrate_dseed.py`

### 3. Registry discovers variants correctly
- `FamilyRegistry.get_variants()` correctly parses the updated pattern and extracts `data_seed` from directory names
- All 17 migrated variants are discovered with `params["data_seed"] == 598`

### 4. `Variant.train()` updated
- `data_seed` is removed as a standalone kwarg with its own default
- `Variant.train()` reads `data_seed` from `self._params["data_seed"]`
- Training config saved in `config.json` continues to record `data_seed`

### 5. Dashboard variant selector handles the new dimension
- The variant selector correctly displays and selects variants that include `data_seed`
- Existing 17 variants (all `dseed=598`) display without regression
- If multiple data seeds exist for the same prime+seed, they appear as distinct selectable variants

### 6. Tests pass
- Registry parsing tests cover the new three-parameter pattern
- Migration script is tested for the rename logic and idempotency
- `Variant.train()` tests reflect the updated parameter sourcing

---

## Constraints

- The analysis dataset (full p×p grid) is **not** affected by `data_seed` — it is used for analysis, not training. No changes to probe generation.
- The migration script must not touch checkpoint or artifact files — only the parent variant directory name.
- `data_seed` controls the train/test split only. It does not affect model architecture, initialization, or analysis logic.
- Existing `config.json` files inside migrated variants already record `data_seed: 598` — no changes needed to those files.

---

## Notes

- **Migration risk is low.** All 17 variants are locally reproducible. The rename only affects the parent directory. Paths inside variant dirs (`checkpoints/`, `artifacts/`) are derived at runtime from the variant name, so they resolve correctly after rename without any file edits.
- **Dashboard impact is expected to be minimal.** The variant selector is driven by pattern discovery. After migration, all 17 variants will appear as before, just with an additional `data_seed` parameter visible. No structural change to the selector is anticipated, but implementation should verify.
- **Stretch / follow-on:** Actually training new variants with non-`598` data seeds is out of scope for this requirement. This requirement establishes the structural foundation; research exploration with new data seeds is the next step.
- **Variant pattern choice:** `_dseed{data_seed}` was chosen to maintain consistency with the existing `_seed{seed}` convention and to be unambiguous (`dseed` = data seed, `seed` = model init seed).
