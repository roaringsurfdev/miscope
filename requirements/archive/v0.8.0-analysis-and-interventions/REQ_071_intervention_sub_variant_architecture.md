# REQ_071: Intervention Sub-Variant Architecture

**Status:** Active
**Priority:** High — blocks running further intervention experiments under a coherent structure
**Related:** REQ_067 (Intervention Family Spec), REQ_068 (Frequency-Selective Attention Tuning)
**Last Updated:** 2026-03-12

## Problem Statement

Interventions are currently implemented as a separate model family (`modadd_intervention`), independent of the baseline variants they modify. This creates a navigational and conceptual mismatch: an intervention is a modification of a specific baseline run, not an independent model family. To find the interventions for p=59/seed=485, a user must switch to a different family rather than inspecting the variant they're studying.

Additionally, intervention variants are currently identified by a truncated config hash that is difficult to read in navigation (e.g., `p59_s485_d598_i3a7f2c10`). Researchers have been memorizing hashes for v1, v2, v3 — which is not sustainable.

The goal is to make interventions first-class citizens of a variant, navigable from the variant they modify, and identifiable by human-readable labels.

## Conditions of Satisfaction

### Filesystem Structure

- [ ] Intervention results live nested under the parent variant directory:
  ```
  results/modulo_addition_1layer/
    p59_s485_d598/
      checkpoints/
      artifacts/
      config.json
      metadata.json
      interventions/
        {intervention_id}/      ← label or hash; see Naming below
          checkpoints/
          artifacts/
          config.json
          metadata.json
  ```
- [ ] Intervention sub-variants share the same file layout as top-level variants (checkpoints, artifacts, config.json, metadata.json)

### Naming

- [ ] Intervention config supports an optional `label` field (e.g., `"label": "v1"`)
- [ ] If `label` is present, it is used as the intervention directory name
- [ ] If `label` is absent, the truncated config hash (8 hex chars) is used as the fallback
- [ ] Labels must be unique within a parent variant's `interventions/` directory (enforced at creation time with a clear error)

### API

- [ ] `Variant` gains an `interventions` property that returns a list of available sub-variants as `InterventionVariant` objects (or equivalent), discovered from the filesystem
- [ ] `Variant` gains `create_intervention_variant(intervention_config, results_dir)` (moving this responsibility from the family) — creates an `InterventionVariant` ready for training
- [ ] `InterventionVariant` exposes the same training, checkpoint loading, and view API as `Variant` (`.train()`, `.load_model_at_checkpoint()`, `.at(epoch).view(...)`)
- [ ] `InterventionVariant` carries a reference to its parent `Variant`

### Family Cleanup

- [ ] `ModAddInterventionFamily` is removed once the sub-variant architecture is in place
- [ ] The `modadd_intervention` family entry is removed from the family registry

### Migration

- [ ] Existing intervention results currently stored in `results/modadd_intervention/` are moved (not re-run) to the appropriate parent variant subdirectory under `results/modulo_addition_1layer/`
- [ ] Moved interventions retain their existing labels (v1, v2, v3) via the `label` field in config.json
- [ ] A new end-to-end test confirms: create intervention variant → train (short) → load checkpoint → view is accessible

### Dashboard

- [ ] The variant selector surfaces available interventions for the selected variant
- [ ] When an intervention is selected, the full view catalog applies to it (same as any other variant)

## Constraints

**Must:**
- Preserve the hash-based unique ID as a stable canonical identifier even when a label is present — the label is a display name, not the identity
- Keep `InterventionVariant` behaviorally identical to `Variant` from the view catalog's perspective — no special-casing in views or renderers

**Must not:**
- Break existing baseline variant discovery or loading
- Require changes to view definitions or renderers

**Decision authority:**
- **Claude decides:** whether `InterventionVariant` is a subclass of `Variant`, a wrapper, or just `Variant` with extra properties; exact protocol for parent reference
- **Claude decides:** error handling when `label` conflicts with existing intervention directory

## Context

The current `ModAddInterventionFamily` is pure delegation — every method calls through to `ModuloAddition1LayerFamily`. Its only function is providing a separate results root and the `create_intervention_variant()` factory. Both of those responsibilities belong more naturally to `Variant` and the filesystem layout, not to a separate family.

The label naming came from how researchers naturally referred to the runs during analysis: "v1 dampened the competition frequencies, v2 added Freq2 boost, v3 boosted Freq15." Hash suffixes were memorized — labels make this explicit and persistent.

## Notes

- The `InterventionVariant` reference to its parent allows the verification page (REQ_070) to load the baseline variant's checkpoint at plateau onset without requiring the user to specify it separately.
- If a future intervention type targets a different family (e.g., a 2-layer model), the same sub-variant structure applies — interventions always nest under their parent, regardless of family type.
