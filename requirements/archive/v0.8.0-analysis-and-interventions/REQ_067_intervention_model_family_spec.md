# REQ_067: Intervention Model Family Spec (POC)

**Status:** Active
**Priority:** High — prerequisite for REQ_068
**Related:** REQ_068 (Frequency-Selective Attention Tuning)
**Last Updated:** 2026-03-11

## Problem Statement

Intervention experiments require running modified training passes against existing model families (e.g., Modulo Addition 1-Layer). Without a proper family spec, there is no structural guarantee that intervention training runs are isolated from baseline artifacts. A bug in an intervention run must never be able to overwrite or contaminate a baseline variant's results. There is also no established way to identify, reproduce, or compare intervention experiments across runs — the intervention parameters need to be part of the variant's identity, not a post-hoc annotation.

The goal of this requirement is to establish the family registration, storage structure, and config schema for intervention variants as a POC — before any actual intervention mechanism is implemented. The hook implementation (REQ_068) should slot into a structure that already exists and already has clear identity semantics.

## Conditions of Satisfaction

### Family Registration

- [ ] A new family `ModAddIntervention` is registered in the family registry as a distinct family — not a subclass or modification of the existing `ModuloAdditionFamily`
- [ ] The family is discoverable through the existing `FamilyRegistry` without changes to its interface
- [ ] The family produces valid probe, context, and Fourier basis — identical to the baseline family for the same prime — so that existing analyzers and views work without modification

### Storage Isolation

- [ ] Intervention variants are stored under a results directory that is structurally distinct from the baseline results directory (e.g., `results/modulo_addition_1layer_intervention/` vs `results/modulo_addition_1layer/`)
- [ ] No code path in the intervention family reads from or writes to the baseline results directory
- [ ] A dry-run of an intervention variant confirms that the baseline results directory is not touched

### Variant Identity

- [ ] Intervention variant config schema includes all original domain parameters: `prime`, `seed`, `data_seed`
- [ ] Intervention variant config schema includes an `intervention` block that captures:
  - `type`: identifier for the intervention kind (e.g., `"frequency_gain"`)
  - `target_heads`: list of head indices, or `"all"`
  - `target_frequencies`: list of frequency indices
  - `gain`: per-frequency scalar (or dict mapping frequency → scalar)
  - `epoch_start`: training epoch at which the intervention window opens
  - `epoch_end`: training epoch at which the intervention window closes
  - `ramp_epochs`: number of epochs over which the gain ramps from 0 to target at window start (and from target back to 0 at window end)
- [ ] Variant names use a short intervention ID rather than encoding full config params in the filename — e.g., `p113_seed999_dseed598_iv_a3f2` — keeping filenames tractable regardless of intervention complexity
- [ ] The intervention ID is stable: the same intervention config always produces the same ID (deterministic, not random)
- [ ] An intervention variant's config can be fully reconstructed from its stored config.json alone — no external state required to reproduce the run

### POC Validation

- [ ] A single intervention variant can be constructed with a real prime/seed/data_seed from the existing modulo addition dataset and a placeholder intervention config (e.g., `gain = 1.0` on all frequencies, which is a no-op)
- [ ] That variant passes the same artifact loading and view rendering checks that baseline variants pass — i.e., `ArtifactLoader`, `EpochContext`, and registered views work against it
- [ ] The variant's results directory is confirmed to be isolated from `results/modulo_addition_1layer/`

## Constraints

**Must:**
- Baseline family and its results are untouchable — no shared write paths
- Config schema is extensible: adding a new intervention type in the future must not require breaking changes to existing intervention variant configs
- The intervention family must satisfy the same family protocol as existing families so analyzers and views remain universal

**Must not:**
- Modify `ModuloAdditionFamily` or its variant class
- Share a results directory root with any baseline family
- Require changes to `FamilyRegistry`, `ArtifactLoader`, `ViewCatalog`, or the analysis pipeline

**Decision authority:**
- **Claude decides:** Exact directory naming for intervention results, whether intervention config is a nested block or flat fields in config.json, intervention ID generation scheme (hash of config dict is a reasonable approach)
- **Resolved:** No `InterventionVariant` subclass. `Variant.train()` gains a single optional parameter `training_hook: Callable[[int], list[tuple[str, Callable]]] | None = None`. The hook takes the current epoch and returns a list of transformer_lens hook tuples (or `[]` outside the intervention window). The training loop switches to `model.run_with_hooks(train_data, fwd_hooks=hooks)` only when the hook returns a non-empty list; otherwise the existing `model(train_data)` path is used unchanged. `InterventionFamily.create_variant()` returns a standard `Variant`; the hook callable is passed at `variant.train(training_hook=...)` call time. Baseline behavior: completely unchanged (hook=None).

## Context

The immediate motivation is REQ_068: a frequency-selective intervention on attention head outputs during the training plateau. The Thrasher variant (p=59, seed=485) is the first target. Future intervention types (weight surgery, activation clamping, curriculum changes) can be added as new `type` values within `ModAddIntervention` without a new family — the intervention type field in the config schema handles that variation.

The decision to keep intervention training entirely separate from baseline training, rather than extending baseline variants with optional intervention flags, is intentional. Intervention runs change the training trajectory — they are not the same experiment as the baseline, even if they share domain parameters. Keeping them in a separate family makes this distinction unambiguous in the artifact record.

## Notes

- The `ramp_epochs` field in the intervention spec is a forward-looking inclusion — the actual ramping behavior is implemented in REQ_068. Including it in the schema now means REQ_068 has a field to read from without a schema migration.
- The "no-op intervention" (gain = 1.0 everywhere) as a POC validation is useful beyond just testing infrastructure: it establishes that an intervention variant trained with a no-op produces results statistically indistinguishable from its baseline counterpart, which is a sanity check on the hook implementation later.
- Other model families get their own intervention family when needed — `ModuloAddition2LIntervention`, `SparseParityIntervention`, etc. `ModAddIntervention` is scoped to Modulo Addition 1-Layer only.
