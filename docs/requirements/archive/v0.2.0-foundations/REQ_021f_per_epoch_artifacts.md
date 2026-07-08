# REQ_021f: Per-Epoch Artifact Storage

**Status:** Active
**Priority:** High (blocks end-to-end operation)
**Parent:** [REQ_021](REQ_021_model_families.md)
**Dependencies:** REQ_021b (Analysis Library Architecture)
**Last Updated:** 2026-02-05

## Problem Statement

The analysis pipeline stores all checkpoint epochs in a single `.npz` file per analyzer. For the Modulo Addition 1-Layer family with 95 checkpoints, `neuron_activations.npz` is 1.4GB compressed (~2.5GB uncompressed). This creates four problems:

1. **Memory exhaustion** — Before any analysis work begins, the pipeline loads all existing artifact data into an in-memory buffer (`_init_results_buffers`). For `neuron_activations`, this decompresses 1.4GB into ~2.5GB of RAM with no CPU/GPU activity, presenting as a steady memory climb at the 10% progress mark.

2. **No incremental computation** — To add results for new epochs, the pipeline must load all existing results, merge, and rewrite the entire file. This makes gap-filling expensive and fragile.

3. **No parallelization** — A single file per analyzer is a write bottleneck. Multiple workers cannot safely compute different epochs concurrently.

4. **Visualization over-fetches** — The dashboard loads all 95 epochs into memory to display one. The slider interaction (viewing one checkpoint at a time) only needs a single epoch's data.

## Design Insight

Training checkpoints are already stored as individual files: `checkpoint_epoch_{NNNNN}.safetensors`. Analysis artifacts should follow the same pattern. The access patterns match:
- **Training:** save one checkpoint at a time, load one checkpoint at a time
- **Analysis:** compute one epoch at a time, visualize one epoch at a time
- **Gap-filling:** check which files exist, compute what's missing

## Solution

Store one file per (analyzer, epoch), mirroring the checkpoint pattern:

```
# Before                           # After
artifacts/                          artifacts/
  dominant_frequencies.npz            dominant_frequencies/
  neuron_activations.npz                epoch_00000.npz
  neuron_freq_norm.npz                  epoch_00100.npz
  manifest.json                         ...
                                      neuron_activations/
                                        epoch_00000.npz
                                        ...
                                      neuron_freq_norm/
                                        epoch_00000.npz
                                        ...
                                      manifest.json
```

Each per-epoch `.npz` contains the raw analyzer output for that epoch only. File existence is the source of truth for completion status.

## Scope

This sub-requirement covers:
1. Pipeline storage format change (save per-epoch, remove in-memory buffer)
2. ArtifactLoader support for per-epoch and multi-epoch loading
3. Dashboard adaptation for on-demand per-epoch loading
4. Visualization renderer interface change (accept single-epoch data)
5. Removal of legacy analysis path using `ModuloAdditionSpecification`

## Conditions of Satisfaction

### Pipeline
- [ ] Pipeline saves each analyzer result immediately after computation (no in-memory buffer)
- [ ] Artifacts stored as `artifacts/{analyzer_name}/epoch_{NNNNN}.npz`
- [ ] Pipeline determines completed work from file existence (not manifest)
- [ ] Gap-filling works: interrupted run can resume, computing only missing epochs
- [ ] Explicit GPU memory cleanup between epochs (`del model, cache; torch.cuda.empty_cache()`)
- [ ] Memory usage stays flat during analysis run (no steady climb)

### ArtifactLoader
- [ ] `load_epoch(analyzer_name, epoch)` loads single-epoch data
- [ ] `load_epochs(analyzer_name, epochs=None)` loads and stacks multiple epochs (for cross-epoch views)
- [ ] `get_epochs(analyzer_name)` returns available epochs from filesystem
- [ ] `get_available_analyzers()` discovers analyzers from subdirectories

### Dashboard
- [ ] Variant selection discovers available epochs without loading artifact data
- [ ] Slider interaction loads single-epoch data on demand
- [ ] Legacy `AnalysisPipeline(ModuloAdditionSpecification)` path removed
- [ ] No regression in visualization rendering

### Renderers
- [ ] Per-epoch renderers accept single-epoch data + epoch number (not full artifact + epoch_idx)
- [ ] Cross-epoch renderers unchanged (use `load_epochs()`)

## Constraints

**Must have:**
- Per-epoch file naming matches checkpoint convention (`epoch_{NNNNN}`)
- Atomic file writes (temp + rename) to prevent corruption on interruption
- Manifest.json retained for metadata (shapes, dtypes) but not as source of truth for completion

**Must avoid:**
- Loading all epochs into memory for single-epoch operations
- Breaking cross-epoch visualization capability (notebook use case)

**Flexible:**
- Whether manifest is updated per-epoch or at end of run
- Compression strategy for per-epoch files

## Breaking Changes

- Artifact storage format changes from single-file to per-epoch directory structure
- Existing artifacts must be regenerated by re-running analysis
- Renderer function signatures change (callers must update)

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-05 | Single file vs per-epoch | Per-epoch | Mirrors checkpoint pattern; fixes memory, parallelization, incremental computation |
| 2026-02-05 | Source of truth for completion | File existence | Simpler than manifest; naturally resumable; no corruption risk |
| 2026-02-05 | Migration strategy | Re-run analysis | Toy models are fast to re-analyze; no migration code needed |
| 2026-02-05 | Renderer interface | Single-epoch data + epoch number | Cleaner; removes unnecessary indexing; matches access pattern |

## Notes

**2026-02-05:** Root cause of memory issue identified. `_init_results_buffers()` decompresses all existing artifact data into RAM before analysis begins. For `neuron_activations` with 95 epochs at (512, 113, 113) float32 per epoch, this is ~2.5GB of memory allocated with no CPU/GPU activity — matching the observed behavior of steady memory climb at the 10% progress mark.
