# REQ_131: Migrate `neuron_freq_norm` consumers to `activation_basis_projection` (unblock `neuron_freq_clusters` retirement)

**Status:** Stub — *problem statement + captured learnings; CoS to be developed.*
**Priority:** Medium — unblocks the second REQ_102 deferral.
**Branch:** TBD
**Dependencies:**
- REQ_102 (Analyzer Deprecation — deferred the `neuron_freq_clusters` retirement because of these consumers).
- **REQ_128 (Analyzer Input Provisioning — sequence *after*).** The re-point loads the granular `activation_basis_projection` cube; without REQ_128's selective/lazy loading the backbone consumers would re-import whole-artifact memory pressure (`load_epoch` has no `fields=` today).
- REQ_126 (provides `activation_basis_projection` and the bitwise-validated reconstruction).

**Attribution:** Engineering Claude (stubbed 2026-05-28, from the REQ_102 analyzer-dependency audit).

---

## Problem

`neuron_freq_clusters` registers the artifact **`neuron_freq_norm`**, which is a live declared dependency of three surviving analyzers: `neuron_dynamics`, `freq_group_weight_geometry`, `neuron_group_pca`. Retiring `neuron_freq_clusters` requires re-pointing those consumers to the generic successor `activation_basis_projection` (the original was specialized rather than generic — the `freq_cluster` naming reflects that). Per user, `neuron_freq_norm` is a **backbone of critical analysis**; do not change it until the migration is clearly understood.

## Captured learnings (from the 2026-05-28 audit — the migration is tractable)

- **Uniform, narrow contract.** All three consumers use a *single* field, `norm_matrix` (shape `(n_freq, d_mlp)` per epoch; stacked `(n_epochs, n_freq, d_mlp)`), and use it identically: `argmax` / `max` over the frequency axis for group assignment and dominant-frequency tracking. Nothing else is read.
- **Reconstruction already exists and is bitwise-validated.** `_reconstruct_legacy_neuron_freq_norm(result, prime, "mlp_out")` (in `test_activation_basis_projection.py`) rebuilds `norm_matrix` from `activation_basis_projection`'s `mlp_out` site at parity tolerance. Promote it to a shared library/adapter helper.
- **Behavior-preserving.** Because the reconstructed `norm_matrix` is bitwise-identical, the consumers' outputs are unchanged — this defuses the "backbone, don't touch" risk; it is a safe re-point, not a behavioral change.
- **The only real complication is loading**, hence the REQ_128 dependency: consumers currently `load_epoch("neuron_freq_norm")` (small); re-pointing to the granular `activation_basis_projection` artifact wants `fields=`/lazy loading to stay cheap.

## Conditions of Satisfaction

*(Deferred — stub. Develop after REQ_128: promote the reconstruction helper; re-point the three consumers' `ArtifactInput`/loads to `activation_basis_projection` (selective fields); verify their outputs are unchanged on canon; then hand the `neuron_freq_clusters` retirement back to REQ_102.)*

### Regression scenario surfaced during REQ_128 (must be a CoS test)

**Scenario.** A variant has *N* checkpoints, but a previous analysis (run when `neuron_freq_clusters` was still in the family) left only *M < N* `neuron_freq_norm` epoch artifacts on disk. The family no longer includes `neuron_freq_clusters` (REQ_102 retired it on the producer side), so re-running analysis does **not** recompute `neuron_freq_norm`. The 4 consumers above still read it.

**Observed failure mode (p101/s999/ds42 dashboard smoke test, 2026-05-28).** With *N=119* checkpoints and *M=95* stale `neuron_freq_norm` artifacts:
- `neuron_dynamics` and `transient_frequency` (which load the full stack) silently produce a 95-row artifact — *misaligned* with the 119-row per-checkpoint axis the other analyzers and `variant_analysis_summary` use.
- `variant_analysis_summary` indexes `neuron_dynamics.max_frac[epoch_index]` with an index derived from `weight_spectra`'s 119-epoch summary → `IndexError: index 118 is out of bounds for axis 0 with size 95` (or 106-vs-95 mid-run).
- `neuron_group_pca` and `freq_group_weight_geometry` only read `neuron_freq_norm` at a *reference epoch* (`sorted_epochs[-1]`) — they happen to find the last in the 95 (e.g. 24999) and produce nominally-aligned output, so the symptom hits the trajectory-readers only.

**CoS test (post-migration).** Construct or use a variant in this state (*N* checkpoints, *M < N* on-disk `neuron_freq_norm`). After REQ_131, the 4 migrated consumers read `activation_basis_projection` (which the active family *does* produce → fresh for all *N*) and emit outputs aligned with the run's analyzed epochs; `variant_analysis_summary` completes without `IndexError`. (Equivalently: confirm the migration removes the stale-upstream failure mode by removing the stale-upstream dependency.)

**Related robustness concern (separate track, not REQ_131 by itself).** `variant_analysis_summary` indexes one analyzer's array (`neuron_dynamics.max_frac`) with an index derived from *another* analyzer's epoch set (`weight_spectra` summary epochs via `_get_nearest_checkpoint_epoch_index`). This cross-analyzer-axis assumption is fragile even outside this scenario; the robust version indexes each array by **its own** epochs (`neuron_dynamics_data["epochs"]` is already loaded into `analysis_data.neurons_checkpoints`). Worth a deliberate pass — likely on REQ_129's track or its own ticket.

## Notes

- **Closure feeds REQ_102.** When the three consumers no longer read `neuron_freq_norm`, REQ_102 retires `neuron_freq_clusters`.
- **Sibling of REQ_130** (`dominant_frequencies` unblock). The two deferred REQ_102 retirements each get their own consumer-migration REQ; this one is gated on REQ_128, REQ_130 is not.
- See [[frequency-choice-frame]] for the broader research context on frequency-related analyzers.
