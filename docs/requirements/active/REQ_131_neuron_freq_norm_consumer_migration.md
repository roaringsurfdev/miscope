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

## Notes

- **Closure feeds REQ_102.** When the three consumers no longer read `neuron_freq_norm`, REQ_102 retires `neuron_freq_clusters`.
- **Sibling of REQ_130** (`dominant_frequencies` unblock). The two deferred REQ_102 retirements each get their own consumer-migration REQ; this one is gated on REQ_128, REQ_130 is not.
- See [[frequency-choice-frame]] for the broader research context on frequency-related analyzers.
