# REQ_131: Migrate `neuron_freq_norm` consumers to `activation_basis_projection` (unblock `neuron_freq_clusters` retirement)

**Status:** CoS developed (2026-05-29) — implementation in progress. **Now unblocked:** REQ_128 merged to `develop` (`022b48e`), so the selective/lazy loading this migration needs is available.
**Priority:** High — unblocks the second REQ_102 deferral *and* removes the root cause of the stale-upstream `IndexError` surfaced at REQ_128 close-out.
**Branch:** `feature/req-131-neuron-freq-norm-consumer-migration` (off `develop`).
**Dependencies:**
- REQ_102 (Analyzer Deprecation — deferred the `neuron_freq_clusters` retirement because of these consumers).
- **REQ_128 (Analyzer Input Provisioning — *now merged*).** The re-point loads the granular `activation_basis_projection` cube; REQ_128's selective (`fields=`) and streaming (`stream`) accessor is what keeps the backbone consumers from re-importing whole-artifact memory pressure (`load_epoch` had no `fields=` before REQ_128).
- REQ_126 (provides `activation_basis_projection` and the bitwise-validated reconstruction).

**Attribution:** Engineering Claude (stubbed 2026-05-28 from the REQ_102 analyzer-dependency audit; CoS developed 2026-05-29).

---

## Problem

`neuron_freq_clusters` registers the artifact **`neuron_freq_norm`**, which is a live declared dependency of three surviving analyzers: `neuron_dynamics`, `freq_group_weight_geometry`, `neuron_group_pca`. Retiring `neuron_freq_clusters` requires re-pointing those consumers to the generic successor `activation_basis_projection` (the original was specialized rather than generic — the `freq_cluster` naming reflects that). Per user, `neuron_freq_norm` is a **backbone of critical analysis**; do not change it until the migration is clearly understood.

## Captured learnings (from the 2026-05-28 audit — the migration is tractable)

- **Uniform, narrow contract.** All three consumers use a *single* field, `norm_matrix` (shape `(n_freq, d_mlp)` per epoch; stacked `(n_epochs, n_freq, d_mlp)`), and use it identically: `argmax` / `max` over the frequency axis for group assignment and dominant-frequency tracking. Nothing else is read.
- **Reconstruction already exists and is bitwise-validated.** `_reconstruct_legacy_neuron_freq_norm(result, prime, "mlp_out")` (in `test_activation_basis_projection.py`) rebuilds `norm_matrix` from `activation_basis_projection`'s `mlp_out` site at parity tolerance. Promote it to a shared library/adapter helper.
- **Behavior-preserving.** Because the reconstructed `norm_matrix` is bitwise-identical, the consumers' outputs are unchanged — this defuses the "backbone, don't touch" risk; it is a safe re-point, not a behavioral change.
- **The only real complication is loading**, hence the REQ_128 dependency: consumers currently `load_epoch("neuron_freq_norm")` (small); re-pointing to the granular `activation_basis_projection` artifact wants `fields=`/lazy loading to stay cheap.

## Design (developed 2026-05-29, grounded in the code on `develop`)

### Consumer inventory — 3 direct, 1 transitive (count correction)

The problem statement names three consumers; the REQ_128 close-out narrative says "4 consumers." Reconciled against the code: there are **3 direct readers** of `neuron_freq_norm`, plus **1 transitive** reader that inherits the fix:

| Analyzer | Read shape (current) | Migration |
|---|---|---|
| `neuron_dynamics` | `deps.load_stack("neuron_freq_norm", epochs=inputs.epochs, fields=["norm_matrix"])` — **full trajectory** | re-point + **stream** `activation_basis_projection` |
| `neuron_group_pca` | `deps.load_epoch("neuron_freq_norm", reference_epoch, fields=["norm_matrix"])` — **reference epoch only** | re-point single `load_epoch` |
| `freq_group_weight_geometry` | `deps.load_epoch("neuron_freq_norm", reference_epoch, fields=["norm_matrix"])` — **reference epoch only** | re-point single `load_epoch` |
| `transient_frequency` | reads **`neuron_dynamics`** (not `neuron_freq_norm`) via `load_cross_epoch` | **no change** — inherits `neuron_dynamics`'s now-N-aligned output |

So the migration touches **3 analyzers**; `transient_frequency` needs no edit but is part of the regression's blast radius and must be re-verified.

### The shared reconstruction helper

Promote `_reconstruct_legacy_neuron_freq_norm` (in `tests/test_activation_basis_projection.py`) to a library helper (e.g. `analysis/library/`). It rebuilds the legacy `norm_matrix` `(n_freq, d_mlp)` from `activation_basis_projection`'s `mlp_out` site at parity tolerance (rtol=1e-3, already validated). It needs three fields and the prime:
- `mlp_out_power` `(d_mlp, K, K)`, `mlp_out_axis_a_marginal_power` `(d_mlp, K)`, `mlp_out_axis_b_marginal_power` `(d_mlp, K)`.
- **`prime` is sourced from `context["params"]["prime"]`** — the same path `activation_basis_projection` uses; it is present in the cross-epoch analysis context, so the two consumers that don't currently read `prime` can.

### Memory shape — why `neuron_dynamics` must `stream`, not `load_stack`

`neuron_freq_norm` stored a small `(n_epochs, K, d_mlp)` matrix, so `load_stack` was cheap. `activation_basis_projection`'s `mlp_out_power` is a large `(d_mlp, K, K)` cube *per epoch*. `neuron_dynamics` must therefore **`stream`** `activation_basis_projection` over `inputs.epochs`, reconstruct the small `(K, d_mlp)` `norm_matrix` per epoch (discarding the big cube), and stack only the reconstructions — keeping one power cube resident at a time. This is exactly the REQ_128 streaming spine; using `load_stack` on the raw cube would re-import the whole-artifact memory pressure this REQ is meant to avoid. The two reference-epoch consumers use a single `load_epoch` (one cube), which is fine.

### Layout note

`activation_basis_projection` is `output_scope="per_epoch"`, so consumers use the per-epoch verbs (`load_epoch` / `stream`), **not** `load_cross_epoch`. `neuron_dynamics`'s own output (`output_scope="cross_epoch"`) and all its artifact keys (`epochs`, `dominant_freq`, `max_frac`, `switch_counts`, `commitment_epochs`) are **unchanged** — only the input source changes, so `transient_frequency`, `variant_analysis_summary`, and the views need no edits.

## Conditions of Satisfaction

### Reconstruction helper

- [ ] `_reconstruct_legacy_neuron_freq_norm` is promoted from the test module to a shared library helper with a clear name, signature `(abp_result, prime, site_prefix="mlp_out") -> norm_matrix (n_freq, d_mlp)`, and a docstring stating the parity claim. The test module imports the promoted helper (no duplicated formula).
- [ ] A unit test asserts the helper reproduces the legacy `neuron_freq_norm.norm_matrix` from `activation_basis_projection` on canon within `rtol=1e-3` (the existing `test_parity_neuron_freq_norm`, re-pointed at the library helper).

### Consumer migration (3 direct)

- [ ] `neuron_dynamics` declares `ArtifactInput("activation_basis_projection")` (drops `neuron_freq_norm`), **streams** it over `inputs.epochs` with `fields=["mlp_out_power", "mlp_out_axis_a_marginal_power", "mlp_out_axis_b_marginal_power"]`, reconstructs the per-epoch `norm_matrix`, and produces the **same output keys/shapes** as before. At most one power cube resident at a time (no `load_stack` of the raw cube).
- [ ] `neuron_group_pca` and `freq_group_weight_geometry` declare `ArtifactInput("activation_basis_projection")` (drop `neuron_freq_norm`), read it at the reference epoch via a single `load_epoch` with the three power fields, reconstruct `norm_matrix`, and are otherwise unchanged.
- [ ] Each migrated analyzer sources `prime` from `context["params"]["prime"]`; the `requires`/`ArtifactInput` declarations match actual reads (REQ_128 scope enforcement passes).
- [ ] No surviving reference to the `neuron_freq_norm` artifact remains in any analyzer (grep-clean), unblocking REQ_102's deletion of `neuron_freq_clusters`.

### Parity

- [ ] On canon, the migrated `neuron_dynamics`, `neuron_group_pca`, and `freq_group_weight_geometry` outputs match their pre-migration outputs within `rtol=1e-3` (per [[feedback_req126_float64_parity]]). **Argmax caveat:** `neuron_dynamics` takes `argmax`/`max` over the frequency axis; near-tie neurons could flip on float-noise reconstruction. If any flips occur, confirm they are at genuine near-ties (frac difference within tolerance) and record as a finding, not a regression.

### Regression (the close-out bug — root cause removed)

- [ ] A CoS test reproduces the stale-upstream scenario and shows the migration removes the failure mode. See the scenario below. Because the active family **always** produces `activation_basis_projection` (it is not gated on the retired `neuron_freq_clusters`), the migrated `neuron_dynamics` emits an output keyed to `inputs.epochs` (all *N*), so `variant_analysis_summary` indexing completes without `IndexError`. The test asserts: (a) `neuron_dynamics` output epoch axis length == *N*; (b) `transient_frequency` (transitive) is *N*-aligned; (c) `variant_analysis_summary.analyze()` completes for the *N*-vs-*N* case.

### Closure

- [ ] Hand the `neuron_freq_clusters` retirement back to REQ_102 (it can delete the analyzer, its renderer, and the `neuron_freq_norm` producer once this lands).

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
