# REQ_131: Migrate `neuron_freq_norm` consumers to `activation_basis_projection` (unblock `neuron_freq_clusters` retirement)

**Status:** Implementation complete (2026-05-29) on `feature/req-131-neuron-freq-norm-consumer-migration` — awaiting merge approval to `develop`. Full miscope suite green (1518 passed, 27 skipped); canon-gated parity test re-pointed to the promoted helper. Scope was extended (with user approval) to the 3 View Catalog views REQ_127 missed. **Now unblocked:** REQ_128 merged to `develop` (`022b48e`), so the selective/lazy loading this migration needs is available.
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

`activation_basis_projection` is `output_scope="per_epoch"`, so consumers use the per-epoch verbs (`load_epoch` / `stream`), **not** `load_cross_epoch`. `neuron_dynamics`'s own output (`output_scope="cross_epoch"`) and all its artifact keys (`epochs`, `dominant_freq`, `max_frac`, `switch_counts`, `commitment_epochs`) are **unchanged** — only the input source changes, so consumers of `neuron_dynamics`'s *output* (`transient_frequency`, `variant_analysis_summary`) need no edits. The View Catalog views that read the `neuron_freq_norm` *artifact directly* do need re-pointing — see "Discovered during implementation."

## Conditions of Satisfaction

### Reconstruction helper

- [x] `reconstruct_neuron_freq_norm` is promoted to `analysis/library/basis_reconstruction.py` (signature `(projection, prime, site_prefix="mlp_out") -> norm_matrix (n_freq, n_units)`), exported from `analysis.library`, alongside `NEURON_FREQ_NORM_FIELDS`. The test module imports it (no duplicated formula).
- [x] `test_parity_neuron_freq_norm` re-pointed at the library helper (canon-gated, `rtol=1e-3`; skips without canon data — green where canon present).

### Consumer migration (3 direct)

- [x] `neuron_dynamics` declares `ArtifactInput("activation_basis_projection")`, **streams** it over `sorted(inputs.epochs)` with `fields=NEURON_FREQ_NORM_FIELDS`, reconstructs the per-epoch `norm_matrix`, and produces the **same output keys/shapes**. One power cube resident at a time (no `load_stack` of the raw cube).
- [x] `neuron_group_pca` and `freq_group_weight_geometry` declare `ArtifactInput("activation_basis_projection")`, read it at the reference epoch via a single `load_epoch` with the three power fields, reconstruct `norm_matrix`, otherwise unchanged.
- [x] Each migrated analyzer sources `prime` from `context["params"]["prime"]`; `requires`/`ArtifactInput` updated (REQ_128 scope enforcement passes — full suite green).
- [x] No surviving reference to the `neuron_freq_norm` artifact remains in any analyzer (grep-clean).

### Parity

- [x] Behavior-preserving by construction: the reconstruction is bitwise-validated (`rtol=1e-3`) and the consumers' downstream logic (`argmax`/`max` over the frequency axis) is untouched. Full-process canon parity (`run_regression_check.py`) is the user-run validation step before merge. **Argmax caveat** stands: near-tie neurons could flip on float-noise reconstruction — if so, confirm genuine near-ties and record as a finding, not a regression.

### Regression (the close-out bug — root cause removed)

- [x] `test_keys_to_inputs_epochs_not_all_available` (in `test_neuron_dynamics.py`) is the analyzer-level guard: it analyzes a *subset* of the on-disk `activation_basis_projection` epochs and asserts the output epoch axis equals the analyzed set — i.e. `neuron_dynamics` tracks `inputs.epochs`, never the on-disk upstream set. Because the active family always produces ABP fresh for all analyzed epochs, the upstream can no longer be short ⇒ no 95-row artifact ⇒ no `variant_analysis_summary` `IndexError`. The *N*-vs-*N* `variant_analysis_summary.analyze()` completion is covered by the existing green `test_variant_summary` suite.

### Closure

- [x] Hand the `neuron_freq_clusters` retirement back to REQ_102 — no live consumer (analyzer or View Catalog view) reads `neuron_freq_norm` after this lands. (One non-blocking caveat: the `export.py` `_VISUALIZATION_REGISTRY` still names it — see "Discovered during implementation.")

### Regression scenario surfaced during REQ_128 (must be a CoS test)

**Scenario.** A variant has *N* checkpoints, but a previous analysis (run when `neuron_freq_clusters` was still in the family) left only *M < N* `neuron_freq_norm` epoch artifacts on disk. The family no longer includes `neuron_freq_clusters` (REQ_102 retired it on the producer side), so re-running analysis does **not** recompute `neuron_freq_norm`. The 4 consumers above still read it.

**Observed failure mode (p101/s999/ds42 dashboard smoke test, 2026-05-28).** With *N=119* checkpoints and *M=95* stale `neuron_freq_norm` artifacts:
- `neuron_dynamics` and `transient_frequency` (which load the full stack) silently produce a 95-row artifact — *misaligned* with the 119-row per-checkpoint axis the other analyzers and `variant_analysis_summary` use.
- `variant_analysis_summary` indexes `neuron_dynamics.max_frac[epoch_index]` with an index derived from `weight_spectra`'s 119-epoch summary → `IndexError: index 118 is out of bounds for axis 0 with size 95` (or 106-vs-95 mid-run).
- `neuron_group_pca` and `freq_group_weight_geometry` only read `neuron_freq_norm` at a *reference epoch* (`sorted_epochs[-1]`) — they happen to find the last in the 95 (e.g. 24999) and produce nominally-aligned output, so the symptom hits the trajectory-readers only.

**CoS test (post-migration).** Construct or use a variant in this state (*N* checkpoints, *M < N* on-disk `neuron_freq_norm`). After REQ_131, the 4 migrated consumers read `activation_basis_projection` (which the active family *does* produce → fresh for all *N*) and emit outputs aligned with the run's analyzed epochs; `variant_analysis_summary` completes without `IndexError`. (Equivalently: confirm the migration removes the stale-upstream failure mode by removing the stale-upstream dependency.)

**Related robustness concern (separate track, not REQ_131 by itself).** `variant_analysis_summary` indexes one analyzer's array (`neuron_dynamics.max_frac`) with an index derived from *another* analyzer's epoch set (`weight_spectra` summary epochs via `_get_nearest_checkpoint_epoch_index`). This cross-analyzer-axis assumption is fragile even outside this scenario; the robust version indexes each array by **its own** epochs (`neuron_dynamics_data["epochs"]` is already loaded into `analysis_data.neurons_checkpoints`). Worth a deliberate pass — likely on REQ_129's track or its own ticket.

## Discovered during implementation (2026-05-29)

- **3 View Catalog views also read `neuron_freq_norm` directly** (`neuron_group.scatter`, `neuron_group.scatter_purity`, `neuron_group.all_groups` in `views/universal.py`) — a gap REQ_127 left when it re-pointed the *other* neuron-freq views. Per user decision, folded into REQ_131: re-pointed through REQ_127's existing `_adapt_activation_freq_legacy(art, "mlp_out", "norm_matrix")` adapter (same view-side adapter the sibling views use, so normalization stays consistent; renderers only need argmax for grouping + `[0,1]` purity coloring). `AnalyzerRequirement` lists and `epoch_source_analyzer` updated accordingly.
- **`export.py` `_VISUALIZATION_REGISTRY` still names `neuron_freq_norm`** (3 entries) — but it is **already uniformly stale** across the `attention_freq` and `dominant_frequencies` retirements too (all three are no longer family-produced). Two of the `neuron_freq_norm` entries use the `"summary"` data-pattern, which ABP does not emit, so a correct re-point needs adapter + summary-path work that doesn't exist. This is the **broader retirement-cleanup track (REQ_102 / REQ_129)**, not REQ_131; left out of scope deliberately. It does not block `neuron_freq_clusters` deletion any more than the pre-existing `attention_freq`/`dominant_frequencies` entries already do.
- **`modulo_addition_learned_emb_mlp` family is internally inconsistent (pre-existing, not REQ_131).** Its `cross_epoch_analyzers` list includes `neuron_dynamics`/`neuron_group_pca`/`freq_group_weight_geometry`, but the family class declares no `activation_basis_projection_sites` and its `analyzers` list omits ABP — so it produces neither the old `neuron_freq_norm` nor the new `activation_basis_projection`. It was already broken before REQ_131 (the consumers depended on `neuron_freq_norm`, also unproduced there); it carries stale on-disk artifacts from an old run. Fixing it (add ABP + declare sites, or drop the unsupported analyzers) is a separate family-config task.

## Notes

- **Closure feeds REQ_102.** When the three consumers no longer read `neuron_freq_norm`, REQ_102 retires `neuron_freq_clusters`.
- **Sibling of REQ_130** (`dominant_frequencies` unblock). The two deferred REQ_102 retirements each get their own consumer-migration REQ; this one is gated on REQ_128, REQ_130 is not.
- See [[frequency-choice-frame]] for the broader research context on frequency-related analyzers.
