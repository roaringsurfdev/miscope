# REQ_063: Fourier Nucleation Predictor

**Status:** Active
**Priority:** Medium
**Related:** REQ_056 (Frequency Specialization Sequencing), REQ_061 (Data Seed as Domain Parameter)
**Drafted by:** Research Claude (fourier-nucleation-spec.md). Scoped and edited by Engineering Claude.
**Last Updated:** 2026-03-08

---

## Problem Statement

We can observe which Fourier frequencies neurons specialize on after grokking, but we cannot currently make predictions about frequency competition *before or early in training*. The hypothesis is that frequency bias is encoded in the initialized weights: neurons whose response profiles (how strongly each neuron fires for each token value) have higher spectral energy at a given Fourier frequency are more likely to specialize at that frequency during training.

If this holds, the initialization weights should function as a nucleation predictor — something that can be read before training resolves to tell us which frequencies are predisposed to win. This is a testable claim: frequencies that dominate the epoch-0 projection should correlate with frequencies that capture the most neurons in the trained model.

The key architectural decision for this analysis: Fourier frequencies are defined over token values (0 to p-1). A neuron's response to each token value is `W_in[n] @ W_E[token]`, which in matrix form is `W_in @ W_E.T` — the neuron response matrix in token space. This is the right space to project onto the Fourier basis. Projecting `W_in` rows directly in the residual stream space (d_model dimensions) is not theoretically grounded for this analysis.

This requirement covers the epoch-0 nucleation predictor only. Cross-epoch frequency commitment trajectories are a natural follow-on once we know whether the signal is there.

---

## Conditions of Satisfaction

### 1. Analyzer: `fourier_nucleation`

- A new analyzer `fourier_nucleation` is implemented under `src/miscope/analysis/`
- The analyzer depends on `parameter_snapshot` having been run — it reads W_in and W_E from the epoch-0 parameter_snapshot artifact
- It is not a per-epoch analyzer; it produces a single artifact for the variant (keyed to epoch 0)
- Artifact stored at: `artifacts/fourier_nucleation/epoch_00000.npz`

**Computation:**

1. Load W_in (d_mlp × d_model) and W_E (vocab_size × d_model) from the epoch-0 parameter_snapshot artifact
2. Compute the neuron response matrix: `R = W_in @ W_E.T[:p]` → shape (d_mlp × p), where p is the prime from family context. Only the first p token indices are used (0 to p-1), excluding any padding or special tokens.
3. Build the Fourier basis for prime p: for each frequency k in {1..floor(p/2)}, vectors cos(2πk·i/p) and sin(2πk·i/p) for i=0..p-1
4. For each neuron n (row of R), compute Fourier energy at each frequency k: `energy(n, k) = proj_cos² + proj_sin²`
5. Normalize per-neuron: `fraction(n, k) = energy(n, k) / sum_k(energy(n, k))`
6. Run iterative sharpening for `iterations` steps (default 12):
   - For each neuron, zero out all Fourier components below the `sharpness` percentile of that neuron's energy
   - Reconstruct the neuron's response profile from surviving components
   - Re-project onto Fourier basis
7. At each iteration (including 0 = raw projection), record:
   - `aggregate_energy[iter, k]`: sum of energy across all neurons at frequency k, normalized by max
   - `neuron_peak_freq[iter, n]`: the frequency with highest energy for each neuron
   - `neuron_committed_count[iter, k]`: count of neurons with >15% of energy at frequency k

**Artifact contents (npz):**

- `aggregate_energy`: shape (iterations+1, n_freqs) — float32
- `neuron_peak_freq`: shape (iterations+1, d_mlp) — int32
- `neuron_committed_count`: shape (iterations+1, n_freqs) — int32
- `frequencies`: shape (n_freqs,) — int32, the k values [1..floor(p/2)]
- Metadata scalar fields: `prime`, `iterations`, `sharpness`

### 2. Views

Two views registered in the view catalog under the universal view set.

**`nucleation_spectrum`** (single-epoch-style, reads epoch-0 artifact):
- Row 1: Iteration × Frequency heatmap — rows=iterations (0 to N), columns=frequencies, color=normalized aggregate energy. Iteration 0 is the raw Fourier projection. Iteration N is the most sharpened. Click a row to select it.
- Row 2: Bar chart of aggregate energy per frequency for the selected iteration, with a secondary axis showing neuron committed count per frequency
- Row 3 (two panels): Neuron peak frequency histogram (distribution of which frequency each neuron peaks at) and Convergence traces (top-N frequencies tracked across iterations as line chart)

**`nucleation_emerging_frequencies`** (single-epoch-style, reads epoch-0 artifact):
- Ranked list of frequencies by energy gain from iteration 0 to final iteration (largest gain = most amplified by sharpening)
- Intended for quick scanning: which frequencies are the initialization most predisposed toward after the sharpening has run?

### 3. Dashboard integration

- Both views are available on the Neuron Dynamics page for any variant that has run the `fourier_nucleation` analyzer
- Views degrade gracefully (hidden, not erroring) when the `parameter_snapshot` epoch-0 artifact is unavailable

### 4. Tests

- Unit test: `W_in @ W_E.T[:p]` computation produces correct shape and that projections onto known Fourier vectors give expected energy values
- Unit test: iterative sharpening reduces cross-frequency entropy with each iteration (signal concentrates)
- Integration test: analyzer runs end-to-end on a small synthetic variant and produces a valid artifact

---

## Constraints

- This analyzer reads from existing `parameter_snapshot` artifacts. No new checkpoint loading infrastructure.
- Cross-epoch trajectory (applying this analysis at each checkpoint) is explicitly out of scope — that is a follow-on requirement
- The iterative sharpening parameters (`iterations`, `sharpness`) are fixed at analysis run time, not tunable per-view. If tuning is desired, re-run the analyzer with different parameters
- W_E tokens 0..p-1 only. If the model's vocabulary includes a padding or equals token at index p, it is excluded from the response matrix
- Artifact format follows the standard `epoch_{NNNNN}.npz` convention even though this is a single-epoch analysis, for compatibility with the existing ArtifactLoader

---

## Notes

- **Validation approach:** The nucleation predictor makes a testable claim. Compare `nucleation_emerging_frequencies` output against per-band neuron specialization data (from `dominant_frequencies` artifacts) for the same variant. If frequencies that rank highest after sharpening are the same frequencies that capture the most neurons in the trained model, the prediction holds. This validation is done by visual inspection for now — not automated scoring.
- **The anomalous variants are the interesting test cases.** p=101/seed=999 and p=113/data_seed=999 are the models we know behaved unexpectedly. If the nucleation predictor assigns high energy to frequencies that the trained model *failed to sustain*, that is an informative result even if it isn't a clean correlation.
- **Iteration 0 is interpretable on its own.** The raw Fourier projection (no sharpening) shows the baseline spectral bias of the initialization. The sharpening iterations amplify latent structure. If iteration 0 and iteration N look similar, the initialization already has concentrated spectral energy. If they differ significantly, the sharpening is revealing structure that wasn't obvious in the raw projection.
- **Follow-on scope:** Cross-epoch frequency commitment trajectory — apply this analysis at each checkpoint to produce a (neuron × epoch) view showing when neurons commit to their dominant frequency. This is the visualization for watching frequency competition resolve over training.
- **Data Compatibility (Component 2 of the research spec) is not in scope here.** It is a strong candidate for a follow-on requirement, particularly given the data_seed sensitivity now being actively investigated via REQ_061.
