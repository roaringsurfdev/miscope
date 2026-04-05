Viability Certificate Tool — Design Requirements
Purpose
A diagnostic tool that answers a specific question: given a prime p, a set of frequencies the model has selected or is accumulating toward, and observed properties of the model's representational space, is this frequency set geometrically viable — and if not, where precisely does it fail?
The tool is not trying to predict whether a model will grok. It's trying to characterize the quality of the destination the model is heading toward, and measure how far the model is from reaching it cleanly. It should be useful both as a post-hoc diagnostic and as an early-warning signal during training.

Background and Learnings from Development
Several assumptions built into the first version turned out to be wrong or incomplete, and understanding why shapes what the redesign needs to do differently.
The rank-based viability certificate was asking the wrong question. Full linear separability of p classes from the Fourier basis alone requires all (p-1)/2 frequencies. But models generalize successfully with 3-4. The certificate needs to ask whether the chosen frequencies produce viable geometry in the ambient embedding space, not whether they span the full Fourier basis. These are different questions with different answers.
Ideal frequency sets are not arithmetic. The first version used evenly-spaced frequencies as a proxy for the ideal set. This is wrong. The ideal set is the minimum-cardinality subset of frequencies that maximizes the minimum pairwise centroid distance in ambient space. It needs to be computed, not assumed. For small primes this is tractable by exhaustive search over subsets.
Separation is fragile under compression. A frequency set that looks geometrically viable in a 128-dimensional space may not be viable when the embedding compresses to an effective dimensionality of 15. The tool's first version had no model of this. The crossover PR — the participation ratio at the moment second descent completes — is the right constraint to apply when asking whether a frequency set is viable under real conditions.
High aliasing risk and high separation margin can coexist. This is counterintuitive but real. The p101 failure case (k=35,41,43,44) showed 68% separation margin alongside 81% aliasing risk. The tool reported this as viable. It isn't — the separation is achieved in a narrow corridor that is sensitive to small perturbations and cannot survive compression. The tool needs to weight aliasing risk as a ceiling on robustness, not just an independent signal.
The tool's effective dimension was trivially equal to dim utilization. Because idealized centroids were placed in perfectly orthogonal subspaces, participation ratio mechanically equaled 2|F|/d_model. This told you nothing. The real question is how the theoretical minimum dimension (2|F|) compares to the observed crossover PR, and whether the frequency geometry survives compression to that PR.
The regime classification is the right frame. Three regimes emerged clearly from the data: compact torus (low frequencies, rich structure, low PR), high-dim separation (fewer frequencies, ambient dimensions doing the work), and aliasing failure (high frequencies creating geometric collisions no ambient dimension can resolve). These map onto real qualitative distinctions across variants. The tool should keep this framing but ground it in real thresholds derived from calibrated examples rather than hardcoded values.

What the Tool Needs to Know
Inputs the tool computes analytically:

The centroid matrix for a given p and frequency set — where each residue class maps in Fourier space
Minimum and mean pairwise centroid distance in ambient space
The ideal minimum frequency set — computed by searching for the minimum-cardinality subset maximizing minimum pairwise centroid distance
The aliasing period for each frequency — p/k, which predicts which residue class pairs will be hardest to separate
Theoretical minimum neuron count — 2|F|

Inputs the tool needs from the model:

The observed frequency set (what the model has committed to or is accumulating toward)
The crossover PR — effective dimensionality at the moment second descent completes, separately for W_E, W_in, W_out, W_O
Optionally: the observed minimum Fisher J pair and the residue gap |r-s|, as a ground truth check on whether the tool's predicted hard pairs match what the model is actually struggling with

Inputs the user sets:

Prime p
d_model
Frequency amplitude α (for exploring coefficient inequality effects)
Which crossover PR to apply as the compression constraint — W_E is the most relevant since it bounds centroid placement


Core Metrics
Separation margin under compression. Not separation in idealized ambient space, but separation after compressing the centroid geometry to match the observed crossover PR. This is the primary viability signal. A frequency set is viable if and only if its minimum pairwise centroid distance remains above threshold after compression to the observed crossover PR.
Aliasing risk per frequency. k/(p-1)/2 — how far into the high-frequency range this frequency sits. Displayed per frequency, not just as a mean, because a single high-aliasing frequency can compromise an otherwise healthy set.
Predicted hard pairs. For each frequency k, the pairs separated by p/k steps are the hardest to distinguish. The tool should display which residue class pairs it predicts will be minimum Fisher J pairs, and these should be checkable against the observed Fisher heatmap.
Distance from ideal set. Which frequencies in the ideal set are missing. Which frequencies in the actual set are not in the ideal set. Whether the actual set is a viable alternative path — meaning it achieves comparable minimum pairwise distance despite different frequency choices — or whether it is categorically worse.
Compression survival. At what PR does the minimum pairwise centroid distance fall below the separation threshold? This tells you how much compression the frequency set can tolerate before geometry fails. A healthy set should survive compression well below the observed crossover PR. A fragile set fails near or above the crossover PR.
Regime classification. Compact torus, high-dim separation, or aliasing failure — with explicit criteria derived from calibrated examples rather than hardcoded thresholds.

Calibration Requirement
The tool's thresholds need to be grounded in observed outcomes before it goes into the platform. At minimum, the following known cases should be used to set separation margin thresholds and regime boundaries:

p59/s999/ds598: generalizes, frequencies {5, 15, 21}, crossover PR≈20
p59/s485/ds598: partial failure, frequencies {5, 21}, crossover PR≈14.8
p101/s999/ds598: failure, frequencies {35, 41, 43, 44}, crossover PR in the low 20s

These three cases span the range from clean generalization to aliasing failure and should produce clearly separated metric values if the tool is measuring the right things. If they don't separate cleanly, the metric definitions need revision before the tool is trusted.

Comparison Mode
The tool should support side-by-side comparison of two frequency sets against the same p and crossover PR. The primary use cases are:

Ideal set vs actual learned set
Two variants of the same prime with different seeds
Pre-second-descent frequency portfolio vs post-second-descent portfolio (to characterize what was lost and whether the loss mattered geometrically)

The comparison should display the delta in each metric explicitly, not just two separate readouts.

Visualization Priorities
Separation profile under compression is the most important visualization the first version was missing. This should show minimum pairwise centroid distance as a function of effective dimensionality, from d_model down to 1, with the observed crossover PR marked as a vertical line. The area where the curve drops below the separation threshold should be clearly indicated. A healthy model's curve stays above threshold well past the crossover PR. A fragile model's curve crosses below near the crossover PR.
Predicted vs observed hard pairs should be visually checkable. The tool predicts which pairs will be hardest to separate based on the aliasing period. This prediction should be overlaid on or directly comparable to the observed Fisher heatmap. Discrepancies between predicted and observed hard pairs are themselves diagnostic — they indicate the model has developed non-Fourier structure that the tool's idealized geometry doesn't capture.
Centroid geometry at reduced dimensionality — the existing centroid projection, but with a slider that applies the compression constraint so you can watch the geometry degrade as PR decreases toward the crossover value. This makes the compression survival concept visually concrete.
The existing visualizations — basis functions, full pairwise distance heatmap — are worth keeping as reference panels but are not primary.

What the Tool Is Not
It is not a predictor of whether a model will grok. It characterizes destination quality and compression robustness, not training dynamics. The cascade velocity, first-mover timing, and frequency accumulation race that determine which frequency set gets selected are upstream of this tool's scope. Those belong to the instrumentation pipeline — epoch-0 gradient energy, per-frequency accumulation trajectories, parameter velocity event detection. This tool takes the frequency selection as given and asks whether the selected set is a viable destination.
It is also not a substitute for the observed geometry. The tool's centroid geometry is idealized — it assumes perfectly orthogonal subspaces and equal amplitude across frequencies. Real models deviate from this in ways that matter. The Fisher heatmap and centroid PCA from the actual model should always be consulted alongside the tool's predictions, not instead of them.