# Transform Foundations & Analyzer Audit

A foundation and quality-check reference for analyzers in MIScope. Every analyzer applies a transformation that *moves* data from one representation to another. The interpretation of any result depends on conditions under which that move is legitimate. This document names those conditions and provides a template for auditing them per analyzer.

## Why this exists

Analyzers in MIScope take data in one space (weights, activations, logits, trajectories) and produce findings in another (principal components, frequency bands, DMD modes, curvature classes). The findings are only as meaningful as the transform is faithful — and faithfulness is not a binary. It is a stack of conditions, each of which can hold or fail in ways that change what the output *means*.

Without an explicit handle on these conditions, it is easy to:

- Report structure that is an artifact of the chosen basis.
- Conflate "the math ran" with "the result is interpretable."
- Make geometric claims (saddle, orbit, cascade) that the representation cannot support.
- Lose the ability to distinguish "the system has no structure here" from "the transform cannot see the structure here."

This document is the reference and quality check that prevents those failures from compounding silently.

## The four condition layers

For every transform `T: X → Y` deployed in an analyzer, four kinds of conditions govern what the output can legitimately mean.

### 1. Existence / well-posedness

When is `T` defined on the input at all?

- Fourier (DFT): finite-length, evenly sampled signal.
- PCA: finite second moments; well-defined covariance estimator.
- DMD: snapshot pairs admit a least-squares solution; snapshot matrix not rank-deficient in a way that destroys the pseudoinverse.
- Takens embedding: underlying dynamics live on a finite-dimensional manifold.
- Hessian via Lanczos: loss is twice-differentiable at the evaluation point; HVP is numerically stable.

**Failure mode:** output is undefined, NaN, or numerically meaningless. Not "wrong" — *nonsense*.

### 2. Faithfulness / invertibility

When does `T(x)` preserve the information about `x` that matters for the claim?

- Fourier on L²: isometry; Parseval guarantees no loss.
- PCA truncated to `k` components: lossy; faithful iff trailing eigenvalues are small relative to retained.
- DMD: faithful to dynamics iff the system is approximately linear in the chosen observable space (Koopman framing). Failure on regime-switching or strongly nonlinear systems.
- Takens: diffeomorphism (topology + geometry preserved up to smooth deformation) iff embedding dim `> 2d` and generic delay.
- Windowed transforms: faithfulness is local to the window; cross-window claims require extra justification.

**Failure mode:** the output is well-defined but represents something other than the input. Structure can be lost, smeared, or invented.

### 3. Interpretive / structural

When do the features read off `T(x)` mean what the analyzer claims they mean?

This is the subtlest layer and the most common site of overreach.

- PCA components as "directions of variance": valid for roughly ellipsoidal data. On multimodal or curved manifolds, PC1 can point in a direction no datapoint occupies.
- DMD eigenvalues as "growth rates and frequencies of modes": valid for approximately LTI systems within the window. On regime-switching systems, global DMD eigenvalues are weighted averages corresponding to no real dynamical structure.
- Fourier coefficients as "amount of frequency ω present": valid for stationary signals. On non-stationary signals, coefficients smear (motivation for STFT/wavelets).
- Phase-space geometry (loops, saddles, heteroclinic connections) as dynamical structure: requires faithful embedding *and* coordinates that don't compress fast directions onto slow ones.
- Lissajous amplitude assignment (PC1 = A_x, PC2 = κA_x, PC3 = A_z): requires that the principal directions actually correspond to the orbit's amplitude axes, not just to variance directions that happen to numerically agree.

**Failure mode:** the output is faithful to the input but the *named features* are not the things in the world the analyzer says they are.

### 4. Statistical / sampling

When does the finite, noisy estimate of `T(x)` approximate the true transform?

- Covariance estimation: requires `n ≫ d` or eigenstructure is spurious (Marchenko–Pastur).
- DMD with noise: biased; total-least-squares DMD or optimized DMD address this.
- Takens with finite, noisy data: delay/dimension selection problem (mutual information, false nearest neighbors).
- Hessian via Lanczos: top eigenvalues converge faster than tail; tail estimates require more iterations.

**Failure mode:** the transform itself is fine but the estimate from available data is not close to the true value.

## How to use this

For every analyzer, before reporting results as findings:

1. Identify the transform(s) the analyzer applies.
2. For each transform, state which conditions hold, which are *assumed*, and which are *checked*.
3. If a condition is assumed rather than checked, name what would falsify it and whether that failure mode would be visible in the output.
4. When presenting results, separate the *transform-conditional* claim ("under conditions X, Y, Z, the data exhibits a saddle in PC space") from the *system-level* claim ("the model is undergoing a saddle transit"). The second requires the first plus an argument that the transform conditions hold.

The breakdown of an interpretive condition is itself often a finding. (Global DMD fitting the Thrasher *best* is a clean instance: the system where the linearity assumption holds globally is the pathological one, because nothing interesting is happening.)

## Note to Engineering Claude

When building, modifying, or reviewing an analyzer in this project, treat this document as the standard the work is held to.

Concretely:

- When proposing a new analyzer, articulate the transform(s) it deploys and walk the four condition layers explicitly before discussing implementation.
- When extending an existing analyzer, flag whether the extension changes which conditions are assumed vs. checked.
- When reviewing output or interpreting a result, prefer transform-conditional language ("the data is consistent with X in this representation, under conditions A, B, C") over system-level language ("the model is doing X") unless the conditions are verified.
- Treat unexpected breakdowns of an interpretive condition as candidate findings, not bugs to paper over.
- When uncertain whether a condition holds, say so — and propose how it could be checked. The honest unknown is more useful here than a confident assertion.

The goal is not exhaustive formal proof for every analyzer. It is shared discipline: that the move from observation to claim is always traceable, and that the representation is never doing silent work the interpretation depends on.

---

## Audit template

Copy this template per analyzer. Live audits should sit alongside the analyzer code (e.g., `analyzers/<name>/AUDIT.md`) and be updated when the analyzer changes.

```markdown
# Analyzer audit: <name>

## Transform(s) deployed

- Input space: <what space the data lives in before the transform>
- Output space: <what space the results live in>
- Map: <the transform applied, named precisely>
- Composition: <if multiple transforms are chained, list them in order>

## Layer 1 — Existence / well-posedness

- Conditions: <what must be true for the transform to be defined>
- Status: <held / assumed / checked — and how>
- Failure visibility: <would a violation produce NaN, error, silent garbage?>

## Layer 2 — Faithfulness / invertibility

- Conditions: <what must be true for the output to preserve the relevant information>
- Status: <held / assumed / checked>
- Known loss: <what information the transform discards by design>

## Layer 3 — Interpretive / structural

- Named features: <what the analyzer calls the features in the output (e.g., "saddle", "mode", "amplitude axis")>
- Conditions for those names: <what must be true for those names to refer to the things in the world they suggest>
- Status: <held / assumed / checked>
- Known confounders: <conditions under which the named features would appear in the output without the corresponding structure existing in the system>

## Layer 4 — Statistical / sampling

- Conditions: <sample size, noise level, sampling regime>
- Status: <held / assumed / checked>
- Sensitivity: <how the output changes with sample size, noise, window length>

## Claims this analyzer can support

- Transform-conditional: <claims of the form "in this representation, under these conditions, the data exhibits X">
- System-level: <claims of the form "the model is doing X"> — and what conditions are required to bridge from transform-conditional to system-level

## Open questions

<conditions that are assumed but not yet checked; planned checks; known edge cases>
```
