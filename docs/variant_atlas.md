# Variant Atlas

**Status:** Living document
**Last updated:** 2026-05-09
**Audience:** Researchers using miscope, contributors planning new analyzers, future collaborators (and future self) needing to remember which variant did what.

---

## Purpose

The Variant Atlas is a curated index of the model variants we've explicitly studied, consolidating signatures across analytical lenses. It serves two functions:

- **A navigator** for the variant set. Looking at a new analysis output and want to know "which variant am I looking at?" — start here.
- **A consolidator** for findings that would otherwise live scattered across `notes/findings_*.md`, fieldnotes drafts, REQ validation outcomes, and project memory.

It is **not exhaustive.** The full variant set has ~40 entries (see `results/modulo_addition_1layer/variant_registry.json` for the compiled aggregate, and per-variant `variant_summary.json` files for raw training metrics). The Atlas covers variants we've examined deeply enough to have research observations attached.

---

## How to read an entry

Every entry carries:

- **Identifier**: `p{prime}/s{seed}/ds{data_seed}` — used throughout the codebase and conversation.
- **Role**: a short archetype label (`canon`, `clean grokker`, `degenerate`, `failure`, `rebounder`, `thrasher`, etc.).
- **Training character**: one paragraph describing the loss curve and grokking timing.
- **Signatures**: bullet points keyed by analytical lens. Each is a one-line claim plus the relevant epoch range; deeper context lives in the linked fieldnotes / findings files.
- **Cross-references**: pointers to the fieldnotes or findings files that go deeper.

Sites referenced: `resid_pre` (post-embed), `attn_out`, `mlp_out`, `resid_post`. Frequencies are written as `k=N` where N is the dominant Fourier basis component.

---

## Reference Variants

The four variants used as the canonical reference set throughout REQ_117 phase 1. Selected to span "clean grokker → canon → degenerate → failure" while sharing the canonical data seed (598).

These variants are also some of the oldest variants in the set, and for that reason, they often serve as the first introduction to anomalous behaviors seen again in later models.

### `p113/s999/ds598` — Canon

The reference / primary model. Most-studied variant in the project; new analysis and models can be validated against this model as the standard.

**Training character.** Memorization plateau through ~9k epochs, second descent ~9–11k, post-grokking convergence by ~15k. Test loss settles around ~80n. Documented frequencies: `{9, 33, 38, 55}`.

**Signatures:**
- *activation_dmd*: pendant of paired complex conjugate excursions near (1, 0) at `mlp_out`; smooth wing arcs at `resid_pre` and `attn_out`. Wide grokking bump at `resid_post` centered ~10k. ([fieldnotes draft: dmd-at-zoom.mdx](../apps/fieldnotes/src/content/drafts/dmd-at-zoom.mdx))
- *parameter_dmd*: 4 populated groups (one per dominant frequency) at default `reference_epoch=24999`. W_out has consistently wider vertical eigenvalue spread than W_in across all four groups. Per-group eigenvalue character is meaningfully distinct (group 8 tight; group 37 wing/loop; group 54 vertical lens).
- *Parameter Trajectory PCA*: self-intersecting PC2/PC3 loop. Lands at the node — the canonical healthy topology. ([fieldnotes: grokking-lemniscate.mdx](../apps/fieldnotes/src/content/drafts/grokking-lemniscate.mdx))
- *Neuron specialization*: diffuse — frac_explained 0.4–0.9 (not ≈1.0) at epoch 24999. The "healthy but structurally diffuse" pattern.
- *Ring geometry*: present and stable post-grokking. ([findings_ring_geometry.md](notes/findings_ring_geometry.md))
- *Performance classification*: `healthy`.

### `p109/s485/ds598` — Clean grokker

One end of the spectrum. Fast, tight grokking trajectory.

**Training character.** Very short memorization plateau (<5k epochs); sharp second descent at ~3–5k; post-grokking convergence by ~6k.

**Signatures:**
- *activation_dmd*: discrete fan + paired complex conjugates eigenvalue signature at `mlp_out`, qualitatively distinct from canon's smooth pendant. Sharp early reorganization peaks at 1.5k–5.5k. Late-training boundaries near 22–24k confirmed real ("something poised to happen at the end" — anticipated, not threshold artifacts).
- *parameter_dmd*: 3 populated groups (k=4, k=14, k=27). **Group 3 (k=4) shows clear W_in / W_out desynchronization at ~2.5k** — W_out drops to near-zero while W_in is still elevated. This is the cleanest empirical justification for treating W_in and W_out as separate matrices.
- *Saddle transit*: nearly-aligned saddle transits with low orthogonality in early training.
- *Site gradient convergence*: reference healthy model — embedding↔MLP and attention↔MLP pairs all converge tightly. ([findings_site_gradient_convergence.md](notes/findings_site_gradient_convergence.md))
- *Group trajectory proximity*: tight-tracking signature; phase lock achieved.
- *End-of-training note*: small `mlp_out` activity 22–25k that does NOT have a corresponding parameter-side counterpart — candidate **neural collapse signature** flagged for follow-up.

### `p101/s999/ds598` — Degenerate (multi-segment orthogonal drift)

Open-loop PC2/PC3 geometry; multi-lens validation case for transient-frequency phenomena.

**Training character.** Long memorization plateau (~12k epochs); messy multi-segment second descent at ~12–15k; never quite reaches canonical loss floor. Has a transient frequency (k=5) committed at ~60 neurons from ~12k onwards, fully abandoned around 25k.

**Signatures:**
- *activation_dmd*: teardrop + isolated complex pairs at `resid_post`. **Stuttering peak pair at 12–14k** across all four sites — the failed-transit-and-retry visible in residual structure.
- *activation_dmd sub-regime event*: coordinated bump at ~24.1k across all four sites simultaneously, immediately before the residual settles to a quieter floor. Below the boundary detector's height threshold (correctly — the frequency abandonment is sub-regime-magnitude, not a structural reorganization), but the signal is present.
- *parameter_dmd*: 4 populated groups at default `reference_epoch=24999` (post-abandonment); **5 populated groups at pinned `reference_epoch=20000`** (where freq 5 is still committed at 39 neurons). Pinned run shows freq 5 with a ~25k abandonment bump in both W_in and W_out. **Group 4's W_out drop after the abandonment peak is steeper than the other groups' W_out** — candidate "transient frequency abandonment" signature for cross-variant investigation.
- *Saddle transit*: failed transit candidate with multi-segment orthogonal drift in parameter space.
- *Triple-lens convergence*: DMD residual + saddle transit + transient frequency tracking all agree on the same dynamical events at appropriate scales (big events get formal regime markers; the freq abandonment leaves traces in the signal across all three lenses without forcing a regime change). ([notes/findings_dmd_convergence.md](notes/findings_dmd_convergence.md))

### `p59/s485/ds598` — Failure mode

Other end of the spectrum from p109. Overshooter pattern.

**Training character.** Long plateau; second descent late and messy; multiple late-training reorganization events past 20k. Eventually grokks but with residual instability.

**Signatures:**
- *activation_dmd*: sparse near-real eigenvalues — qualitatively distinct from the three grokkers' wing/fan/teardrop signatures. The lens that produced rich structure elsewhere produces almost a flat line here.
- *parameter_dmd*: 3 populated groups. W_in and W_out track each other tightly — but in a non-learning trajectory where neither matrix is doing meaningful work, so the agreement is uninformative.
- *Parameter Trajectory PCA*: overshoots through the node with too much momentum, ends up on the far side of the loop. Shows backtracking under extended training (the intersection is a genuine attractor). ([fieldnotes: grokking-lemniscate.mdx](../apps/fieldnotes/src/content/drafts/grokking-lemniscate.mdx))

---

## Anomaly variants (selected by dense checkpointing during exploration)

These three were examined during REQ_117 development as "what does activation_dmd surface on atypical models?" Picked specifically because of their existing dense checkpointing.

### `p101/s999/ds999` — Two transient frequencies

Same model seed as the degenerate p101 above, but with data seed 999, which has shown to create instable models.

**Signatures:**
- Has two transient frequencies (vs. one for `p101/s999/ds598`).
- *activation_dmd*: late-training Post-Embed eigenvalues sit *outside* the unit circle (Re(λ) up to 1.10, |λ| > 1) — growing modes that don't decay. Loss curve corroborates with a rebound bump at 7–9k and a settling floor higher than canonical (~200n vs ~80n). Post-embed dynamics never fully stabilize.
- *activation_dmd Attn Out*: vertical-ellipse complex structure most prominent in *early* epochs, collapses to a tight cluster at (1, 0) by late training. Inverted timing relative to the other sites.
- *parameter_dmd*: not yet examined under pinned reference_epoch (use case for studying both transient frequencies in their committed windows).

### `p113/s485/ds999` — Rebounder

Test loss reaches a minimum, then rises back up.

**Signatures:**
- *Loss curve*: sharp drop at ~17k to ~1µ, then rises to ~700µ by 25k.
- *activation_dmd*: two-peak `resid_post` structure (peaks at ~17k and ~19k). Beautiful arc on the `attn_out` eigenvalue plot — green-teal points sweep from ~(0.7, 0) up through complex space and back to (1, 0). The full transit is visible.
- *activation_dmd `mlp_out`*: vertical lens is the cleanest "mode-pair commitment" signature among the anomaly variants.

### `p89/s999/ds999` — Didn't complete descent

Loss curve drops at ~14–17k but stalls before reaching the canonical floor. Settles at ~30µ–50µ.

**Signatures:**
- *activation_dmd*: distinctive **"C-shape" / parenthesis-arc patterns** at Post-Embed, Attn Out, and MLP Out — vertical arcs that almost close but don't. Geometric signature of "got close to the basin, didn't sit down in it."

---

## Cross-variant patterns surfaced

### Variants suggesting a pattern (provisional)

- **MLP leads second descent in DMD residual** — across the dense-checkpoint anomaly variants (p101/s999/ds999, p113/s485/ds999, p89/s999/ds999) plus the four reference variants, MLP_out's residual peak precedes attn_out's by 500–1000 epochs at second descent. Selection-biased to dense-checkpoint variants; needs balanced validation. Connects to existing finding that MLP Fourier alignment commits in 1–6% of training while attn FA commits at 0.50–0.86 of grokking. ([notes/findings_mlp_leads_second_descent.md](notes/findings_mlp_leads_second_descent.md))
- **W_in / W_out independent timelines** — the empirical justification for treating them as separate matrices in `parameter_dmd`. p109 group 3 (k=4) is the cleanest case: W_out drops to ~0 while W_in is at ~0.7. p113 canon, by contrast, has nearly-coordinated W_in / W_out — "canon is canon precisely because everything is coordinated." Disagreement between W_in and W_out is a feature of imperfect grokking.

### Universal patterns (across all studied variants)


- **Within-DMD multi-lens convergence**: `activation_dmd`'s mlp_out residuals and `parameter_dmd`'s per-group W_in/W_out residuals align on major training events. Magnitude differences encode event scope: localized parameter events (e.g., freq 5 abandonment) show more relative magnitude on the parameter side because they aren't diluted by averaging over uninvolved neurons. ([fieldnotes draft: dmd-at-zoom.mdx](../apps/fieldnotes/src/content/drafts/dmd-at-zoom.mdx))

---

## Data seed sensitivity

`data_seed` is *not* a tuning knob in the conventional sense — it materially affects training outcomes for any given (prime, seed) pair. Cross-variant comparisons across data seeds risk conflating data-side and model-side signal. Specific empirical observations:

- **p113 has been tested across multiple data seeds**; only `ds598` produces a healthy training trajectory. `ds42` and `ds999` are catastrophic for `p113/s999`.
- *(2026-03-21)* At epoch 0, data-seed differential pressure lands almost entirely on the MLP. Embedding and attention are near-uniform and data-seed-agnostic at init; MLP is already structured and spiky. Data seed selectively amplifies MLP's existing frequency preferences.
- **Methodological note**: don't compare structural metrics directly across data seeds. Divergence between data seeds is immediate (visible at first descent) and the differences accumulate across training; like-for-like comparison should hold `data_seed` fixed.

---

## Cross-references

- **Raw metrics** (machine-readable, per-variant):
  - `results/modulo_addition_1layer/{variant_dir}/variant_summary.json` — per-variant snapshot of loss, grokking epochs, frequency commitments, performance classification.
  - `results/modulo_addition_1layer/variant_registry.json` — compiled aggregate of all variants.
- **Analytical lens catalog**: [analysis_atlas.md](analysis_atlas.md) — what analyses we have, what they measure, where their results live.
- **Fieldnotes drafts** (research narrative; one entry per finding, longer-form):
  - `apps/fieldnotes/src/content/drafts/dmd-at-zoom.mdx` — REQ_117 narrative; per-variant eigenvalue signatures; p101 multi-lens convergence.
  - `apps/fieldnotes/src/content/drafts/architecture-ladder.mdx` — one-hot MLP vs learned-emb MLP vs transformer; geometry-from-attention finding.
- **Notes / findings** (older or more granular):
  - `docs/notes/findings_site_gradient_convergence.md`
  - `docs/notes/findings_ring_geometry.md`
  - `docs/notes/findings_intragroup_manifold.md`
  - `docs/notes/findings_saddle_transport.md`
  - `docs/notes/findings_transient_frequencies.md`
  - `docs/notes/findings_circularity_sites.md`

---

## Open questions

### Unstudied variants

Variants in the registry that are good candidates for new entries but haven't been examined deeply yet:

- **Variants under `data_seed=42`** — present in the registry but rarely examined. The "ds42 shows only 1 case with 0% homeless" observation from the transient-frequency cross-variant survey is the only concrete characterization we have.
- **Variants at primes other than 113** — p59, p89, p97, p101, p103, p107, p109, p127 are all in the registry. Only the ones listed above have been studied with multiple lenses.
- **Cross-prime training across model seeds** — needed to determine whether data seed sensitivity is universal or specific to particular (prime, seed) pairs.

When a new variant gets dense checkpointing and a deeper look, add an entry. Aim for one paragraph + 3–6 signature bullets; longer-form goes into fieldnotes or findings files.

### Unquantified per-variant observations

Observations that appear visually across variants but lack metrics — until quantified, they live in shared verbal language at risk of drifting between sessions.

#### Parameter trajectory loop closure

In the cross-epoch PCA of the parameter trajectory (specifically PC2 × PC3, projected from the 3D PC1×PC2×PC3 trajectory), the trajectory passes through a region of self-intersection where the curve at one epoch comes spatially close to the curve at a much later epoch. The 3D view reveals this as a *saddle-shaped* path — the apparent third "ghost axis" in 3D is the principal direction of the saddle's unstable manifold, which in PC2×PC3 (the saddle's stable plane) shows up as the apparent self-intersection. **What "loop closure" means precisely**: the trajectory crossed one end of the unstable axis and landed on or near the other end.

**Why we don't have it as a metric yet**: the visual is sensitive to checkpoint density, training length, and how long the model spends in each regime. PCA orientation is also arbitrary, so the apparent intersection can rotate between runs. Some quantifications (see [analysis_atlas.md](analysis_atlas.md) → Candidate Derived Metrics) are PCA-invariant; others depend on a fixed projection and would be variant-comparable only with care.

**Open question**: does loop-closure in parameter space cleanly classify variant outcome, or does it describe something subtly different about the parameter trajectory's geometry?

#### DMD-derived phase windows

The phase markers in `variant_summary.json` (`first_descent`, `plateau`, `cascade`, `second_descent`, `final`) are heuristic-derived from loss curve features. The windowed DMD residual peaks (from `activation_dmd` / `parameter_dmd`) are mathematically derived from the same trajectory data and don't depend on tuning thresholds against an external loss signal — likely a more principled basis for "what regime is the model in at epoch X." See [analysis_atlas.md](analysis_atlas.md) → Candidate Derived Metrics for what would need to be designed before this can replace the heuristic markers.
