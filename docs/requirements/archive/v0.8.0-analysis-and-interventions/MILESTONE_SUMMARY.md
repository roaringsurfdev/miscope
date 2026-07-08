# Milestone v0.8.0 — Analysis Suite & Interventions

**Released:** 2026-03-23
**Requirements archived:** 20 (REQ_052 through REQ_077, excluding REQ_053, 055, 073, 075)

---

## Major Themes

### 1. Multi-Stream Analysis (REQ_066, REQ_072)
- `multi_stream_specialization` view: 4-panel trajectory (MLP neurons / attention aggregate / embedding dims / effective dimensionality)
- Weight trajectory divergence views: per-group and per-weight-matrix PCA trajectories, component velocity comparison
- Group trajectory overlay (`parameters.pca.group_overlay`): normalized overlay of embedding/attention/MLP trajectories on shared PC axes with sign-flip correction
- Trajectory proximity view (`parameters.pca.proximity`): pairwise L2 distances over training, epoch cursor enabled

### 2. Site Gradient Convergence (REQ_077)
- `GradientSiteAnalyzer`: post-hoc, checkpoint-based per-site per-frequency gradient energy computation
- Sites: embedding (grad_W_E[:p] direct), attention (Q/K/V projected through W_E[:p], RMS across heads), MLP (W_E[:p] @ grad_W_in)
- Single `.npz` artifact per variant: direction-normalized energy, raw magnitude, pairwise cosine similarity
- Views: `site_gradient_convergence` (similarity trajectory + magnitude panel), `site_gradient_heatmap` (3-panel per-site frequency heatmap)
- Window boundaries from `variant_summary.json` marked on all plots

### 3. Intervention Architecture (REQ_067, REQ_068, REQ_070, REQ_071)
- `InterventionVariant` subclass nested under parent variant: `results/{family}/{parent}/interventions/{label}/`
- `Variant.create_intervention_variant()` factory; `Variant.interventions` discovers sub-variants from filesystem
- `FrequencyGainHook`: hook on `hook_attn_out`, W_E-based frequency directions
- Intervention Check dashboard page: family→variant→intervention→epoch hierarchy, hook verification display
- Key finding: hook correctly targets frequencies; torus rotation is genuine; snap-back begins at epoch 5800

### 4. Variant Registry & Peer Comparison (REQ_074, REQ_076, REQ_057, REQ_065)
- `variant_summary.json` per variant: loss min/final, grokking window markers, frequency gains/losses, performance classification
- `variant_registry.json`: compiled aggregate across all variants
- Peer comparison dashboard page: cross-variant view with shared axes
- Second descent diagnostics surfaced via registry fields

### 5. Analytical Views Completed (REQ_052, REQ_056, REQ_058, REQ_060, REQ_062, REQ_063, REQ_064)
- Fourier frequency quality scoring view
- Frequency specialization sequencing view
- Neuron band concentration health metric
- Neuron dynamics runtime threshold configuration
- View availability enforcement per variant state
- Fourier nucleation predictor view
- Data compatibility analyzer

### 6. Infrastructure (REQ_054, REQ_061)
- DataView catalog: `variant.at(epoch).dataview(name)` pattern for tabular/structured data views
- Data seed as domain parameter: `data_seed` field in variant params, multi-seed variants supported throughout

---

## Key Architectural Decisions

- **Single artifact per variant for gradient convergence** — training-arc summary does not fit per-epoch pattern; stored as `artifacts/gradient_site/gradient_site.npz`
- **InterventionVariant nested under parent** — interventions are sub-variants, not separate family members; parent carries family reference
- **ArtifactLoader extended with `load_variant_artifact()`** — non-epoch-keyed artifacts loaded by name, distinct from cross-epoch loader
- **Sign-flip correction in proximity renderer** — PCA orientation is arbitrary; `min(dist(A,B), dist(A,-B))` prevents mirrored trajectories from reading as diverged

---

## Active Requirements (not archived)
- REQ_053 — Frequency Quality & Class Error Analysis (may bundle into REQ_075)
- REQ_055 — Attention Head Phase Analysis
- REQ_073 — Weight-Space DMD (exploratory work done, more training windows needed)
- REQ_075 — Per-Input Prediction Trace (next focus)
