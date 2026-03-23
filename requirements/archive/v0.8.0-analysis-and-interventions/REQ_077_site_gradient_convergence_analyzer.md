# REQ_077: Site Gradient Convergence Analyzer

**Status:** Active
**Priority:** High
**Related:** REQ_066 (Multi-Stream Specialization Trajectory), REQ_072 (Weight Trajectory Divergence), REQ_074 (Variant Outcome Registry)
**Last Updated:** 2026-03-22

## Problem Statement

The platform's existing views observe what the model has already committed to — activation patterns, weight geometry, neuron specialization. None of them can show what each computational site is currently *being pushed toward* at a given training epoch.

The site-level gradient analysis addresses this gap. By computing gradient energy at the embedding, attention, and MLP sites separately at each checkpoint, we can measure whether the three sites are pointing toward the same frequency set or diverging. This directional alignment between sites is the gradient-space signature of the "MLP proposes, attention disposes" mechanism: when sites converge on the same frequency preferences, the model is moving toward phase lock; when they diverge and fail to recover, the model is exhibiting the site-level conflict that characterizes unhealthy grokking.

The notebook prototype (`early_gradient_analysis.py`) has validated that this signal is real and diagnostic across the four models examined. It now needs to move into the analysis pipeline so any variant can be analyzed and compared.

## Conditions of Satisfaction

### Analyzer: `GradientSiteAnalyzer`

- [ ] A new analyzer `GradientSiteAnalyzer` is implemented and registered in the analysis library
- [ ] The analyzer runs post-hoc from saved checkpoints — it does not require training-time integration
- [ ] Epoch sampling is driven by window boundaries in `variant_summary.json` (first_descent, plateau, second_descent, final), plus a configurable number of interior samples per window (default: 2)
- [ ] Sampling snaps requested epochs to nearest available checkpoint using `get_available_checkpoints()`
- [ ] At each sampled epoch: loads the checkpoint, generates training data using the variant's own data seed, runs forward+backward, computes per-site per-frequency gradient energy using the same projection logic validated in the notebook (`W_E[:p]` projects each site's gradient into token space; Fourier basis then decomposes into frequency components)
- [ ] The three sites computed are: embedding (`grad_W_E[:p]` projected directly), attention (Q, K, V projected through `W_E[:p]`, combined as RMS across all heads and matrices), MLP (`W_E[:p] @ grad_W_in`)

### Artifact

- [ ] A single `.npz` artifact per variant is stored (not per epoch) since this spans the full training arc
- [ ] Artifact contains: `epochs` (n_sampled,), direction-normalized energy per site `energy_{site}` (n_sampled × n_freqs), raw total gradient magnitude per site `magnitude_{site}` (n_sampled,), pairwise cosine similarity per pair `similarity_{pair}` (n_sampled,) for the three pairs (emb↔attn, emb↔mlp, attn↔mlp)
- [ ] Direction normalization: each epoch's energy vector is normalized by its L2 norm before storage; raw magnitude (pre-normalization L2 norm) is stored separately
- [ ] Vectors with negligible norm (< 1e-30) are stored as zero vectors with magnitude = 0; cosine similarity is stored as NaN for those epochs

### View: `site_gradient_convergence`

- [ ] A new cross-epoch view `site_gradient_convergence` is registered in the view catalog
- [ ] Renders the cosine similarity trajectory over sampled epochs, one line per site pair (three lines total)
- [ ] Window boundaries from `variant_summary.json` are marked as vertical dotted lines
- [ ] A secondary panel (or optional overlay) shows the raw magnitude trajectory per site — this captures where gradient activity concentrates over training, independent of directional preference
- [ ] NaN similarity values render as gaps in the line, not zeros
- [ ] Y-axis range [0, 1] for the similarity panel

### View: `site_gradient_heatmap`

- [ ] A new cross-epoch view `site_gradient_heatmap` is registered in the view catalog
- [ ] Renders three heatmap panels (embedding / attention / MLP) sharing a y-axis (epoch)
- [ ] X-axis: frequency k; Y-axis: sampled epochs; color: direction-normalized energy
- [ ] Key frequencies from the variant's dominant frequency set are marked as vertical dashed lines
- [ ] Window boundaries are marked as horizontal dotted lines on the y-axis

### Validation

- [ ] p109/s485/ds598 (clean grokker): convergence plot shows brief dip during competition window followed by full recovery to ≥ 0.85; all three site pairs remain tightly coupled (spread < 0.15)
- [ ] p101/s999/ds598 (open loop): convergence plot shows recovery that does not reach 0.85; at least one site pair ends below 0.75
- [ ] p113/s999/ds598 (canon, diffuse): embedding↔MLP similarity ends measurably lower than attention↔MLP, reflecting the documented pair divergence pattern
- [ ] Notebook cell demonstrates both views for at least one healthy and one anomalous variant

## Constraints

**Must:**
- Use the variant's own training data (via `generate_training_dataset()`) at each checkpoint — this captures the accumulated effect of the data seed on gradient pressure, not just the weights alone
- Sample epochs from `variant_summary.json` window boundaries — sampled epochs must be semantically anchored to the model's training dynamics, not hardcoded
- Store both direction-normalized energy and raw magnitude — the direction is the primary diagnostic signal; magnitude encodes whether a site is active or quiescent at a given epoch

**Must not:**
- Require dense checkpoints — the analyzer must work with the existing checkpoint schedule (typically 100-epoch spacing during key windows)
- Run during training — post-hoc only
- Merge the attention Q/K/V matrices into a single pre-aggregated artifact field — store the combined result, but document that Q, K, V are combined as RMS across all heads and three matrices, so future decomposition remains possible

**Decision authority:**
- **Resolved:** Single artifact per variant, not per epoch. This is a training-arc summary and does not fit the per-epoch artifact pattern used by other analyzers. ArtifactLoader conventions may need a minor extension for non-epoch-keyed artifacts — Claude decides the storage path and loading interface.
- **Resolved:** Direction-normalized energy as primary storage. The 7-order-of-magnitude collapse in gradient magnitude between epoch 0 and post-grokking makes raw energy unviable for visualization without normalization. Raw magnitude is stored as a scalar summary, not as a per-frequency vector.
- **Resolved:** Variant's own training data used at each checkpoint. Alternative (fixed evaluation probe across all epochs) would conflate weight evolution with data-gradient interaction — the variant's own data preserves the accumulated training trajectory.
- **Claude decides:** Artifact file naming and storage path (within the convention that non-epoch-keyed artifacts live under `artifacts/` in the variant dir); whether the magnitude secondary panel is a subplot or an optional overlay on the convergence plot

## Context

This requirement implements the pipeline version of analysis developed in `notebooks/early_gradient_analysis.py` (the multi-site sweep sections added 2026-03-22). The core finding driving this work is documented in `notes/findings.md` under "2026-03-22: MLP Proposes, Attention Disposes — Phase Lock as the Grokking Mechanism."

The convergence plot is the primary diagnostic. Healthy grokking shows a characteristic shape: high initial similarity (all sites uniform at init), dip during competition (sites develop idiosyncratic frequency preferences), recovery after second descent (phase lock achieved). Unhealthy grokking shows one of three failure signatures:
- **No recovery** (open loop, p101/s999): sites diverge and remain diverged
- **Pair divergence** (diffuse specialization, p113/s999/ds598): pairs separate — embedding↔MLP ends lower than attention↔MLP, indicating MLP has diverged from the embedding-attention system
- **Progressive degradation** (rebounder, ds999): similarity declines monotonically with no recovery phase

The heatmap view provides the mechanistic detail behind the convergence signal — which frequencies each site is emphasizing at each epoch, and whether the emphasis patterns shift together or independently.

The magnitude secondary signal addresses the open question of whether gradient quiescence (near-zero magnitude) at a site after grokking is uniform across sites or site-specific. A site that goes quiescent earlier may have "finished its job" before the others.

## Notes

- The prototype uses ALL_DATA_SEEDS for comparison within a single figure. The pipeline version is per-variant (one variant → one artifact → one set of views). Cross-seed comparison belongs in the peer comparison page (REQ_076) or a future cross-seed analysis view.
- The Q/K/V aggregation (RMS across all heads and three matrices) conflates individual head contributions. If per-head decomposition becomes research-relevant, it would require a separate artifact. Flag this as a future extension rather than blocking this requirement.
- p109/s485 consistently shows the cleanest geometry across all analysis lenses (neuron specialization, centroid geometry, now gradient convergence). Its training speed may be mechanistically related — a model where all three sites rapidly agree on the same frequency set would converge faster. This is worth testing once the pipeline is in place.
