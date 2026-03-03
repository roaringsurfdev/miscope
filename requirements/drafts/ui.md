### Implementation Notes (from discussion 2026-02-15)

**Framing: Lenses, not reports.** A lens is an opinionated analytical perspective on the same underlying data. Each lens determines arrangement, filtering, and juxtaposition. You switch lenses depending on what question you're asking. A small number of well-designed lenses, not an open-ended canvas.

**Approach: New Dash page → lens infrastructure.**
1. First implementation: `/summary` page in dashboard_v2 that arranges existing renderers in the dense layout described above. Doesn't disrupt current analysis page.
2. Design the page so it could evolve toward a general lens framework (shared variant selector, shared layout primitives, lens-specific arrangement).
3. Starred (*) items are visualizations that don't exist yet (embedding/unembedding specialization, unembedding Fourier). These are dependencies — the lens can ship without them and gain them as analyzers are built.

**Static export as quick win.** The existing `export_variant_visualization()` infrastructure can generate a Summary Lens as an HTML file or image grid per variant with minimal work. Useful immediately for research alongside the dashboard implementation.

---

REQ_??? - Cross-Variant Lens (same family)

We're seeing a lot of common structures show up across multiple model variants, but we're also seeing some outliers. It would be helpful to be able to compare the models within a model family to support more rigorous comparison (as opposed to visual inspection).

One of the challenges is that there are multiple ways to compare the models. Some core questions might help guide us:
- *When* do models grok (how early/late in training)
- *How specialized* do parts of the network get:
    - how concentrated are the frequencies in embeddings, neurons, attention heads
    - do models share an effective dimensionality threshold that marks the transition from memorization to generalization?
- *How similar* are the paths through weight space during training? (Here is where I would expect to start seeing a "manifold channel" show up...successful models have paths that stay within the channel, unsuccessful models stray in some meaningful way(s))

### Implementation Notes (from discussion 2026-02-15)

**Scope: 3 cross-variant visualizations to start.**

1. **Overlaid test loss curves** — all variants in a family on one plot. Simplest cross-variant visualization, immediately highlights timing outliers. Uses existing metadata (train_losses/test_losses from metadata.json).

2. **Variant summary table** — key metrics at final epoch for each variant in a family. Candidates: grokking epoch, final test loss, specialized neuron count, frequency band count, attention head max variance, convergence status (plateaued vs still moving). Provides the "variant health dashboard" view.

3. **Grokking-aligned comparison** — normalize the time axis to grokking onset, then overlay specialization trajectories. Shows whether models follow the same *relative* timeline even if absolute timing differs.

**"How similar are the paths" — deferred but scoped.**
- Geometry-invariant metrics (loop closure, path curvature, self-intersection) are the right starting point
- These don't require a shared PCA basis — they characterize trajectory shape independently
- SueYeon Chung's work on neural manifold geometry is directly relevant to formalizing these metrics. Her colleague may have tractable computation methods worth investigating (research task, not engineering task yet).
- Implementation: compute scalar geometry metrics per variant, visualize as a comparison (small multiples, table, or scatter of metric pairs)

**Anomalous variants identified so far (motivation for this lens):**

| Variant | Anomaly Type | Key Signature |
|---------|-------------|---------------|
| p101/999 | Degenerate Fourier solution | Open PC2/PC3 loop, 6:1 cos/sin imbalance |
| p107/999 | Post-grokking instability | All heads converge to freq 22, then unstable diversification |
| p59/485 | Incomplete grokking | Open PC2/PC3 loop, missing frequency band, gradient-poor path |

All 3 would be immediately identifiable with the proposed visualizations.
