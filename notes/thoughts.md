# Thoughts

Unstructured parking lot for ideas and observations. See Claude.md for context.

---

# Analysis Findings

Insights from exploratory analysis that may inform interpretation or future investigation.

---

## 2026-02-02: Multi-Scale Structure in Activations and Attention

### Discovery

When viewing neuron activation heatmaps at reduced resolution (via bilinear interpolation to 28×28), distinct "blob" regions emerge that aren't visible at full resolution (113×113). This suggests activation patterns have structure at multiple spatial scales.

### Key Observations

**Neuron activations** (first 10 neurons analyzed):
- Most neurons (1-7, 9) have energy concentrated in high frequencies (FFT no-DC ~0.005)
- Neuron 8 is an outlier with genuinely coarse structure (FFT no-DC = 0.962)
- Neuron 0 has moderate low-freq content after DC removal (0.226)

**Attention heads** show distinct specialization patterns:

| Head | FFT no-DC | Gradient | Structure Type |
|------|-----------|----------|----------------|
| 0 | 0.014 | 0.324 | High-freq content, variable intensity |
| 1 | 0.349 | 0.241 | Mixed frequencies |
| 2 | 0.969 | 0.094 | Low-freq, smooth uniform regions |
| 3 | 0.013 | 0.334 | High-freq content, most edge variation |

### Nuanced Interpretation

Initial framing of "blob vs grid" was imprecise. The metrics reveal different structural properties:

- **FFT (no DC)**: Captures spatial frequency of the dominant pattern
  - High = large-scale/low-frequency structure (Head 2's big uniform squares)
  - Low = fine-grained/high-frequency structure

- **Gradient magnitude**: Captures local intensity variation / edge density
  - High = more edges, more intensity variation (Heads 0, 3)
  - Low = smoother, more uniform regions (Head 2)

Head 2's "checkerboard" pattern is actually large flat regions (low gradient) at low spatial frequency (high FFT). Heads 0 and 3 have more continuous intensity variation creating higher gradients despite lower FFT scores.

### Open Questions

1. Does this multi-scale structure emerge during training, or is it present from initialization?
2. Do low-frequency patterns (like Head 2) converge before or after high-frequency patterns?
3. What is the functional role of heads with different scale characteristics?
4. Are there neuron clusters that share similar scale profiles?

### Experimental Validation

Working prototype cells in `ModuloAdditionRefactored.py`:
- Bilinear downsampling experiments (lines 295-334)
- Coarseness metrics: FFT energy ratio, variance preservation, gradient magnitude (lines 340-529)

Screenshots archived in `temp_artifacts/` (session 2026-02-02).

---

## 2026-02-02: Quantitative Metrics for Structural Analysis

### Metrics Tested

**1. FFT Low-Frequency Energy Ratio**
```python
fft_2d = torch.fft.fft2(activation_map)
power = torch.abs(torch.fft.fftshift(fft_2d)) ** 2
# With circular mask for low frequencies (radius = 12.5% of spectrum)
coarseness = power[low_freq_mask].sum() / power.sum()
```
- **Issue**: DC component (mean) dominates the ratio
- **Solution**: Zero out center pixel before computing ratio

**2. Variance Preservation**
```python
downsampled = F.interpolate(map, size=(14, 14), mode='bilinear')
upsampled = F.interpolate(downsampled, size=(p, p), mode='bilinear')
score = upsampled.var() / original_var
```
- Captures how much structure survives aggressive downsampling
- Correlates with FFT metric but provides different perspective

**3. Gradient Magnitude**
```python
dx = activation_map[:, 1:] - activation_map[:, :-1]
dy = activation_map[1:, :] - activation_map[:-1, :]
grad = (dx.abs().mean() + dy.abs().mean()) / 2
```
- Best at distinguishing "smooth uniform regions" vs "variable intensity"
- Complements FFT metric - captures edge density rather than frequency content

### Recommended Combination

For training dynamics analysis, track both:
- **FFT no-DC**: Spatial frequency of dominant patterns
- **Gradient magnitude**: Local intensity variation / smoothness

These capture orthogonal properties and together give a richer picture of structural evolution.

---

# Workbench Improvement Ideas

Suggestions for future enhancements to the Training Dynamics Workbench.

---

## 2026-02-02: Architectural Pressures from Multi-Scale Visualization (REQ_019)

### 1. Data Reuse Between Analyzers

**Observation**: Multiple analyzers need the same underlying data (neuron activations, attention patterns) but currently each analyzer independently loads checkpoints and runs forward passes.

**Specific case**: Both `NeuronActivationsAnalyzer` and a future `DownsampledActivationsAnalyzer` need `probe_cache["post", 0, "mlp"][:, -1, :]`. The downsampled version is just a transform of the same data.

**Potential approaches**:
- Shared "activation extraction" step that multiple analyzers consume
- Hierarchical analyzer pattern where transforms (like downsampling) operate on base artifacts
- Lazy evaluation / caching at the probe_cache level

**Trade-off**: More complexity vs. avoiding redundant computation. May not matter for single-checkpoint analysis but becomes relevant for full training run analysis across many checkpoints.

### 2. Viewing More Neurons

**Observation**: Showing only 5 neurons is arbitrary and limiting. The multi-scale visualization revealed that different neurons have different scale characteristics. Seeing more neurons would help identify patterns.

**Potential approaches**:
- **Paginated/scrollable**: Show N neurons at a time with navigation controls
- **Summary view**: Cluster neurons by scale characteristics and show representatives
- **Configurable N**: Let user choose how many to display (with performance warnings for large N)
- **Small multiples**: Dense grid of tiny heatmaps showing all 512 neurons (patterns visible even if individual details aren't)
- **Filtering**: Show only neurons matching certain criteria (e.g., "neurons where blob structure dominates")

**Related**: The "Neuron Frac Explained by Freq" heatmap already shows all neurons on one axis. Could use this as a selection/filtering mechanism.

### 3. Attention Head Specialization Labels

**Observation**: Different attention heads appear to specialize at different scales. This could be surfaced in the UI.

**Potential approaches**:
- Auto-compute scale metrics and display as labels/badges
- Group heads by scale characteristics in visualizations
- Color-code heads by their structural type

---

*These are observations for future consideration, not immediate action items.*

---

## 2026-02-03: Analysis Report Expansion Ideas

Brainstormed directions for expanding analysis report functionality:

### 1. Report Generation
Structured output summarizing analysis findings in shareable formats (markdown, HTML, PDF). Could include:
- Summary statistics per checkpoint
- Key observations (e.g., "grokking detected at epoch X")
- Embedded visualizations or links to interactive versions

### 2. Comparison Reports
Diff analysis between variants with different parameters. Examples:
- Compare p=113 vs p=97 modulus variants
- Compare different random seeds to identify consistent vs seed-dependent behaviors
- Side-by-side visualization grids

### 3. Aggregate Metrics
Summary statistics or derived insights across checkpoints:
- When does grokking occur? (test loss inflection point)
- Which frequencies dominate at convergence?
- Neuron specialization timeline

### 4. Export Formats
Packaging analysis artifacts in shareable formats:
- Bundled artifact archives for reproducibility
- Standalone HTML reports with embedded data
- Integration with external tools (notebooks, papers)

---

*These ideas emerged from discussion about analysis report requirements.*

---

## 2026-02-10: Post-v0.3.1 Discussion — Future Directions

Three threads emerged from discussion after the v0.3.1 release. Ordered by estimated readiness.

### 1. Turn Detection (Near Requirement-Ready)

**Problem**: Grokking onset ("the turn") is currently identified by visual inspection. Quantitative detection would enable:
- Automated identification of when grokking begins
- Dynamic checkpoint scheduling (uniform spacing by default, then retroactively densify around the turn)
- Downstream analytics that key off the turn epoch

**Observations**:
- The turn appears to precede all other meaningful structural events (frequency specialization, attention head differentiation, trajectory curvature)
- The existing system already handles non-uniform checkpoint schedules gracefully — adding dense checkpoints to a variant and re-running analysis "just worked"
- Candidate signals: test loss inflection, parameter velocity spike, trajectory curvature change

**Scoping notes**:
- User suggests keeping initial scope isolated: detect the turn, report the epoch, done
- Dynamic checkpoint scheduling and downstream analytics would be separate follow-on requirements
- Could start with test loss curvature (simplest signal, doesn't require new analyzers)

### 2. Dashboard UI Redesign — Observed Friction Points

**Context**: Dashboard was designed for 4-5 visualizations, now has 18. User is not frustrated but recognizes cognitive load is growing and the UI is approaching its limits. "I get *lost* in analysis. I love it."

#### Observed Workflows

**A. Epoch discovery loop** (high friction):
1. Scroll down to a summary visualization
2. Mouse over an interesting area to find a good epoch
3. Scroll back to the top
4. Mouse over train/test curves to find the right slider index
5. Type in the index
6. Scroll back down to see the result

User has memorized some indices but can't always remember which variant they belong to (e.g., is index 71 important for 101/999 or 113/999?).

**B. Neuron discovery** (moderate friction):
1. Look at Neuron Frequency Specialization (clusters) plot
2. Mouse over a yellow block to find the neuron ID
3. Navigate to Neuron Index slider input box
4. Enter the neuron ID
5. Scroll to neuron heatmap

**C. Two modes of use**:
- **Summary mode**: Scrolling the full page to find and compare summary information across visualizations
- **Deep dive mode**: Focused on one visualization (especially neuron heatmaps), using frequency clusters to identify neurons of interest

#### Specific Pain Points

1. **Top controls steal vertical space**: Variant selector, epoch slider, neuron index, and other inputs at the top push all content down, requiring constant scrolling
2. **Epoch cross-referencing**: Summary plots show the interesting epoch but don't connect to the epoch selector
3. **Neuron cross-referencing**: Cluster plots show the interesting neuron but don't connect to the neuron selector
4. **Mixed layout**: Summary and detail visualizations interleaved in one long scroll — serves neither mode well

#### Design Patterns Implied

- **Click-to-navigate**: Click a point on any summary plot → epoch slider updates. Click a neuron in clusters → neuron index updates. Eliminates the discovery-to-input indirection.
- **Collapsible left sidebar**: Global controls (variant, epoch, neuron index) in a toggleable left nav instead of top controls. Always accessible without stealing plot space.
- **Summary vs detail modes**: Summary dashboard (compact, all trajectories) vs detail view (one visualization large with controls). Match the two actual usage patterns.
- **Drill from summary** as general principle: Every summary view supports drilling into detail. This is the unifying interaction model.

#### Key Constraints (from user feedback on initial design ideas)

1. **No category tabs for summary scanning.** Dividing visualizations into category tabs and requiring click-through would feel like regression. The current "everything on one page" has value for summary scanning — tabs would fragment that workflow. Any solution must preserve the ability to scan summaries without tab-switching.

2. **Click-to-navigate has dual intent.** Sometimes a click means "navigate to this epoch" (temporal navigation). Sometimes it means "navigate to a deep dive" (structural navigation — e.g., go to neuron heatmap). The interaction model must handle both, and it won't always be obvious which the user wants. This may resolve naturally by visualization type (trajectory click = epoch; cluster click = neuron), but it's a real design tension to watch.

3. **Some summary plots are navigation elements for their adjacent details.** Example: Neuron Frequency Specialization clusters is used to find interesting neurons, then the user clicks through several neurons from the same band to compare their heatmaps. Separating the cluster plot from the heatmap would break this workflow. Summary and detail are sometimes *paired*, not separate modes. This argues against a clean summary/detail split and toward *groups* where each group has its own summary-to-detail drill-down.

#### Performance

- All 18 plots re-render on every epoch slider change
- Lazy rendering (only render visible/active plots) would address this
- Tabbed or collapsible sections would naturally limit what renders

#### Framework Decision: Gradio → Dash

Assessed Gradio's ceiling against emerging requirements:
- **No sidebar layout**: Linear rows/columns, no fixed-position panels during scroll
- **No figure patching**: Every update serializes entire figure to browser
- **Limited click data**: `.select()` returns basic coords, no trace/point identity
- **No client-side callbacks**: Every interaction round-trips to server

Dash provides native solutions for all four. Migration cost is bounded — only `dashboard/app.py` and `dashboard/state.py` are framework-specific. Renderers return `go.Figure` objects that work identically in both.

**Strategy**: Freeze Gradio dashboard (stays available for ongoing analysis), build Dash app in `dashboard_v2/` incrementally. Spike first (5-6 visualizations to prove patterns), then full migration. Training tab stays in Gradio (no interaction problems there).

### 3. Cross-Variant Comparison (Biggest Gap)

**Problem**: The workbench currently shows one variant at a time. Comparing variants requires:
- Manually switching between variants in the dashboard
- Holding differences in memory
- Or exporting images and comparing side by side

**Why it matters**:
- Core scientific question is "how do different parameters affect training dynamics?"
- 10 model variants exist (different primes, seeds) and comparing them is the whole point
- The p=101/seed=999 anomaly was identified through laborious manual comparison

**User's preferred approach**: Start with summary statistics comparison across variants, then drill into details from there. The "drill from summary" pattern applies here too — it's how they'd want to work generally.

**Potential approaches** (not yet scoped):
- Side-by-side dashboard panels (two variants, same visualization)
- Overlay mode for trajectory plots (multiple variants on same axes)
- Diff/delta views showing quantitative differences between variants
- Summary tables aggregating key metrics across all variants

**Dependencies**: May benefit from turn detection first (align comparisons to training phase rather than raw epoch number).

### 4. Notebook Prototyping Workflow

**Observation**: The `export_variant_visualization()` function and renderer library enable a new workflow:
- Prototype new visualizations in notebooks
- User reviews rendered output
- Iterate on design before committing to dashboard integration

This is a useful side effect of REQ_033, not a separate requirement. Worth noting as a development pattern.

---

*These threads are candidates for future requirements. Turn detection is closest to ready.*

---

## 2026-02-11: Analysis Coverage — Unembeddings, Logits, and Probes

### Current Coverage Gaps

The analysis pipeline currently covers: model weights, embeddings, attention heads, and neurons. Missing coverage: **logits** and **unembeddings**.

### 1. Unembedding Fourier Analysis (New Analyzer)

Pair with existing `dominant_frequencies` (embedding) analyzer. Both are weight-based — extract `W_U`, compute Fourier coefficients, store them. Side-by-side visualization of embedding vs unembedding Fourier structure across training would show how these complementary representations co-evolve during grokking.

**Status**: Ready to scope as a requirement.

### 2. Unembedding PCA (Front-End Add)

The `parameter_pca` analyzer already saves out weight trajectories. Adding unembedding to PCA visualization is likely a front-end/renderer change — the data may already be available depending on which weight matrices are currently tracked. Needs verification of what `parameter_pca` currently captures.

**Status**: Verify current PCA coverage, then likely a small front-end task.

### 3. Logit Lens / Per-Input Analysis

Logit lens is most meaningful per-input (or small batch) rather than across the full p² grid. For p=113, the full grid produces ~1.4M logit values per intermediate position per epoch — too much to store as bulk artifacts, and not directly visualizable.

This connects to a broader design question: **support for additional/custom probes**. The current architecture enforces one canonical probe (full input grid) per variant. Per-input investigation is currently best served by notebook exploration (load checkpoint, `run_with_cache`, inspect).

Adding first-class support for custom probes would enable:
- Logit lens on specific inputs or small batches
- Targeted activation tracing (e.g., "how does the model handle 4+29 across training?")
- Probe subsets (e.g., only inputs where a < b)

**Status**: Brewing. UI design pressure points and probe architecture need further thought before scoping.

---

*Unembedding Fourier is ready to scope. PCA coverage needs verification. Logits/probes need more design thinking.*

---

## 2026-02-11: Per-Input Trace Analysis — Core Research Direction

### The Central Question

**"What does the model's full computation look like for a specific input, and how does that change across training?"**

This is an orthogonal axis to existing analysis. Current analyzers answer "what does [aggregate property] look like at epoch N?" Per-input tracing asks "what does the model's computation look like for *this input* at epoch N?" It cuts across all existing analyzers rather than fitting into one.

### Key Framing: Trace + Context

A single input trace gains meaning from the surrounding context of what we already know about that epoch. It's the difference between "here's a trace" and "here's a trace, and I already know this neuron just started specializing on frequency k, and this attention head just differentiated, and we're 50 epochs past the turn."

The trace becomes a case study that either confirms or challenges the macro-level story the existing analysis tells.

### Why Modulo Addition Is the Right Starting Point

The model's simplicity is an asset. With modular addition, the correct algorithm (Fourier multiplication) is known. So the question isn't just "what is the model doing?" but "how close is what the model is doing to the algorithm we know it converges to?" This is a luxury not available with larger models where the target function is opaque.

### Research Questions

- **Train vs test input comparison**: During memorization, model has seen input A but never input B. After the turn, both work. What does the internal representation shift look like — from "I've memorized that 4+29=33" to "I know how modular addition works"?
- **Trace through the turn**: For a test set input, at what point in the residual stream does the correct answer start appearing? How does this relate to when neurons and attention heads show specialization?
- **Memorization signatures**: Can we identify what "memorized computation" looks like vs "generalized computation" in the trace?

### Longer-Term Vision: Superposition and Sub-Network Evolution

For models that exhibit superposition, per-input tracing across training could help "pull apart" sub-network evolution and collapse. If you can watch sub-networks form, specialize, and potentially collapse into superposed representations over the course of training, that could help identify where superposition lives in a fully trained model. The training dynamics reveal structure that's invisible in a static snapshot.

This is the same question — "what does this specific computation look like, and how does it change?" — applied to increasingly complex models where the answer is less obvious.

### Analyzer Categories (Emerging Pattern)

Analyzers naturally cluster by what they consume, even though the interface is uniform:

| Category | Inputs Used | Examples |
|----------|-------------|---------|
| Weight-space | `model` only | dominant_frequencies, parameter_pca, future unembedding Fourier |
| Activation-based | `cache` | neuron_activations, freq_clusters |
| Attention-based | specific cache entries | attention pattern analyzers |
| Output-based | logits/residual stream | future logit lens, trace analysis |

Not an argument for changing the pipeline interface (uniform signature works), but informs:
- **Reports layer**: which analyzers group together in a view
- **Pipeline optimization**: weight-space analyzers don't need a forward pass
- **Trace analysis**: how per-input analysis relates to grid-level analysis

### Application Sub-Domains (Roadmap Vision)

The application usage breaks into 4 sub-domains:

1. **Train** (async job management): List/edit variants, kick off/monitor training runs, generate checkpoints + weight/loss summaries
2. **Analyze** (async job management): Configure analyzers and probes per variant, view analysis gaps, kick off/monitor runs, generate per-epoch + summary artifacts
3. **Reports/Dashboards** (data consumption): Standard navigation (epoch, variant, probe), summary-to-deep-dive drilling, cross-variant comparison, **user-created report views specific to research questions** (key insight — shifts from fixed layout to report builder)
4. **Ad-Hoc Analysis** (notebook/API): Load checkpoints or artifacts directly, prototype visualizations, one-off exploration

Training and Analysis share similar UX patterns (configure → run → monitor → results). Separating them from Reports lets the consumption layer focus entirely on data without carrying job management weight. The report builder concept solves the scaling pressure from the current 18-visualization dashboard.

### Status

Notebook-first exploration is the right next step for per-input tracing. The research questions are still forming — premature infrastructure would constrain exploration. Once patterns emerge from hands-on investigation, what to systematize will be clearer.

---

*Per-input trace analysis is the core research direction. Start in notebooks, systematize later.*

---

## 2026-02-11: PC2 vs PC3 Loop Structure and p101/999 Anomaly

### Universal Self-Intersecting Loop

Across all 11 variants with parameter_snapshot data, the PC2 vs PC3 trajectory forms a **self-intersecting loop** — the model passes through roughly the same region of PC2/PC3 space during early training and again during the grokking transition. This is consistent across all primes (97, 101, 103, 109, 113, 127) and both seeds (485, 999).

**One exception: p101/seed999.** Its trajectory never self-intersects. The early-epoch excursion goes so far (PC2 ≈ 30, PC3 ≈ -15) that the return path doesn't reach the original neighborhood. The loop stays open.

**Caveat:** Each variant's PCA is computed independently, so loop "size" isn't directly comparable across variants. To rigorously compare, would need shared PCA basis or geometry-invariant metrics (e.g., loop area normalized by path length).

### p101/999: Coherent Anomaly Across All Analyses

p101/999 deviates from the other 10 variants on every dimension examined:

| Dimension | Normal variants (10/11) | p101/999 |
|-----------|------------------------|----------|
| PC2/PC3 trajectory | Self-intersecting loop | Open loop — never self-intersects |
| Grokking speed | 3-5K epochs (turn to convergence) | >10K epochs |
| Frequency specialization | Clean neuron clusters | Many neurons still mixed |
| Frequency concentration | Balanced across frequencies | Unusually concentrated in high frequencies |
| Embedding balance | cos(k)/sin(k) roughly comparable | cos 13 = 2.2957, sin 13 = 0.3573 (~6:1 ratio) |
| Frequency evolution | Progressive specialization (acquiring structure) | Lost a low frequency (shedding structure) |
| Training duration | 25K epochs | Extended to 35K epochs |

The cos/sin imbalance is particularly significant. The Fourier algorithm for modular addition requires both cos(k) and sin(k) to compute the full rotation. A 6:1 ratio suggests a partial/degenerate solution — the model is using frequency 13 but not with full rotational structure.

**Key observation:** p101/seed485 looks normal — clean loop, normal grokking. Same prime, different seed. This rules out something inherent to p=101 as the cause. The specific random initialization sent p101/999 on a trajectory that overshoots the generalizing neighborhood.

### Hypothesis: Trajectory Geometry Determines Grokking Quality

The pattern suggests:
1. All models initially pass near a generalizing manifold during early training (first dip in PC3)
2. The optimizer's momentum carries them past it
3. Models that don't go too far eventually find their way back (self-intersecting loop) and converge to a clean Fourier solution
4. p101/999 went too far and never came back — converging instead to a degenerate partial solution

**Testable predictions:**
- **LR scheduling on p101/999**: Slowing learning rate during early excursion should keep the model closer to the generalizing neighborhood. If it groks faster and develops clean specialization, this confirms trajectory geometry drives grokking quality.
- **LR scheduling on a normal variant (e.g., 113/999)**: Slowing at epoch 200 (first dip) — does the model settle into the first basin? If so, the generalizing solution was accessible early.

### Per-Input Tracing as Diagnostic

Per-input traces through the *existing* p101/999 model could reveal what the partial/degenerate solution actually computes:
- Does it get the right answer for some inputs but not others?
- Are the errors structured (e.g., wrong for inputs that require the weak sin component)?
- How does the trace differ from a clean-grokking model at the same training stage?

This would connect the weight-space pathology (imbalanced cos/sin, poor specialization) to computational pathology (what the model actually does wrong on specific inputs). The trace becomes a bridge between the macro-level trajectory analysis and the micro-level computation.

### Hyperparameter Scheduling — Design Considerations

LR scheduling is standard practice in ML training (`torch.optim.lr_scheduler`). Supporting it in the workbench would require:
- Extending variant naming: e.g., `_schedule1` suffix (opaque label mapping to defined schedule config)
- Schedule is a *policy* (step decay, cosine, warmup+decay) with its own parameters — qualitatively different from scalar domain parameters like prime/seed
- Could be modeled as a **training config override** rather than a domain parameter
- Near-term: ad-hoc experiment (notebook + manual directory) to test the hypothesis
- Longer-term: formal support if scheduling experiments become routine

**Status**: Ad-hoc experiment first. Formalize if the hypothesis proves productive.

---

*The self-intersecting loop is universal (10/11 variants). p101/999's open loop correlates with every observed pathology. LR scheduling and per-input tracing are the natural next investigations.*

---

## 2026-02-11: Database Selection and Integration Timing

### Current State: Filesystem + JSON

Working well for:
- Large binary artifacts (safetensors, .npz) — these stay on filesystem regardless
- Directory hierarchy encoding relationships (family → variant → artifacts → analyzer → epoch)
- Discoverability and debuggability (human-readable structure)
- Single user, single machine, no concurrency issues

### When to Introduce SQLite

Not yet, but likely soon. The forcing functions (in rough priority order):

1. **Async job management** — tracking training/analysis run status, history, and progress. This is the most painful to do with JSON files and the most likely first trigger.
2. **Multiple probes per variant** — adds a dimension to artifact organization that directory conventions don't accommodate cleanly.
3. **Hyperparameter scheduling** — named schedule configs (e.g., `schedule1`) that map to policy definitions. Database as the natural home for "named configurations" with complex structure.
4. **Cross-variant metadata queries** — "which variants have grokked by epoch 10K?", "compare grokking speed across primes." Currently requires scanning multiple directories.
5. **Domain parameter sets** — broader collections of parameters (e.g., `domain_set_1`) as named, queryable records.

### Architecture: Database + Filesystem Coexistence

- **Database manages**: metadata, relationships, configurations, job state, named parameter sets, probe definitions, report/dashboard configurations
- **Filesystem manages**: heavy binary data (checkpoints, artifacts, exported visualizations)
- **Database indexes what's on disk** — it doesn't replace the filesystem, it makes it queryable
- The current filesystem structure already implies most of the schema (families, variants, analyzers, epochs, probes are all existing domain concepts)

### Why SQLite

- No server to manage (single file)
- Python built-in support (`sqlite3`)
- Sufficient for single-user local workbench
- Easy to back up (it's just a file alongside the other files)
- Migration path to PostgreSQL exists if the project ever needs multi-user or cloud deployment

### Risk: Drift Between Database and Filesystem

The primary risk of adding a database is state drift — database says one thing, filesystem says another. Mitigation strategies:
- Filesystem remains source of truth for artifact existence (database is an index, not the authority)
- Database is authoritative only for things that don't exist on disk (job state, configurations, named parameter sets)
- Clear ownership boundaries: if it's bytes on disk, filesystem owns it; if it's metadata or config, database owns it

### Status

Wait for a concrete forcing function (likely async job management or multi-probe support). When it arrives, the migration is straightforward — wrapping existing domain relationships in tables, not redesigning anything.

---

*SQLite is the right choice. Wait for the forcing function. Filesystem stays authoritative for heavy data; database owns metadata and configuration.*

---

## 2026-02-14: p107 Variants — New Outlier and Late Grokker

### p107/seed999: Delayed Attention Diversification

Trained and analyzed via the new Dash Training/Analysis Run pages (REQ_040).

**Timeline**:
1. Attention heads converge to frequency 22 by ~12.5K epochs (after the test loss drop / grokking onset)
2. Heads begin diversifying around 15K
3. By 22K, the specialization chart looks "wild" — unstable diversification
4. By 25K: Head 0 has specialized on frequency 51, other 3 heads still prefer frequency 22

**Other observations**:
- Weak MLP specialization on frequency 45; overall MLP range between frequencies 22 and 51
- No pathological half-frequency specialization in embeddings (unlike p101/999's cos/sin imbalance)
- Does not perform particularly well by 25K compared to typical variants

**What's unusual**: In normal variants, attention heads diversify *during* grokking (simultaneous with test loss drop). Here, they converge first (all to freq 22), then diversify *after* grokking. The post-grokking diversification is unstable and slow. This suggests the initial convergence to a single frequency may be a local minimum that the model has to escape.

**Comparison with p101/999**: Both are anomalous, but differently:
- p101/999: Degenerate Fourier solution, open PC2/PC3 loop, cos/sin imbalance
- p107/999: Initial consensus (all heads → freq 22), followed by slow unstable diversification, no embedding pathology

Both share: poor final performance relative to typical variants, and seed 999 involvement. The seed=999 correlation across two different primes is worth watching.

### p107/seed485: Very Late Grokker

Doesn't grok until ~16.5K epochs — the latest observed grokking onset in the dataset. For comparison, most variants grok between 2-5K epochs.

**Open questions**:
- What does p107/485's attention head specialization timeline look like? Does it follow the normal pattern (diversify during grokking) but just later?
- Is p=107 inherently harder, or is this a seed interaction?
- Does the late grokking correlate with any particular trajectory geometry (open vs closed loop)?

### Emerging Pattern: seed=999 Anomalies

| Variant | Anomaly |
|---------|---------|
| p101/999 | Degenerate Fourier solution, open PC2/PC3 loop, 6:1 cos/sin imbalance |
| p107/999 | Post-grokking attention head diversification instability |
| p113/999 | (Reference "normal" variant — no anomaly) |

Two anomalies out of ~6 seed=999 variants could be coincidence or could indicate that this specific initialization creates trajectories that are more susceptible to unusual dynamics. Need more data points to distinguish.

---

*p107/999 shows a distinct failure mode from p101/999: consensus-then-diversification rather than degenerate solution. Both produce poor final performance. seed=999 correlation across primes is worth tracking.*

---

## 2026-02-14: Dimensionality Compression as Grokking Precondition

### Observation

Across all variants examined (p113/999, p113/485, p101/999, p107/999, p107/485), effective dimensionality (participation ratio) follows the same qualitative pattern:
- All weight matrices start at high participation ratio (~80-120) from random initialization
- Monotonic decline throughout training
- Sharp collapse in W_in, W_out, W_U around participation ratio ~25-30
- The sharp collapse coincides with or slightly precedes grokking onset

**The timing varies but the threshold value is consistent:**
- Fast grokkers (p113): sharp collapse at ~10-12K epochs
- Slow grokkers (p101/999): sharp collapse at ~15K epochs
- Late grokkers (p107/485): sharp collapse at ~18-20K epochs

### Hypothesis: Compression Threshold for Grokking

The model needs to compress its weight representations below a participation ratio threshold (~25-30) before the algorithmic (Fourier) solution becomes energetically favorable. Weight decay drives compression throughout training; grokking happens when compression crosses the threshold where a low-rank algorithmic solution is cheaper than a high-rank memorization solution.

**Predictions:**
- Stronger weight decay → faster compression → earlier grokking
- Variants that compress slowly (p107/485) grok late
- Variants that reach the threshold but with a distorted subspace (p101/999) grok poorly

### Spectral Gap as a Finer Metric

The participation ratio measures how many singular values contribute, but doesn't capture *how cleanly separated* the signal subspace is from the noise subspace. The spectral gap (ratio of dominant to non-dominant singular values, or the gap between top-k and remaining) may be a more mechanistically meaningful predictor.

**Connection to subnetwork competition (Merrill):** A growing spectral gap is literally the weight space separating into "subnetwork that matters" and "subnetwork that doesn't." The velocity spike during grokking could be the dominant subnetwork's singular values growing while the others shrink — visible as a spectral gap widening.

**Connection to graph Laplacian / connectedness:** The second eigenvalue of the graph Laplacian measures connectivity. Applied to weight matrices, the spectral gap measures how coherent/structured the learned representation is. Small gap = diffuse, weakly structured. Large gap = tight global structure (the algorithmic solution).

### Next Steps (post-restructuring)

The singular value spectrum is already captured per-epoch by `EffectiveDimensionalityAnalyzer`. Computing the spectral gap metric from stored singular values would be a lightweight notebook exploration — no new analyzers needed. Compare gap trajectory against participation ratio trajectory to see which better predicts grokking onset.

---

*Dimensionality compression to a consistent threshold (~PR 25-30) appears to be a precondition for grokking. Spectral gap may be a finer-grained predictor. Both connect to subnetwork competition through the lens of weight space separating into signal vs noise subspaces.*

---

## 2026-02-15: p59/485 — Second Open-Loop Anomaly

### Discovery

p59/seed485 is the second model (after p101/999) to show an open PC2/PC3 trajectory and anomalous grokking behavior. p59/seed999 looks normal — clean self-intersecting loop, grokked by epoch ~8K, robust specialization.

### Key Metrics

| Metric | p59/999 (normal) | p59/485 (anomalous) |
|--------|------------------|---------------------|
| Grokking onset (test_loss < 0.1) | Epoch 8,058 | Epoch 24,248 |
| Final test loss | 0.000000 | 0.020680 |
| Specialized neurons at 25K | ~460 (plateau) | ~180 (still climbing) |
| Frequency bands used | 3 (5, 15, 21) | 2 (5, 21 only) |
| PC2/PC3 loop | Self-intersecting | Open |

### What's Distinct About This Anomaly

**Missing frequency band.** p59/485 only developed neuron specialization for frequencies 5 and 21. Frequency 15 is completely absent — zero neurons specialized to it across the entire training run. p59/999 uses all three bands (5, 15, 21) with robust neuron counts in each.

**Nothing converged.** At epoch 25K, dominant Fourier norms are still climbing, neuron counts still rising, effective dimensionality still dropping, attention heads still specializing. The model was caught mid-transition — it barely crossed the grokking threshold 750 epochs before training ended.

**Low velocity throughout.** Component velocity drops to near-zero early and stays there. The model moved through weight space extremely slowly — it wasn't that it tried and failed, it was barely moving. The initialization put it on a gradient-poor path.

**Landscape flatness still noisy.** Oscillations persist past epoch 15K (p59/999 settles by ~8K). Baseline loss only recently approached zero.

### Comparison with Other Anomalies

| Property | p101/999 | p107/999 | p59/485 |
|----------|----------|----------|---------|
| PC2/PC3 loop | Open | Not yet characterized | Open |
| Grokking quality | Degenerate (cos/sin imbalance) | Post-grokking instability | Incomplete (missing freq band) |
| Seed | 999 | 999 | 485 |
| Root cause hypothesis | Overshooting generalizing manifold | Consensus-then-diversification | Gradient-poor initialization path |

**Notable: p59/485 is the first seed=485 anomaly**, breaking the pattern where only seed=999 produced pathological variants. This suggests the anomaly mechanism isn't specific to one seed — it's about the interaction between prime and seed producing an unlucky initialization.

### Implications for Cross-Variant Analysis

With three anomalous models now identified (p101/999, p107/999, p59/485), quantitative cross-variant comparison becomes increasingly valuable. Four metrics that would cleanly separate normal from anomalous:

1. **Loop closure** — does the PC2/PC3 path self-intersect?
2. **Grokking timing** — when does test loss cross threshold?
3. **Frequency coverage** — how many expected frequency bands developed specialized neurons?
4. **Convergence status** — have dimensionality/specialization/norms plateaued by training end?

These could form the basis of a "variant health dashboard" in the cross-variant analysis requirement.

---

*p59/485 is a second open-loop anomaly with distinct pathology (missing frequency band, gradient-poor path). First seed=485 anomaly — mechanism is prime×seed interaction, not seed-specific.*

---

## 2026-02-16: Competition Window and Basin Landing — Thrashing as Momentum

*Note: "thrashing" = neurons repeatedly switching dominant frequency across epochs.*

### The Triad

Three variants illustrate a spectrum of subnetwork competition dynamics, visible in both the PC2/PC3 trajectory loop and neuron frequency thrashing (REQ_042):

| Variant | Loop Geometry | Thrashing Pattern | Grokking | Interpretation |
|---------|--------------|-------------------|----------|----------------|
| p59/485 | Open (overshoot) | Thrashes throughout training, never fully resolves | Never groks cleanly | Too much momentum — overshoots the basin |
| p101/999 | Open (undershoot) | Stops thrashing early, commits prematurely | Slow, degenerate solution | Too little momentum — undershoots the basin |
| p109/485 | Self-intersecting (sweet spot) | Competition resolves productively | Early grokker (5.5K) | Right momentum — lands in the basin |

### The Basin Landing Hypothesis

There appears to be a basin in weight space that the model must reach for clean grokking. The dynamics look like a momentum problem:

1. The model needs **enough competition** (thrashing) to build momentum toward the generalizing basin
2. But **not so much** that it overshoots and can't settle

This frames grokking as a landing problem: the training trajectory must arrive at the generalizing basin with the right velocity — fast enough to reach it, slow enough to stay.

### Causal Direction: Open Question

Two plausible framings:

**A. Thrashing drives momentum.** Subnetwork competition generates the gradient signal that pushes the model through weight space. Early commitment (p101/999) means losing the gradient signal from competition too soon — the model stalls. Persistent thrashing (p59/485) means the gradient signal never coherently points toward the basin — the model wanders.

**B. Momentum drives thrashing.** The initialization determines the trajectory through weight space. Trajectories that pass near the basin at the right speed naturally resolve competition (neurons commit because the basin is attractive). Trajectories that miss the basin keep thrashing because there's no attractor to settle into.

These aren't mutually exclusive — thrashing and trajectory likely co-determine each other. But the causal priority matters for intervention design: if (A), you'd want to modulate competition directly; if (B), you'd want to modulate trajectory (e.g., LR scheduling).

### Supporting Evidence

- **p59/485** chooses frequencies and follows a "somewhat normal path" on some metrics, yet thrashes throughout. This suggests the thrashing isn't just about *finding* frequencies — it's about the dynamics of *settling* into them. The model found frequencies but couldn't commit. (Supports framing B: trajectory determines commitment.)
- **p101/999** stops thrashing early and underperforms. If thrashing were purely noise, early cessation should be beneficial. Instead, it correlates with a degenerate solution. (Supports framing A: competition is productive signal.)
- **p109/485** shows the sweet spot: enough competition to explore, decisive enough commitment to converge. The loop closes at the right moment.

### Potential Metrics

- **Competition window duration**: epoch range where >50% of neurons are actively switching
- **Resolution sharpness**: rate of commitment (sigmoid-like vs gradual)
- **Terminal commitment fraction**: what % of neurons are committed at training end
- **Commitment timing relative to grokking onset**: does commitment precede, coincide with, or follow the test loss drop?

These could be computed from existing neuron_dynamics cross-epoch data without new analyzers.

---

*The competition-resolution dynamics may be the mechanistic core of grokking. Too much competition (overshoot), too little (undershoot), or just right (basin landing). Causal direction between thrashing and momentum is the key open question.*

---

## 2026-02-16: p59/485 Extended Training — Settling With Damage

### Follow-Up Experiment

Extended p59/485 training from 25K to 35K epochs.

### Results

1. **The thrashing did settle.** The chaotic frequency switching resolved shortly after 25K. The model was not stuck in a fundamentally different attractor landscape — it just needed more damping time.
2. **The PCA loop tightened.** Recomputing parameter trajectory with 35K epochs reduced the original overshoot. The trajectory geometry moved toward the self-intersecting pattern seen in normal variants.
3. **Test loss remains underperformant.** Despite settling, the model did not achieve clean grokking performance.

### Interpretation: Arrival State Matters

Settling late is not the same as settling well. By the time p59/485 resolved its thrashing, it had already committed to an impoverished frequency allocation (2 bands instead of 3 — frequency 15 never developed). The model landed in *a* basin, just not the optimal one.

This suggests the **competition window is not just about duration but about timing**. The productive phase — where subnetwork competition shapes which frequency bands develop — has a finite window. What the model brings with it when it finally settles determines solution quality:

| Variant | Arrival state | Outcome |
|---------|--------------|---------|
| p109/485 | 3 frequency bands differentiating, right momentum | Clean grokking |
| p59/485 | Missing frequency band, late arrival after extended chaos | Suboptimal convergence |
| p101/999 | Premature commitment, degenerate cos/sin ratio | Degenerate solution |

### Refinement to Basin Landing Hypothesis

The basin exists for p59/485 — the model eventually found it. But the basin has structure: arriving with incomplete frequency coverage means settling into a suboptimal region of the basin. The competition window doesn't just determine *whether* you reach the basin, but *where in the basin* you land.

---

*Extended training confirmed p59/485's chaos was temporary but damaging. The competition window shapes what structure the model carries into convergence. Late settling = impoverished solution, even if the model eventually stabilizes.*

---

## 2026-02-16: Representational Geometry — Activation-Space Confirmation of Weight-Space Story

### Three-Variant Comparison (REQ_044 Visualizations)

The representational geometry analyzer (centroid PCA, distance heatmaps, time-series metrics) provides activation-space evidence that directly confirms and extends the weight-space observations from parameter trajectory analysis.

| Metric | p113/999 (control) | p101/485 (slow) | p101/999 (anomalous) |
|--------|-------------------|-----------------|---------------------|
| Circle formation epoch | ~9K (grokking turn) | ~16K+ (delayed) | ~18K (transient) |
| Var explained (top-2 PCs) | 76% → 57% (stable) | 30% → 39% (slowly improving) | 30% → 29% (lost) |
| Circle quality | Tight, evenly spaced | Wobbly, less crisp | Momentary then collapsed |
| Distance heatmap banding | Clean diagonal stripes | Faint then moderate | Brief structure, then muddy |
| Fisher discriminant (Resid Post) | ~13 (strong) | ~10 (moderate) | ~6.5 (weak) |
| Dimensionality collapse | Sharp at ~9K | Gradual at ~15K | Gradual, never fully resolves |
| Circularity time-series | Clean transition | Noisy oscillation 8-15K | Violent oscillation 8-20K, never settles |

### Key Insight: Variance Explained as Compression Proxy

The top-2 PCA variance explained tracks how well the model has compressed its class representations into a 2D Fourier subspace. This is the **activation-space mirror** of the weight-space dimensionality compression (participation ratio ~25-30 threshold) observed earlier:

- **p113/999**: 76% at epoch 9K → the model has already packed most class structure into 2 dimensions. This is a *compressed* representation.
- **p101/485**: Only 39% even at 25K → class structure is spread across more dimensions. The model is working but inefficiently.
- **p101/999**: 30% and declining → the model briefly found 2D structure but couldn't maintain it. The representations are *expanding* back into higher dimensions.

### Connection to Basin Landing Hypothesis

The momentary circle at epoch 18K in p101/999 is the geometric manifestation of the "transient approach to the generalizing manifold" from the basin landing hypothesis. The model's activation-space centroids briefly arrange into the Fourier circle — the model *found* the solution — but couldn't maintain it because the weight-space dynamics (degenerate cos/sin ratio, premature neuron commitment) don't support stable circular structure.

This provides a new way to think about grokking failure: it's not that p101/999 never discovers the algorithm, it's that the algorithm is **transiently expressed but not consolidated**. The weight-space substrate can't hold the activation-space geometry.

### Circularity Time-Series as Diagnostic

The circularity & Fourier alignment panel in the time-series view turns out to be an excellent diagnostic for grokking health:
- **Clean transition** (p113/999): single sharp rise, stable plateau → healthy grokking
- **Noisy oscillation** (p101/485): oscillations that eventually settle → delayed but viable
- **Violent persistent oscillation** (p101/999): never settles → pathological

This could be a strong candidate metric for the "variant health dashboard" concept from the cross-variant analysis discussion.

---

*Representational geometry provides activation-space confirmation of the weight-space grokking story. Variance explained tracks compression quality; circularity time-series tracks grokking health. The transient circle in p101/999 shows the model found but couldn't hold the solution.*

### p59/485 — Overshooter Lands Late But Sticks

p59/485 adds the critical fourth comparison point:

| Metric | p113/999 | p101/485 | p101/999 | p59/485 |
|--------|----------|----------|----------|---------|
| Circle at training end | Solid (57%) | Wobbly (39%) | Lost (29%) | Solid (56%) |
| Fisher discriminant | ~13 | ~10 | ~6.5 | ~20 (highest) |
| Center spread (Resid Post) | ~50, plateaued | ~40, plateaued | ~35 | ~120, still climbing |
| Centroid scale (PC axes) | ±50 | ±30 | ±20 | ±100 |
| Grokking onset | ~9K | ~15K | Never clean | ~24K |

**Key finding**: p59/485 arrives late to the generalizing basin but builds the strongest separation of any variant observed. Its Fisher discriminant (~20) exceeds even p113/999 (~13). The centroid scale is enormous (±100 vs ±50) — the model appears to be *overcompensating* for having only 2 frequency bands instead of 3 by pushing centroids further apart.

**Contrast with p101/999**: Both have open PC2/PC3 loops. Both are anomalous. But p59/485 eventually forms a solid circle (56% var explained, rising), while p101/999 loses its circle (29%, declining). The difference: p59/485 had enough momentum to stick when it arrived; p101/999 committed too early and bounced off.

**Refinement to basin landing hypothesis**: Arrival velocity matters more than arrival timing. Late arrivals can succeed if they carry enough momentum. Early arrivals fail if they lack the dynamical structure (frequency balance, neuron competition) to sustain the solution. The basin is sticky for well-prepared models and slippery for under-prepared ones.

---

*p59/485 proves the basin is reachable even via the overshoot path. The overshooter compensates with massive scale. The undershooter (p101/999) fails not because it can't find the basin, but because it can't stay.*

---

## 2026-02-23: Phase Scatter Voids — Dual View of Centroid Geometry

### Observation

The phase alignment scatter (ψ_m vs. φ_m at dominant frequency) shows characteristic void regions — not noise, not missing data, but structured gaps. This pattern immediately recalled the centroid PCA Lissajous structure.

### What Creates the Voids

Both φ and ψ are constrained to [-π, π] by arctan2, but the alignment relationship is ψ = 2φ. This requires ψ ∈ [-2π, 2π] — double the range available. The wrapping at ±π splits what would be a single diagonal into **three bands**:

1. **Central band**: φ ∈ [-π/2, π/2], ψ = 2φ ∈ [-π, π] — lies on the dashed ideal line
2. **Lower-right band**: φ ∈ (π/2, π], theoretical ψ = 2φ wraps to 2φ - 2π ∈ (-π, 0]
3. **Upper-left band**: φ ∈ (-π, -π/2], theoretical ψ = 2φ wraps to 2φ + 2π ∈ (0, π]

The voids are the gaps at φ ≈ ±π/2 — the wrapping discontinuities. They're structural proof the doubled-phase relationship is real.

### Connection to Centroid PCA

Both representations are projections of the same torus topology:

- **Centroid PCA**: plots output class centroids on a circle/Lissajous parameterized by e^{ikx} and e^{2ikx}. The self-intersections and figure-8 shapes come from the double-angle relationship cos(kx)cos(ky) → cos(k(x+y)).
- **Phase scatter**: plots per-neuron weight phases (φ_m, ψ_m) subject to ψ ≈ 2φ. The three-band void structure comes from the same doubling, wrapped back into [-π, π].

Both are signatures of the same circuit: the model computes cos(kx)·cos(ky) - sin(kx)·sin(ky) = cos(k(x+y)), and the double-angle relationship is the mathematical spine of that computation. Centroid PCA sees it in output-space organization; phase scatter sees it in weight-space organization.

### Diagnostic Potential

For pathological variants (p101/999, p59/485): a model with degenerate or incomplete frequency coverage should show a distorted version of this three-band structure — bands that are faint, smeared, or misaligned. The clarity of the three-band void pattern may be a visual analog to circularity in the centroid time-series: sharp bands = well-formed Fourier circuit; blurred/missing bands = degraded or incomplete alignment.

Worth comparing the phase scatter for p101/999 (degenerate cos/sin imbalance, transient circle that collapses) against p113/999 (clean). If the three-band structure is absent or smeared for p101/999, the weight-space and activation-space views are telling the same story from complementary angles.

---

*The phase scatter voids are not artifacts — they're the weight-space signature of the same doubled-phase topology that creates Lissajous figures in centroid PCA. Both views are projections of the model's ψ = 2φ constraint onto accessible phase ranges.*

---

## 2026-02-23: Initialization Implicated in p=101/999 — Lottery Ticket Non-Null Result

### Key Finding

Null lottery ticket result (neither phase margin nor magnitude ratio at init predicts winning frequency) holds for p=113/999 (healthy variant). p=101/999 shows structure in the phase margin vs magnitude scatter — initialization IS predictive for the anomalous variant.

This is diagnostic. In healthy variants, the competition plays out through gradient flow — init doesn't matter. In p=101/999, amplitude locks early (β IPR rises to ~0.9 ahead of phase alignment), and wherever the neuron started is closer to where it ends up. The competition isn't resolved by training dynamics; it's resolved by the initialization.

### p=101/485 as Control Case

p=101/485 shows canonical grokking from a healthy loss curve. Same prime, different seed. This rules out p=101 being intrinsically hard — seed=999 is a bad initialization draw for this prime. The lottery ticket non-null result is not a property of p=101; it's specific to p=101/999's initialization.

**The mechanism**: p=101/999's amplitude competition commits prematurely before phase gradient flow can establish ψ≈2φ (dissociation: β IPR ≈0.9, α IPR ≈0.4). Because amplitude competition resolved early based on initial magnitudes rather than sustained competition, initialization is more deterministic in this variant than in healthy ones.

### p=109/485 — Fast Grokker as the Ideal Case

p=109/485 locked in fast — appears to have landed on a converged representation quickly but still went through a refinement phase. This is the opposite of p=101/999's pathology: amplitude competition resolved productively AND the model continued refining phase alignment post-commitment. Fast grokking here is evidence of arriving at the right place with the right momentum, not just arriving fast.

**Open question**: Does p=109/485's phase scatter show the clean three-band structure earlier than slower grokkers? If so, it would suggest the fast landing came with intact phase alignment from the start — the initialization happened to be well-positioned for both amplitude and phase competition.

---

*p=101/999's non-null lottery ticket result is diagnostic of broken competition dynamics. p=101/485 control confirms the pathology is init-dependent, not prime-intrinsic.*

---

## 2026-02-23: Neuron Fourier Views — Dashboard Comparison Gap

### The Friction

Comparative analysis across variants (p=101/999 vs p=101/485, healthy vs anomalous) is high friction in notebooks. Each comparison requires: re-loading data, rerunning cells, holding context between runs. Quick comparison of IPR trajectories, phase scatter, or margin vs switches across variants is the core scientific workflow but currently requires significant cognitive overhead.

The neuron Fourier views (IPR, phase alignment, phase scatter) are registered in the View Catalog (REQ_047) and artifacts are on disk for all 12 variants. The missing piece is a dashboard surface that exposes them for interactive multi-variant comparison.

### What's Needed

Not a generic multi-panel comparison tool (too ambitious). Specifically:
- IPR trajectory (input + output) side-by-side for two variants on the same axes
- Phase scatter at a selected epoch — healthy reference vs. target variant
- (Stretch) Conditioned alignment progress as a single overlaid time-series

Dash's dynamic loading capabilities (explored in sandbox UX work) are well-suited to on-demand, ad-hoc multi-graph composition. This is the next natural frontend requirement after the View Catalog architecture stabilizes.

**Status**: Candidate requirement. Blocked by: none. Priority: next after current branch.

---

*Notebook is the right excavation tool; the dashboard is the right comparison tool. The neuron Fourier views need dashboard exposure to support the cross-variant comparative workflow.*

---

## 2026-03-02: Grokking Window Definition — The Core Methodological Gap

### Why This Matters

Multiple claims in the findings depend on "before grokking onset" or "during the grokking window." Without a rigorous, cross-variant-stable definition of onset, these claims are unanchored. The definition of onset also determines whether Claim 8 (geometry precedes grokking) is testable at all — "how long before" is meaningless if onset is fluid.

### The Two-Problem Split

**Performance-based definition (literature standard):** Test loss drop onset. The inflection point (maximum d²acc/dt²) is the most well-defined candidate — doesn't require knowing the final plateau value, is mathematically deterministic given sufficient smoothing. But this measures the *consequence* of grokking, not the cause.

**Mechanistic definition (appropriate for scaffolding hypothesis):** For claims about what precedes grokking, a representation-level marker is the right reference point. Circularity & Fourier Alignment exceeding a threshold (current candidate: >0.40) is promising because: (a) it's already computed, (b) it measures a property that should precede performance, (c) its timing relative to the test loss drop is itself evidence for the scaffolding hypothesis.

### DMD Residual as Third Candidate

The DMD pipeline (REQ_050/051) will produce a per-epoch residual norm measuring where the linear approximation of centroid dynamics breaks down. This is a dynamics-based onset marker — expect low residual during memorization, a spike during grokking transition, and settling post-grokking. If this aligns with both the performance marker and the Fourier Alignment marker, it strengthens all three simultaneously.

### Validation Approach

Pick two or three candidate definitions, compute them for all variants, check whether the relative ordering (fast vs. slow grokkers, successful vs. failed) is stable across definitions. If ordering is preserved, the choice is less critical than it appears. If not, the instability is informative about what's actually varying.

### Downstream Connections

- **Claim 8 resolution**: The Fourier Alignment marker, if it consistently precedes the performance marker, resolves the OR (synchronization vs. alignment) and anchors the timing claim.
- **Cross-variant table in findings.md**: Grokking window column currently based on visual inspection — a stable metric definition turns this into a computable, reproducible table.
- **PC3 Circularity/Fourier revisit**: Current circularity metric uses only top-2 PCA components (built before PC3 significance was recognized). Once global PCA exists (REQ_050), revisiting this calculation to include PC3 is a small but potentially important correction. A 3-component circularity metric may be more predictive than the current 2-component version.

---

*Grokking window definition is the methodological linchpin. DMD residual, Fourier Alignment threshold, and test loss inflection point are the three candidates. Cross-variant ordering stability is the validation test. PC3 circularity revisit is a natural follow-on once global PCA is in place.*

---

## 2026-03-02: Converging Platform Needs — View Naming, Cross-Variant Analysis, Export

### View Naming Guidelines

The View Catalog currently has no naming convention. As views multiply, names should signal two things at a glance: (a) what the view offers and (b) the temporal scope (per-epoch snapshot vs. cross-epoch trajectory). The current asymmetry is already visible: `centroid_pca` is a per-epoch snapshot in activation space, while `parameter_trajectory` is a cross-epoch trajectory in weight space — but nothing in the names signals this.

Candidate convention: `{subject}_{method}` for per-epoch, `{subject}_{method}_timeseries` or `{subject}_{method}_trajectory` for cross-epoch. Gaps and asymmetries become readable from the name list alone.

This also aids AI-assisted navigation — a well-named catalog is faster to reason about from context.

### Cross-Variant Analysis (Elevated Priority)

The 2026-02-23 entry identifies this as a candidate requirement. It is now more urgent: the grokking window definition work produces metrics that only become meaningful in comparison across variants. A single-variant view of the DMD residual is a diagnostic; a cross-variant overlay is evidence.

The variant health dashboard concept (from 2026-02-15: loop closure, grokking timing, frequency coverage, convergence status) is the right first surface. The neuron Fourier comparison view (IPR trajectories, phase scatter) is the second.

### Export Pipeline for Collaborative Analysis

`BoundView.export()` now supports canonical path derivation from variant + epoch + view + kwargs. This enables: (a) programmatic export sets for sharing with collaborators or Claude, (b) consistent naming for notebook → PDF workflows, (c) eliminating manual naming errors. The `results/exports/` directory is the accumulation point.

---

*View naming, cross-variant analysis, and export pipeline are converging toward the same need: reproducible, communicable findings rather than ephemeral notebook sessions.*

---

## 2026-03-02: DMD Residual — Grokking Dynamics Across Variants

### Validated on 113/999 and 101/999 (log scale)

The DMD residual norm (REQ_051), viewed on log scale, is a viable per-site grokking onset signal with distinct signatures across variant archetypes.

**Site hierarchy is consistent**: Post-Embed < Attn Out < MLP Out < Resid Post throughout training. Residuals scale with depth and computation — the embedding layer is nearly always the most linearly predictable, and the residual stream the least. This is a cross-variant invariant.

### 113/999 — Broad Plateau During Grokking Window

Resid Post shows a broad elevated plateau from ~5k–12k epochs — diffuse rather than a sharp spike. After the plateau resolves, residuals drop slightly (~12–13k) corresponding to post-grokking stabilization. The geometry story: the model found its circle early and maintained it, then reorganized during grokking. The DMD sees this as a sustained departure from linearity during the reorganization phase, not a sharp onset event.

### 101/999 — Sharp Spike at ~13k; Site Dissociation

The spike at ~13k–14k is a full order of magnitude above the pre-spike baseline for Resid Post and MLP Out. This is the most prominent feature in the 35K training history. Timing: right at the labeled grokking onset.

**New observation on log scale**: Post-Embed has a dramatic *minimum* at exactly the spike moment — dropping to near 0.05 while MLP Out and Resid Post are maximal. The embedding layer briefly becomes ultra-predictable precisely when downstream computation is most nonlinear. This dissociation between sites was invisible on linear scale.

**Hypothesis (to verify with 109/485)**: The 101/999 spike is more dramatic than 113/999's because the model *did not have an organized circle entering the grokking window* — the Circularity/Fourier Alignment graph shows 101/999 finds a circle only transiently and weakly, right before what the user calls the "second search phase." Without pre-organized geometry, the grokking attempt is a sudden departure from established dynamics rather than a smooth reorganization.

### Post-spike structure for 101/999

After the 13k spike, residuals remain elevated and structured (second bump ~17–25k) before declining. This second bump aligns with the parameter velocity spike at ~24k that prompted extended training. The model didn't recover — it lost a lower-frequency MLP band instead. The DMD residual captures this second dynamics excursion as its own event.

### Connection to Basin Landing Hypothesis

The Post-Embed minimum at the spike moment for 101/999 may be the embedding layer briefly settling into a regular configuration — a momentary alignment — just before the downstream computation attempts (and fails to sustain) the grokking transition. This is the weight-space analog of the transient circle seen in the representational geometry: the model *approaches* the basin briefly (Post-Embed settles, geometry almost organizes) but the substrate (degenerate cos/sin ratio, premature neuron commitment) can't hold it.

### 109/485 — Scaffolding Hypothesis in Direct Form

**Residual:** No sharp spike at grokking onset (~5k). The broad elevated region from 1k–5k simply resolves around grokking — the model had its geometry ready, so the test-loss transition was not a sudden departure from established dynamics. This directly confirms the hypothesis that spike magnitude inversely correlates with pre-grokking geometric organization.

**Reconstruction:** At epoch 3100 (labeled pre-grokking), the ring is already fully and cleanly formed. Grokking occurs at ~5k. The geometry preceded the test-loss drop by at least 1500–1900 epochs. This is the scaffolding hypothesis in its most direct form: representational structure was in place well before generalization performance improved.

**Global PCA:** 72.3% variance in PC1+PC2 (vs 41.3% for 113/999). PC3 is only 9.8% vs 18% for 113/999. The fastest clean grokker has the most compressed representation — class structure concentrated in two dimensions.

**Post-Embed minimum at 1200 (train loss):** Fires at memorization completion, not at grokking. This separates the Post-Embed minimum from the 101/999 case, where it fired at the grokking *attempt*. The phenomenon appears to be a "stable regime achieved" marker: embeddings briefly align whenever the model enters a stable attractor, whether that's memorization or grokking. In 101/999, the failed grokking attempt produces a transient stability that the embedding layer registers, even though the downstream substrate can't hold it.

### 59/485 — Oscillatory Waves; Post-Embed Fires at Second Search Phase

**Residual:** Broad oscillatory waves throughout training — structured chaos, not random noise. A trough is visible at ~13-15k (first attractor approach) before the dynamics resume their oscillatory character. No clean spike; instead the entire training history is elevated and wave-like. Confirms the prediction from the 101/999 vs 109/485 comparison: variants without pre-organized geometry show more dramatic DMD departure events, and the most chaotic geometry produces the most chaotic residual profile.

**Post-Embed minimum at ~24-25k:** Fires at the parameter velocity spike (the "second search phase"). User confirmed: the model begins locking in at exactly this moment. This is consistent with the "stable regime" interpretation: the embedding layer registers whenever the model enters a stable attractor. For 59/485, the first search phase (~13-15k) was a transient, and the real commitment starts at 24-25k.

**Global PCA:** Asymmetric ring at enormous scale (±100), heart-shaped 3D structure rather than torus. PC1 35.3% / PC2 26.2% / PC3 18.5%. The 3D structure shows diagonal scatter in PC2 vs PC3 rather than organized Lissajous — consistent with a two-band-only solution, different from clean grokkers' clean toroidal structure.

**Reconstruction:** DMD dashed lines predict an oscillatory rise-peak-fall trajectory (what most of training looks like), while actual solid lines show sustained growth into the final ring. DMD correctly characterized the oscillatory regime but the final phase transition to ring structure was a dynamics event outside the linear fit. This is expected: approximate DMD captures the dominant long-term linear structure, and the dominant structure for 59/485 is the oscillatory search phase.

**Global PCA compression:** 61.5% in PC1+PC2 — less compressed than 109/485 (72.3%) but more than 101/999 (~27%). The overshooter achieves partial organization; the anomalous variant achieves almost none.

### Cross-Variant Summary

| Variant | Pre-grokking geometry | Residual signature | Post-Embed event |
|---------|----------------------|-------------------|-----------------|
| 109/485 (fast) | Ring fully formed at 3100, grokking at 5k | Broad plateau resolves at grokking | Minimum at train loss (~1200) |
| 113/999 (canonical) | Ring building during grokking | Broad diffuse plateau 5k–12k | Not yet characterized |
| 59/485 (overshooter) | Oscillatory search; ring during second phase | Broad oscillatory waves throughout; trough ~13-15k | Minimum at second search phase (~24-25k) |
| 101/999 (anomalous) | Transient ring at ~18k, collapses | Sharp spike at grokking attempt (~13k) | Minimum at grokking attempt spike |

The pattern: better pre-organized geometry → smoother residual transition → Post-Embed alignment tied to an earlier stable event. 59/485 sits between 113/999 and 101/999 in organization quality: it eventually locks in, but through a chaotic oscillatory search rather than a clean reorganization.

---

*Hypothesis confirmed: spike magnitude inversely correlates with pre-grokking geometric organization. The ring preceding grokking onset in 109/485 is direct scaffolding evidence. Post-Embed minimum is a "stable regime" marker, not a grokking-specific signal — fires at memorization (109/485), grokking transition (113/999 TBD), grokking attempt (101/999), and second search phase commitment (59/485).*
