# Thoughts - Platform ideas

Unstructured parking lot for ideas for evolving the platform. See Claude.md for context.
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
### Dashboard UI Redesign — Observed Friction Points

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

### Cross-Variant Comparison (Biggest Gap)

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

## 2026-03-11: Intervention Detail Pane in Variant Selector

When an intervention variant is selected in the dashboard, a detail pane should surface the full intervention config alongside the standard variant info (params, state, etc.). This lets users verify they're looking at the right hash without having to open config.json.

The data is already there — the `intervention` block is stored in `config.json` on every intervention variant directory. The requirement is purely display: detect the `intervention` key in the variant's stored config and render it as a readable summary (type, gain dict, window, ramp).

Useful especially when multiple intervention variants exist for the same base variant and the short hash alone is not self-documenting.

---

## 2026-03-11: Unembedding Fourier Alignment Panel

The `parameter_snapshot` analyzer already captures W_U. Adding a Fourier alignment view for W_U alongside the existing MLP and embedding panels would let us see whether the unembedding is pulling toward specific frequencies — potentially explaining why certain frequencies resist dampening (the gradient from W_U may be sustaining them upstream). Came up during intervention analysis: Freq 10 showed strong resistance to 0.3x dampening, possibly gradient-supported from W_U.

---
