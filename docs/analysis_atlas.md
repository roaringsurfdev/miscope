# Analysis Atlas

**Status:** Living document
**Last updated:** 2026-05-07
**Audience:** Researchers using miscope, contributors planning new analyzers, future collaborators making consolidation and scope decisions.

---

## Purpose

The Analysis Atlas is a map of miscope's analytical surface — the analyzers that exist today, the analyzers that should exist, and the conceptual structure that organizes them. It serves two functions simultaneously:

- **An inventory** of current capability, suitable as the first thing an external researcher reads after `pip install miscope`.
- **A roadmap** for planned analyzers, with status fields that double as scoping inputs for downstream requirements.

---

## How to read this document

Every analyzer entry carries a **status** and a **bucket**.

**Status** describes the entry's lifecycle position:

- `existing` — implemented and registered today.
- `existing-rename` — implemented today; planned rename and refactor onto REQ_109 primitives.
- `planned-consolidation` — multiple existing analyzers fold into a single new analyzer; the new shape is the design target.
- `planned-new` — no current implementation; capability is missing.
- `retire` — currently exists but slated for removal.

**Bucket** describes the validation strategy and which downstream REQ picks up the work:

- `refactor` — same conceptual measurement, cleaner implementation built on REQ_109 primitives. Parity validation against the old analyzer is meaningful. Picked up by REQ_111.
- `reorganization` — new conceptual shape. Inputs, outputs, or scope change. Validation against primitives + research-grade reference (reproducing known findings on canon variants), not parity. Picked up by a new scoped REQ.
- `new` — net-new capability. No old analyzer exists. Validated against primitive correctness and reproduction of reference computations.
- `retain` — no change planned.

---

## Framing: three dimensions

The analyzer set organizes naturally along three orthogonal dimensions. Every analyzer lives primarily in one of them.

### Universal Core

Task-agnostic instruments that should run on any small classification network by default. The first-pass treatment of any new model.

For small models, exhaustiveness is a feature, not waste. Snapshotting every weight matrix and every activation site is cheap relative to the analytical leverage it provides. Stinginess is a constraint that pays at scale; at workbench scale it just hides structure.

### Family Basis Projections

Analyses that require a family-supplied basis or interpretive lens — Fourier decomposition for modular arithmetic, helical bases for counting tasks, etc. The mathematical primitives are universal; the **basis** is family-supplied. This honors the architectural rule from [PROJECT.md](../PROJECT.md): families are context providers, views and analyzers are universal instruments.

These are necessarily secondary to first-level analysis — they require knowledge or interpretation of the algorithm the model has learned, which the universal core surfaces first.

### Dynamical Proxies

Observations about how the system *moves* through training, not just what it looks like at a given checkpoint. Four sub-categories:

- **Trajectory geometry** — properties of the path θ(t) through parameter or representation space.
- **Landscape geometry** — properties of the loss surface at a given θ.
- **Operator dynamics** — modal decompositions and coupling between sites.
- **Phase-space fits** — characterization tools that fit hypothesized dynamical-system structures (Lissajous oscillation, saddle-center-center linearization, sigmoidal transit) to observed trajectories.

Together these form a layered dynamical picture: how the system moves, what landscape it moves through, how its components couple, and what phase-space structure the trajectory appears to imply.

---

## Universal Core

### `parameter_snapshot`
**Status:** existing | **Bucket:** retain

Per-epoch capture of all 9 weight matrices: `W_E, W_pos, W_Q, W_K, W_V, W_O, W_in, W_out, W_U`. Foundational input for every weight-space derivation.

### `weight_spectra`
**Status:** existing-rename (currently `effective_dimensionality`) | **Bucket:** refactor

Per-epoch singular values of all weight matrices; participation ratios as summary. Rename clarifies scope (computes spectra; participation ratio is one of several derivable metrics). REQ_111 covers the rename + REQ_109 primitive integration.

### `representation_geometry`
**Status:** existing-rename (currently `repr_geometry`) | **Bucket:** refactor

Per-epoch class manifold geometry at 4 sites: centroids, radii, dimensionality, mean radius, center spread, SNR, Fisher discriminants, PCA variance per PC. The `fourier_alignment` and `circularity` fields are conceptually Family Basis Projections; surface them through that column even though they currently live in this analyzer's output dictionary.

The class-manifold framing generalizes to any classifier (residues = classes for modadd; could be any output label). Generalization to non-classifier tasks (regression, generative) is an open question; see *Open questions* below.

### `parameter_trajectory`
**Status:** existing-rename (currently `parameter_trajectory_pca`) | **Bucket:** refactor

Cross-epoch PCA on weight trajectories with first-order velocity. Rename drops the implementation detail (PCA) from the name. Acceleration / curvature / torsion live in Dynamical Proxies → Trajectory geometry, fed by this analyzer's outputs.

### `representation_trajectory`
**Status:** planned-consolidation (absorbs `global_centroid_pca` and the trajectory portion of `centroid_dmd`) | **Bucket:** reorganization

Per-class centroid trajectories in a single cross-epoch PCA basis, plus standard trajectory metrics. The DMD modal analysis splits out into Dynamical Proxies → Operator dynamics; see also REQ_073.

### `activation_snapshot`
**Status:** planned-consolidation (absorbs `neuron_activations` and `attention_patterns`) | **Bucket:** reorganization

Generalizes raw activation capture into a single snapshot analyzer parameterized by site. Cheap on small models; foundation for downstream activation-space analysis.

### `neuron_grouping` *(home: REQ_118, shipped 2026-05-08)*
**Status:** existing | **Bucket:** retain

Data-driven clustering of neurons by learned behavior. Input: per-neuron activation profiles or weight signatures. Output: group assignments + group centroids + dispersion metrics.

**Implementation (REQ_118):** per-epoch SecondaryAnalyzer that consumes `parameter_snapshot` artifacts and produces per-epoch `GroupAssignment` + `GroupSummary` artifacts. Universal kmeans path runs by default on per-neuron weight signatures (`W_in[:, n]` concatenated with `W_out[n, :]`). The modadd family registers an override that performs argmax-by-basis on the Fourier-projected composed input weight (`W_E[:p] @ W_in`) with a calibrated variance-fraction threshold — recovers the documented canon frequencies cleanly.

Family override mechanism is the architectural pattern for any future task-specific grouping: families are context providers (per `PROJECT.md`); the universal analyzer dispatches on whether the family supplied an override, but does not encode any task-specific logic itself.

Cross-epoch consumers (e.g., `parameter_dmd`) pick a single epoch's grouping via a `reference_epoch` configuration on their side; the grouping analyzer itself does not make that choice. This keeps the per-epoch artifact contract simple and lets consumers study transient phenomena (e.g., a frequency that's been abandoned by the last epoch) by pinning to a snapshot where the grouping captures it.

### `group_geometry`
**Status:** planned-consolidation (absorbs `neuron_group_pca`, `freq_group_weight_geometry`, `intragroup_manifold`) | **Bucket:** reorganization

Operates on whatever grouping `neuron_grouping` produced. Outputs: per-group centroid, radius, dimensionality, separation metrics (Fisher mean/min, SNR, circularity), within-group manifold curvature (R²_quadratic). Replaces the current Fourier-locked group geometry with a basis-independent shape.

### `neuron_dynamics`
**Status:** existing | **Bucket:** retain (target of generalization once `neuron_grouping` lands)

Per-neuron frequency dynamics: switch counts, commitment epochs. Currently Fourier-specific. After `neuron_grouping` lands, this conceptually becomes `group_membership_dynamics` (tracking how neurons move between groups over training, regardless of basis). Defer the rename until the grouping primitive is in place.

### `input_trace`
**Status:** existing | **Bucket:** retain

Per-epoch predictions on every input pair, with split labels and confidence. Generic across classifiers.

### `input_trace_graduation`
**Status:** existing | **Bucket:** retain

Cross-epoch derived: epoch at which each input first becomes correct (with stability window). Generic dynamical observation.

### `landscape_flatness`
**Status:** existing | **Bucket:** retain

Random-perturbation flatness proxy. Coarse but cheap. Lives in Universal Core because it runs on any model with a loss function. Complemented (not replaced) by Hessian top-k in Dynamical Proxies → Landscape geometry; flatness remains valuable as a fast first-pass signal.

---

## Family Basis Projections

Stubs only. Full schemas defer to family-led work — they require the family's context and learned-algorithm interpretation.

### `weight_basis_projection`
**Status:** planned-consolidation (absorbs `dominant_frequencies`, `attention_fourier`, `neuron_fourier`, projection step of `fourier_nucleation`) | **Bucket:** reorganization

Projects weight matrices onto a family-supplied basis, parameterized by site. Modadd family supplies a Fourier basis; other families supply different bases or none.

### `activation_basis_projection`
**Status:** planned-consolidation (absorbs `attention_freq`, `neuron_freq_clusters`) | **Bucket:** reorganization

Same shape, activation-side. Family-supplied basis, parameterized by site.

### `fourier_frequency_quality`
**Status:** existing | **Bucket:** retain (becomes derived view on `weight_basis_projection`)

Modadd-family-only. R² of ideal mod-p tensor projected onto dominant frequency subspace.

### `fourier_nucleation`
**Status:** existing | **Bucket:** retain

Modadd-family-only. Iterative Fourier projection of neuron response profiles; latent frequency bias at init. The iterative refinement is the value; conceptually distinct from the projection step now absorbed into `weight_basis_projection`.

### `transient_frequency`
**Status:** existing | **Bucket:** retain (open: see *Open questions*)

Detects frequencies that appear in neuron groupings transiently. Currently depends on Fourier-grouped neurons. Whether this generalizes to "transient groups" (basis-independent) or stays Fourier-specific is open until `neuron_grouping` lands.

### `coarseness`
**Status:** retire (subsumed by `activation_basis_projection`) | **Bucket:** retire

Blob-vs-plaid neuron classification via low-frequency energy ratio. Verify that `activation_basis_projection` outputs preserve the signal before retiring.

---

## Dynamical Proxies

### Trajectory geometry — path properties

#### `trajectory_metrics`
**Status:** planned-new | **Bucket:** new

Second-order trajectory metrics on `parameter_trajectory` outputs: acceleration, curvature, torsion. First-order velocity is already in `parameter_trajectory`. Detects sharp reorganizations (turning points), grokking-onset signatures, momentum effects. Cheap to compute on top of existing PCA projections.

### Landscape geometry — point properties

#### `landscape_flatness`
*(cross-referenced from Universal Core)* — coarse first-pass via random perturbation. Already exists.

#### `hessian_topk`
**Status:** planned-new | **Bucket:** new

Top-k Hessian eigenvalues + eigenvectors via Lanczos iteration with Hessian-vector products (Hvp). Outputs:

- **Saddle signature** — count of negative eigenvalues at the current θ. This is the mathematically correct definition of "saddle" and reframes the saddle work currently scattered across `Roadmap_Analysis_rough.md` (*Formalizing the Saddle Shape in Parameter Trajectory PCA*, *Formalizing the Saddle Shape in Neuron Frequency Group PCA*) as a quantitative analyzer rather than a visual-inspection task.
- **Sharpness** — λ_max as a sharpness/flatness metric, complementing `landscape_flatness`.
- **Top-k eigenvectors** — directions of maximum landscape curvature, available for visualization in PCA bases.

Distinct from trajectory curvature: trajectory curvature describes the *path*; Hessian eigenvalues describe the *landscape* at the path's current point. Both belong; they answer different questions.

### Operator dynamics — modal decompositions and coupling

#### `activation_dmd` *(home: REQ_117 phase 1, shipped 2026-05-08)*
**Status:** existing | **Bucket:** retain

Per-site windowed DMD on per-class centroid trajectories in global PCA space. Pipeline: sliding-window DMD across the centroid trajectory at each of four sites (`resid_pre`, `attn_out`, `mlp_out`, `resid_post`) → eigenvalue tracking via greedy nearest-neighbor matching across windows → peak-based regime detection (`scipy.signal.find_peaks` with median+3·MAD height + 1·MAD prominence + edge-padding) → per-regime DMD as a recursive second pass.

**Per-variant signatures observed at zoom** (see [variant_atlas.md](variant_atlas.md) for full per-variant detail): canon `mlp_out` shows a pendant of paired complex conjugates near (1, 0); p109 shows a discrete fan + paired pairs; p101 shows teardrops + isolated complex pairs at `resid_post`; p59 shows sparse near-real eigenvalues. Each variant has its own visual identity, distinguishable at zoom but invisible at the textbook ±1.5 view of the complex plane. Dashboard renders at `/activation-dmd` with auto-zoom by default.

Supersedes REQ_073's parameter-track-only proposal (now part of `parameter_dmd`); absorbs the Research Claude drafts (REQ_001 / REQ_002 / REQ_003) that originally proposed the windowed treatment.

#### `parameter_dmd` *(home: REQ_117 phase 2, shipped 2026-05-09)*
**Status:** existing | **Bucket:** retain

Per-(group, matrix) windowed + per-regime DMD on weight matrices. For each `(group_id, matrix in {W_in, W_out})` where the grouping comes from `neuron_grouping` at a configurable `reference_epoch`: build per-epoch state trajectory by flattening the group's slice of the matrix, fit a global PCA across the trajectory, project to top components covering ≥95% variance (capped at 50), then run the same four-stage DMD pipeline as `activation_dmd`.

W_in and W_out are treated as separate matrices (not concatenated) because they reorganize on independent timelines — empirically validated on `p109/s485/ds598` group 3 (k=4) where W_out drops to near-zero around epoch 2.5k while W_in is still elevated. Concatenating would have averaged this signal away.

`reference_epoch` defaults to the last available checkpoint; configurable via the `parameter_dmd_reference_epoch` context key (use `pipeline.run(extra_context={"parameter_dmd_reference_epoch": ...})`). Pinning to an earlier epoch lets downstream analysis study weight dynamics relative to a grouping that captures transient phenomena — e.g., on `p101/s999/ds598` with `reference_epoch=20000`, freq 5 (group 4, 39 neurons) appears as a populated group and its ~25k abandonment shows up as a residual bump in both W_in and W_out.

Dashboard renders at `/parameter-dmd` with dynamic group dropdown (populated from the loaded variant's `populated_groups`), matrix dropdown (W_in / W_out), and a reference_epoch indicator surfacing which `neuron_grouping` snapshot was used.

#### Within-DMD multi-lens convergence (2026-05-09)

`activation_dmd` and `parameter_dmd` operate on fundamentally different mathematical objects (activation centroids vs. weight slices) but recover the same regime structure on the same training events. Per-variant overlays (`mlp_out` from `activation_dmd` overlaid against per-group W_in/W_out from `parameter_dmd`, max-normalized) confirm tight timing alignment on major events across all four reference variants. Magnitude differences between the two views encode event scope: localized parameter events (e.g., the freq 5 abandonment on `p101/s999/ds598`) show more relative magnitude on the parameter side because they aren't diluted by averaging over uninvolved neurons; broad geometric events (e.g., grokking-window reorganization) show comparable magnitudes across both lenses. mlp_out is consistently smoother than W_in/W_out — part of a smoothness gradient where activation residuals smooth what parameter residuals sharpen.

This convergence is structural evidence for DMD as a regime detector: two distinct operations recover the same boundaries. It also makes parameter DMD genuinely additive rather than redundant — the cases where the two views *disagree* (e.g., end-of-training mlp_out activity on p109 without a parameter-side counterpart, candidate neural-collapse signature) are exactly where the diagnostic carries new information.

#### `gradient_site`
**Status:** existing-rename + integrate | **Bucket:** refactor

Per-site per-frequency gradient energy across training, with cross-site similarity. Currently lives in its own analyzer with direct checkpoint loading (bypasses the artifact pipeline). Refactor: integrate into the artifact pipeline; generalize per-frequency to per-basis (Family Basis Projections column).

#### `cross_site_coupling`
**Status:** planned-new | **Bucket:** new

Phase-lock and synchrony metrics between sites (embedding ↔ attention ↔ MLP). Builds on `gradient_site`'s cross-site similarity hints. Targets the saddle-mediated transport and intragroup-manifold timing observations in current research notes. Possibly overlaps with REQ_055 (attention head phase analysis); to coordinate scope when implementing.

### Phase-space fits — fitting dynamical-system structure to observed trajectories

Distinct from the other sub-categories: these analyzers do not measure intrinsic properties of the trajectory, the loss surface, or modal structure. Instead, they *assume* a phase-space model (Lissajous oscillation, saddle-center-center linearization, sigmoidal transit between basins) and fit its parameters to observed trajectory data. The fits are characterization tools — when one holds, the local dynamics has the assumed eigenvalue signature; when it fails, the assumption was wrong. This sub-category bridges internal representation and underlying dynamics while staying within generic nonlinear dynamics.

This sub-category is expected to grow as classical dynamical-systems tooling matures in the project. Initial entries are the three below.

#### `lissajous_fit`
**Status:** planned-new (specified in REQ_111 as a research-active addition) | **Bucket:** new

Per-epoch `characterize_lissajous` fit on centroid PCA at any registered site. Tracks `LissajousParameters` over training: ratio of the two oscillation frequencies, phase relationship, and quality-of-fit residual. The planar signature of two coupled slow oscillations.

#### `saddle_center_center_fit`
**Status:** planned-new | **Bucket:** new

Fits a saddle-center-center linearization on `representation_trajectory` (Class Centroid PCA). Outputs eigenvalue signature (one real pair plus two imaginary pairs in the 3D case), saddle direction, two oscillation planes with their frequencies, and quality-of-fit residual. Tracked across epochs.

Sibling of `lissajous_fit`: Lissajous is the planar signature observed when two slow oscillation directions are active; saddle-center-center adds the unstable direction and identifies all three structures jointly.

Reference variants: p109/s485/ds598 (expected to fit cleanly given observed ring geometry), p59/s485/ds598 (expected not to). Discrepancy between expected and observed is itself informative — ruling-in and ruling-out are both valid outcomes.

Equilibrium identification is a prerequisite. Initial implementation accepts caller-supplied equilibrium location; a fixed-point-detection primitive may emerge later.

#### `saddle_transport_sigmoidality`
**Status:** planned-new (specified in REQ_111 as a research-active addition) | **Bucket:** new

Per-segment `characterize_sigmoidality` fit characterizing transit between basins or across saddle regions. Segment boundaries from caller-configured input (manual, or from a future segment-discovery primitive). Reference parity: framework-notebook reported numbers on canon variants.

---

## Consolidation map

How current analyzers map to Atlas entries:

| Existing analyzer | Target | Bucket |
|---|---|---|
| `parameter_snapshot` | Universal Core / `parameter_snapshot` | retain |
| `effective_dimensionality` | Universal Core / `weight_spectra` | refactor |
| `repr_geometry` | Universal Core / `representation_geometry` | refactor |
| `parameter_trajectory_pca` | Universal Core / `parameter_trajectory` | refactor |
| `global_centroid_pca` | Universal Core / `representation_trajectory` | reorganization |
| `centroid_dmd` *(trajectory portion)* | Universal Core / `representation_trajectory` | reorganization |
| `centroid_dmd` *(modal portion)* | Dynamical / `activation_dmd` (REQ_117) | reorganization |
| `centroid_dmd` *(modal portion, weight-space mirror)* | Dynamical / `parameter_dmd` (REQ_117, blocked on REQ_118) | reorganization |
| `neuron_activations` | Universal Core / `activation_snapshot` | reorganization |
| `attention_patterns` | Universal Core / `activation_snapshot` | reorganization |
| `neuron_group_pca` | Universal Core / `group_geometry` | reorganization |
| `freq_group_weight_geometry` | Universal Core / `group_geometry` | reorganization |
| `intragroup_manifold` | Universal Core / `group_geometry` | reorganization |
| `neuron_dynamics` | Universal Core / `neuron_dynamics` (rename pending) | retain |
| `dominant_frequencies` | Family Basis / `weight_basis_projection` | reorganization |
| `attention_fourier` | Family Basis / `weight_basis_projection` | reorganization |
| `neuron_fourier` | Family Basis / `weight_basis_projection` | reorganization |
| `attention_freq` | Family Basis / `activation_basis_projection` | reorganization |
| `neuron_freq_clusters` | Family Basis / `activation_basis_projection` | reorganization |
| `fourier_frequency_quality` | Family Basis / retain | retain |
| `fourier_nucleation` | Family Basis / retain | retain |
| `transient_frequency` | Family Basis / retain (open) | retain |
| `coarseness` | retire | retire |
| `landscape_flatness` | Universal Core / Dynamical landscape | retain |
| `gradient_site` | Dynamical / `gradient_site` (refactored) | refactor |
| `input_trace` | Universal Core / `input_trace` | retain |
| `input_trace_graduation` | Universal Core / `input_trace_graduation` | retain |

**Net:** 25 existing analyzers → ~16 target analyzers + 4 planned-new entries.

---

## Planned new analyzers

Capabilities with no existing predecessor:

- **`neuron_grouping`** — data-driven clustering primitive. Universal Core. Unlocks the basis-independent shape of `group_geometry`.
- **`trajectory_metrics`** — second-order trajectory geometry (acceleration, curvature, torsion). Dynamical / Trajectory.
- **`hessian_topk`** — Hessian top-k eigenvalues via Lanczos+Hvp. Dynamical / Landscape.
- **`cross_site_coupling`** — phase-lock and synchrony between sites. Dynamical / Operator.
- ~~**`parameter_dmd`** — specified in REQ_117; blocked on REQ_118 (`neuron_grouping`). Dynamical / Operator.~~ Shipped 2026-05-09.
- ~~**`neuron_grouping`** — specified in REQ_118.~~ Shipped 2026-05-08.
- **`lissajous_fit`** — already specified in REQ_111 as a research-active addition. Dynamical / Phase-space fits.
- **`saddle_center_center_fit`** — saddle-center-center linearization fit on `representation_trajectory`. Dynamical / Phase-space fits.
- **`saddle_transport_sigmoidality`** — already specified in REQ_111 as a research-active addition. Dynamical / Phase-space fits.

Together these represent the dynamical lean-in: they shift miscope from a snapshot-and-trajectory tool toward a tool that observes *how the system moves, what it moves through, and what phase-space structure it appears to imply*.

---

## Iterative release strategy

The Atlas supports incremental publication. The first PyPI release establishes a clean baseline that subsequent releases extend. The Atlas itself is the public commitment: "here's what's stable in v1, here's what's planned for vN, here's the conceptual structure they fit into."

Suggested v1 PyPI scope (the **clean baseline**):

- **Universal Core** — fully functional. Refactors landed (renamed, REQ_109-grounded). Reorganizations landed where the new shape is well-understood.
- **Family Basis Projections** — modadd family populated. Other families documented as TBD.
- **Dynamical Proxies** — partial:
  - Trajectory geometry: velocity (existing); `trajectory_metrics` planned.
  - Landscape geometry: `landscape_flatness` (existing); `hessian_topk` planned.
  - Operator dynamics: `gradient_site` refactored; `parameter_dmd` (REQ_073) and `cross_site_coupling` planned.
  - Phase-space fits: all planned (`lissajous_fit`, `saddle_center_center_fit`, `saddle_transport_sigmoidality`). Subset may ship in v1 if the Class Centroid PCA write-up depends on them.

Subsequent releases extend the baseline rather than disrupting it. External researchers landing on v1 see a working tool with explicit documentation of what's coming. Honesty over completeness.

---

## Cross-references

- [PROJECT.md](../PROJECT.md) — architectural constraints (universal views, family context providers).
- [REQ_103: PyPI Publication Hardening](requirements/active/REQ_103_pypi_publication_hardening.md) — the Atlas is the document a researcher reads first after install.
- [REQ_106: Analysis Layer Architecture](requirements/active/REQ_106_analysis_layer_architecture.md) — analyzer layering (extract → transform → load).
- [REQ_107: Discoverability Registry](requirements/active/REQ_107_discoverability_registry.md) — programmatic registry. The Atlas and the registry complement each other: the registry enumerates fields programmatically; the Atlas explains the territory in prose.
- [REQ_109: Measurement Primitives](requirements/staging/REQ_109_measurement_primitives.md) — primitives library. Every Atlas analyzer (existing or planned) consumes REQ_109 primitives for its transform step.
- [REQ_110: Lakehouse Surface](requirements/active/REQ_110_lakehouse_surface.md) — tabular output contract. Atlas analyzers respect it where applicable.
- [REQ_111: Parallel Analyzer Buildout](requirements/active/REQ_111_parallel_analyzer_buildout.md) — to be rescoped: covers `refactor` bucket entries (parity validation meaningful). Reorganization-bucket and new-bucket entries get their own scoped REQs.
- [REQ_117: DMD Reorganization](requirements/active/REQ_117_dmd_reorganization.md) — canonical home for `activation_dmd` and `parameter_dmd` (both shipped 2026-05). Supersedes REQ_073; absorbs the Research Claude drafts that proposed the windowed treatment. Includes validation outcomes per-variant in the Notes section.
- [REQ_118: Neuron Grouping Primitive](requirements/active/REQ_118_neuron_grouping.md) — prerequisite for REQ_117's parameter track (shipped 2026-05). Canonical home for `neuron_grouping`.
- [REQ_073: Weight-Space DMD](requirements/active/REQ_073_weight_space_dmd.md) — superseded by REQ_117. Retained for archaeology.
- [Variant Atlas](variant_atlas.md) — companion document for variants studied across these analyzers. The Analysis Atlas catalogs lenses; the Variant Atlas catalogs models being studied.
- [REQ_055: Attention Head Phase Analysis](requirements/active/REQ_055_attention_head_phase_analysis.md) — possibly overlaps with `cross_site_coupling`; coordinate scope when implementing.
- [REQ_102: Analyzer Deprecation](requirements/active/REQ_102_analyzer_deprecation.md) — handles retirement of analyzers marked retire-bucket.
- [Roadmap_Analysis_rough.md](requirements/Roadmap_Analysis_rough.md) — superseded *Analysis Catalog* section. Other sections remain valid in that document.

---

## Candidate derived metrics (surfaced during REQ_117 work)

Concrete diagnostic claims that emerged during REQ_117 inspection but are not yet implemented as standalone analyzers. Each could become a small follow-up REQ if the empirical pressure justifies it.

- **W_in / W_out residual peak lag** per (variant, group). The empirical observation that W_in and W_out reorganize on independent timelines, with the lag being a per-variant signature, could become a derived per-(variant, group) scalar. Sign and magnitude might separate clean grokkers from rebounders from incomplete-descenders.
- **MLP–attn residual peak lag**, an analogous scalar at the activation site level. Currently provisional on dense-checkpoint variants only; needs balanced validation. See `findings_mlp_leads_second_descent.md` (auto-memory).
- **W_out abandonment-shape signature** per transient frequency. The "sharp W_out drop after the abandonment peak" pattern noted on `p101/s999/ds598` group 4 may generalize to other transient frequencies; if it does, it's a candidate metric for *predicting* abandonment from W_out shape before it completes.
- **Activation-vs-parameter residual decoupling** — events visible in `activation_dmd` without a corresponding `parameter_dmd` peak (or vice versa). Candidate neural-collapse detector at end-of-training; first observed on `p109/s485/ds598`.

- **DMD-derived phase windows** *(replaces heuristic phase markers in `variant_summary.json`)*. Current phase markers (`first_descent`, `plateau`, `cascade`, `second_descent`, `final`) are heuristic-derived from loss-curve features. Windowed DMD residual peaks are mathematically derived from the same trajectory data and don't depend on tuning thresholds against an external loss signal. To make this a replacement, two design problems need solving: (1) **aggregation rule** — DMD boundaries are per-(site, group, matrix); a single "second descent began" boundary requires a rule (union of all sites? `resid_post` only as canonical? majority across sites?). (2) **semantic labeling** — current markers carry character claims (`plateau` vs `cascade`), not just timing claims; DMD gives boundaries but not classification of what happens between them. The semantic labels would be derived (e.g., low residual + flat = plateau; rising residual = cascade), and the derivation rule needs to be made explicit.

- **Parameter trajectory loop closure**, quantified. See [variant_atlas.md → Unquantified per-variant observations](variant_atlas.md) for the geometric description (a saddle-shaped path in `parameter_trajectory_pca` PC1×PC2×PC3 with apparent self-intersection in the PC2×PC3 stable plane, distinct from the Class Centroid PCA self-intersection). Three concrete metric candidates, each at different cost / commitment levels:
    1. **Minimum endpoint-to-trajectory distance in PC2×PC3** — for the final-epoch parameter point, compute the minimum distance to every earlier point along the trajectory. Per-variant scalar; doesn't commit to the saddle interpretation; cheapest first step.
    2. **Saddle-center-center fit on `parameter_trajectory_pca`** — the same fit shape Atlas already plans for `representation_trajectory` (one unstable real eigenvalue + two oscillation pairs). If the fit converges with low residual, the saddle reading is supported; if not, the geometric impression doesn't survive the math.
    3. **Trajectory near-passes graph** — nodes are epochs, edges are "this pair of epochs are close in PC2×PC3." Healthy variants would show a single dominant near-pass connecting an early epoch to a late epoch; failure modes would show different topology.

These are tracked rather than promoted to first-class Atlas entries because they are derivations on top of existing analyzers' outputs, not new analyzers themselves. If any becomes load-bearing for a research narrative, it earns its own REQ.

## Open questions

- **Generalization to non-classifier tasks.** The class-manifold framing in `representation_geometry` assumes discrete output classes. Induction heads, popcount, integer sqrt — output spaces don't decompose cleanly into classes. Defer until the workbench actually trains a non-classifier. Worth not promising too much in the public API shape.
- **Scope of `neuron_grouping`.** Should grouping operate on activations, weights, or both? Probably both, configurable. Validate during implementation.
- **`coarseness` retirement.** Verify that `activation_basis_projection` outputs preserve the blob-vs-plaid signal before retiring.
- **`transient_frequency` generalization.** Whether to lift the analyzer to "transient groups" (basis-independent) or keep it Fourier-specific. Defer until `neuron_grouping` lands.
- **`gradient_site` integration depth.** How tightly should it integrate with the artifact pipeline given its current direct-checkpoint loader? Trade-off: pipeline integration is cleaner; direct loading lets it operate on any checkpoint set without prior analysis runs.
- **`neuron_dynamics` rename.** Conceptually becomes `group_membership_dynamics` after `neuron_grouping` generalizes the grouping. Defer the rename — premature generalization without the grouping primitive in hand is just churn.
- **Phase-space fits scope.** Initial v1 entries are Lissajous, saddle-center-center linearization, and sigmoidal transit. As classical dynamical-systems tooling matures in the project, additional fits (Floquet stability, manifold computation, basin-of-attraction characterization) may join this sub-category. Defer planning until the initial three operate cleanly on canon variants.
- **Equilibrium and segment identification.** `saddle_center_center_fit` needs an equilibrium location; `saddle_transport_sigmoidality` needs segment boundaries. v1 accepts caller-supplied boundaries (manual or notebook-derived). A fixed-point-detection or segment-discovery primitive may emerge later.
