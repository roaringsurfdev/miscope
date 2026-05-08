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

### `neuron_grouping`
**Status:** planned-new | **Bucket:** new

Data-driven clustering of neurons by learned behavior. Input: per-neuron activation profiles or weight signatures. Output: group assignments + group centroids + dispersion metrics.

**This is a missing primitive.** Today, neuron grouping happens implicitly via Fourier dominant frequency (`neuron_freq_norm` from `neuron_dynamics`), which is task-specific. A generic grouping primitive lets the geometry analyzers operate on any partition; modadd family can override with a Fourier-based grouping where appropriate.

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

#### `activation_dmd`
**Status:** existing-rename | **Bucket:**: reorganization

DMD applied to activation class residues. Need rework to properly capture the different dynamical regimes present in different models. A refactor would include windowed DMD to capture regime changes (spikes in residuals) that feed into each regime's separate DMD analysis.

#### `parameter_dmd` *(home: REQ_073)*
**Status:** planned-new | **Bucket:** new (carries forward `centroid_dmd`'s modal portion)
DMD applied to weight trajectories rather than activations.

REQ_073 already specifies this work in detail — the Atlas registers it as the canonical home for the modal-dynamics analyzer.

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
| `centroid_dmd` *(modal portion)* | Dynamical / `parameter_dmd` (REQ_073) | reorganization |
| `centroid_dmd` *(modal portion)* | Dynamical / `activation_dmd` | reorganization |
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
- **`parameter_dmd`** — already specified in REQ_073. Dynamical / Operator.
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
- [REQ_073: Weight-Space DMD](requirements/active/REQ_073_weight_space_dmd.md) — canonical home for `parameter_dmd`.
- [REQ_055: Attention Head Phase Analysis](requirements/active/REQ_055_attention_head_phase_analysis.md) — possibly overlaps with `cross_site_coupling`; coordinate scope when implementing.
- [REQ_102: Analyzer Deprecation](requirements/active/REQ_102_analyzer_deprecation.md) — handles retirement of analyzers marked retire-bucket.
- [Roadmap_Analysis_rough.md](requirements/Roadmap_Analysis_rough.md) — superseded *Analysis Catalog* section. Other sections remain valid in that document.

---

## Open questions

- **Generalization to non-classifier tasks.** The class-manifold framing in `representation_geometry` assumes discrete output classes. Induction heads, popcount, integer sqrt — output spaces don't decompose cleanly into classes. Defer until the workbench actually trains a non-classifier. Worth not promising too much in the public API shape.
- **Scope of `neuron_grouping`.** Should grouping operate on activations, weights, or both? Probably both, configurable. Validate during implementation.
- **`coarseness` retirement.** Verify that `activation_basis_projection` outputs preserve the blob-vs-plaid signal before retiring.
- **`transient_frequency` generalization.** Whether to lift the analyzer to "transient groups" (basis-independent) or keep it Fourier-specific. Defer until `neuron_grouping` lands.
- **`gradient_site` integration depth.** How tightly should it integrate with the artifact pipeline given its current direct-checkpoint loader? Trade-off: pipeline integration is cleaner; direct loading lets it operate on any checkpoint set without prior analysis runs.
- **`neuron_dynamics` rename.** Conceptually becomes `group_membership_dynamics` after `neuron_grouping` generalizes the grouping. Defer the rename — premature generalization without the grouping primitive in hand is just churn.
- **Phase-space fits scope.** Initial v1 entries are Lissajous, saddle-center-center linearization, and sigmoidal transit. As classical dynamical-systems tooling matures in the project, additional fits (Floquet stability, manifold computation, basin-of-attraction characterization) may join this sub-category. Defer planning until the initial three operate cleanly on canon variants.
- **Equilibrium and segment identification.** `saddle_center_center_fit` needs an equilibrium location; `saddle_transport_sigmoidality` needs segment boundaries. v1 accepts caller-supplied boundaries (manual or notebook-derived). A fixed-point-detection or segment-discovery primitive may emerge later.
