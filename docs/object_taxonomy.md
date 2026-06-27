# Mechanistic Object Taxonomy

**Status:** Draft lens (2026-06-19). Repo-grounded companion to
`docs/requirements/drafts/mech_interp_domain_model/REQ_001`. Built from the
*live* registry (`miscope.registry.index()`), not from memory.

## The inversion

Today the project groups analyzer outputs two ways:

- **by Analyzer** — the registry: `analyzer → its outputs`.
- **by coord-signature** — the warehouse: a field keyed by `(variant, epoch, site)`.

The coord-signature grouping is already noun-organized at the *storage* layer
(REQ_110D: `neuron_frequency` became the single conformed `(epoch,neuron)→freq`
source; REQ_144: the variant registry is a pure view over `variant_outcomes`).
What churns is **authoring and reasoning**, because the only *named, enumerable*
index is analyzer-first. When two analyzers overlap, we refactor the analyzer —
because the analyzer is the only noun we have.

This taxonomy makes the **third grouping** first-class: attributes grouped by the
*mechanistic object they describe*.

> **Objects are named after the mechanism. Analyzers are named after the
> computation.** The mechanism (AttentionHead, QKCircuit, FrequencyMode) changes
> on the timescale of the literature and our findings. The computation changes
> every time we improve a method. Organize by the slow-changing axis and the
> churn stops: splitting `FourierAnalyzer` into three becomes invisible to anyone
> reading `Site`, as long as the attributes still land in the same home.

**The atom is the *attribute*.** An *object* is a named bundle of attributes
sharing a coord key. An *analyzer* is a visitor that fills some attributes. Both
object and analyzer are groupings over the same atoms, along different axes.

### What earns objecthood (litmus)

A thing is a first-class **object** when it has *intrinsic identity and measured
attributes an analyzer writes to it*. A thing is a **query**, not an object, when
it is fully reconstructible by a predicate/join over other objects' attributes —
it carries no measurement of its own.

- `HeadPair` (heads sharing a frequency) → **query**. It has no attribute you'd
  measure independently; it is a self-join on shared frequency above a power
  threshold over `attention_head_frequency` (keyed `(variant, epoch, head,
  frequency)`, with `power`). Modeling it would ossify a single-family finding we
  don't yet understand or know to generalize. **Promote on evidence** (recurs
  across architectures), never on one finding.
- `FrequencyMode` → **object**. Survives the same knife: it has frequency-keyed
  *measured* attributes of its own (`active_frequencies`, `energy`, committed
  status). Its *membership* (which heads/neurons) is the query-time part; the mode
  itself is measured.

This is REQ_001's "cross-variant comparison is query-time, not schema" pushed one
level down: emergent **relationships** live in queries; measured **entities** live
in the schema. (Note `attention_head_frequency` doesn't exist yet — it's the CHEAP
QK move: widen `weight_basis_projection` to emit per-head `power` keyed by
frequency, not just `dominant_frequency`. The pairing query then runs with no new
object at all.)

### What this is NOT

Not a new data layer. The object catalog is a **reprojection of the existing
registry**, grouped by coord-signature instead of by analyzer (`GROUP BY
coord-signature`), plus a layer of mechanism-grounded *names* the coord-tuples
don't carry. Objects are reader projections over warehouse coord-signatures (the
way `variant.warehouse` / `query.open_variant` already are) — never a second
persistence path (architectural invariant #3: storage layout is internal to the
API).

### Architectural notes to surface

- **Families gain object-type ownership.** Today "families are context providers,
  not view owners." Naming family-scoped objects (FrequencyMode exists only for
  modular-arithmetic families) *extends* invariant #2 — families would own
  object-type *definitions*. Defensible, but a deliberate change to flag.
- **Versioning overlaps REQ_145.** Object-scoped staleness (drafts REQ_003/004)
  substantially re-derives REQ_145's signature-based freshness. The one genuinely
  new idea to lift: a signature change can't distinguish "recomputed, still
  comparable" from "recomputed, now incomparable" — that needs a
  *developer-declared semantic break*.

---

## Three temporal types

Inherited from REQ_001, confirmed by the coord-signatures that actually exist:

- **Snapshot** — keyed with `EPOCH`. Point-in-time measurement of an object.
- **Trajectory** — keyed by `VARIANT` (no `EPOCH`), spanning checkpoints. A
  property of the path through weight/activation space.
- **Event** — a *transition* located in training time. **Currently has no home**
  — events are buried as scalar attributes on other objects (see below).

---

## The taxonomy

Maturity legend: **rich** (deep coverage) · **solid** · **thin** ·
**near-empty** · **empty** (named, no attributes yet) · **homeless** (attributes
exist or are designed, no object yet).

Every "current attributes" list below is verbatim from the live registry. The
producing analyzer is in parentheses.

### Snapshot objects

> **DECISION (2026-06-19): the overloaded `site` coord splits into two objects —
> `ActivationSite` and `WeightMatrix`.** The current `(variant, epoch, site)`
> grouping (36 attrs) is our de-facto primary object, but `site` mixes
> *activation locations* (`resid_post`, `mlp_out`, `attn_out`) with *weight
> matrices* (`W_in`, `W_Q`, `W_E`). These are different *kinds* of object, not two
> flavors of one. **Named in the field's own vocabulary, not a coined symmetric
> `-Site` pair** — a matched `WeightSite`/`ActivationSite` would smuggle back a
> fake shared `Site` parent and re-import the very conflation we're removing.
>
> *Earned on three independent seams that all fall on this boundary:*
> (1) **attribute kind** — an activation site's attributes are
> input-distribution-dependent (exist only relative to a probe); a weight
> matrix's are intrinsic to the parameters; (2) **the SNR/energy guard** —
> "exclude sites with class-centroid energy ≈ 0" applies *only* to activation
> sites (`resid_pre` is c-independent by construction) and its absence caused a
> retracted finding; (3) **DMD variant** — `resid_post = resid_pre + writes`
> (additive identity → DMDc) vs orthogonal weight subspaces (→ joint DMD). What
> they "share" (SVD runs on any matrix) is the universal-instrument pattern, not
> a shared object type. `Coord.GROUP_TYPE` already half-encodes the split
> (`weight_matrix` vs `activation_site`).

#### `ActivationSite` — `(variant, epoch, site)` · **rich** (activation half)
Runtime activation locations (`resid_post`, `mlp_out`, `attn_out`). The Elhage
residual stream **is** `ActivationSite[resid_post]` — naming win, not new compute.
Attributes are probe-relative.

- geometry (`repr_geometry`): `circularity`, `center_spread`, `mean_radius`,
  `mean_dim`, `pca_var_pc1..3`, `fisher_mean/min`, `fisher_argmin_*`, `snr`,
  `centroids`; detail facet `(…, row_id)`: `dimensionality`, `radii`
- freq-norm (`activation_frequency_norm`): `freq_norm`
- alignment (`centroid_fourier_alignment`): `fourier_alignment`

#### `WeightMatrix` — `(variant, epoch, site)` · **rich** (weight half)
Parameter tensors (`W_E/W_Q/W_K/W_V/W_O/W_in/W_out/W_U/W_pos`). Attributes are
intrinsic to the parameters. (Subsumes the previously-proposed `WeightGroup`.)

- fourier basis (`weight_basis_projection`): `power`, `magnitudes`, `phases`,
  `fractional_power`, `cos/sin_coeffs`, `cos_cos/cos_sin/sin_cos/sin_sin_coeffs`,
  `dominant_frequency_pair`
- spectra (`weight_spectra`): `u`, `vt`
- gradient (`gradient_site`): `magnitude` *(adjudicate: weight-space gradient)*
- **Homeless inbound** (escape note): `rel_velocity` (`stream_weight_velocity`,
  per weight group); the per-group velocity-peak epoch (lead-lag E→attn→MLP
  ordering) — itself an **Event**.

> **The split resolves the dump's worst collision for free.** On `(v,e,site)`,
> `circularity`/`center_spread`/`fisher_mean`/`fisher_min`/`mean_radius`/`snr`
> were each declared by **both** `repr_geometry` and `freq_group_weight_geometry`.
> They were never the same attribute — `repr_geometry` measures *activation*
> centroids (→ `ActivationSite`), `freq_group_weight_geometry` measures *weight*
> group geometry (→ `WeightMatrix`/`FrequencyGroup×WeightMatrix`). The overloaded
> `site` coord was hiding two different objects under one name. A fourth seam on
> the same boundary.

#### `FrequencyGroup` — `(variant, epoch, group)` · **solid** (13 + facets)
A set of neurons sharing a dominant Fourier frequency. One of our most-developed
objects.

- manifold (`intragroup_manifold`): `a`, `b`, `c`, `r2_curvature`, `r2_linear`,
  `r2_quadratic`, `shape_int`
- grouping (`neuron_grouping`): `centroids`, `n_per_group`, `radii`
- pca (`neuron_group_pca`): `mean_spread`; facet `(…, row_id)`: `pc_var`
- × Site facet `(…, site, group)` (`freq_group_weight_geometry`):
  `dimensionality`, `f_top3`, `pr3`, `radii`

#### `Neuron` — `(variant, epoch, neuron)` · **thin** (4)
- `assignments`, `confidence` (`neuron_grouping`)
- `dominant_freq`, `max_frac` (`neuron_frequency_attribution`)
- **Homeless inbound** (escape note): `radial`, `tangential`, `norm`,
  `tangential_fraction` (`rotational_dynamics` — the headline escape detector);
  `power` (`neuron_activation_spectrum`, the detonator signal).

#### `FrequencyMode` — `(variant, epoch, frequency)` · **thin, TRANSVERSE** (4)
The family's true central object — and provably transverse. Its *identity* is
`frequency`, but most of its attributes live on **other objects' rows**:
`neuron_peak_freq` is keyed `(v,e)`, `dominant_freq` is `(v,e,neuron)`,
`group_freqs` is `(v,group)`, `energy` is `(v,e,site,frequency)`. The frequency
is the **join target** tying neurons, groups, and sites together — which is why
it cannot be a leaf attribute hanging off a single circuit.

- here: `active_frequencies` (`fourier_frequency_quality`); `frequencies`
  (× **3 collision**: `activation_frequency_norm`, `fourier_nucleation`,
  `weight_basis_projection`)
- × Site facet `(…, site, frequency)`: `energy` (`gradient_site`)
- **`FrequencyMode` and `FrequencyGroup` are two views of nearly the same
  mechanism and are currently unrelated in the schema.** `group_freqs` is the
  missing edge. The membership-churn confound (escape note: neuron→freq
  membership is time-varying, pinned at `reference_epoch` → artifacts on p101)
  is exactly this missing time-varying edge between FrequencyMode and Neuron.

#### `InputTrace` — `(variant, epoch, row_id)` · per-input lens (5)
`row_id = a·p + b` (per-input). "What does the model compute for *this input* at
epoch N." One object on the emergence spine — a per-input instrument onto
representational structure, not the research goal itself.

- `confidence`, `correct`, `predictions`, `split` (`input_trace`)
- `delta_losses` (`landscape_flatness`)

#### `AttentionHead` — `(variant, epoch, site, head, row_id)` · **near-empty** (2)
The canonical community object is the *least* first-class thing we have. `HEAD`
appears in exactly one coord-signature, with 2 attributes.

- `dominant_frequency` (`weight_basis_projection`), `sv` (`weight_spectra`)
- Attention `patterns` are a tensor blob keyed `(v,e)` (`attention_patterns`) —
  **the head axis is buried inside the blob**, not exposed as a coordinate.
- **Homeless inbound**: `D = vel(attn QK)/median(...)` (`attention_divergence`).
  The head-pairing structure is **not** a schema gap — it's a self-join query over
  the (not-yet-built) `attention_head_frequency` table; see the objecthood litmus.

#### `VariantCheckpoint` — `(variant, epoch)` · grab-bag (41)
The parking lot: raw weight tensors (`W_E…W_U`, `parameter_snapshot`) plus
scalars that *belong elsewhere* — `neuron_peak_freq`/`neuron_committed_count`
(→ FrequencyMode/Neuron), `patterns` (→ AttentionHead), `fisher_mean/min`
(→ Site), `coverage_hard`/`quality_score` (→ FrequencyMode). Should shed
attributes to the named objects above; what remains is genuinely
checkpoint-level (loss, raw parameter tensors).

### Trajectory objects (key = `variant`, span epochs)

#### `ActivationSiteTrajectory` — `(variant, site)` · **rich** (21 + 15 detail) — DMD home
(Same seam: `(variant, site)` here is activation-side — `global_centroid_pca` and
`activation_dmd`. Weight-side trajectories live on `WeightMatrix`/`FrequencyGroup`
via `parameter_dmd`.)
`global_centroid_pca` (`basis`, `mean`, `projections`, `explained_variance_ratio`)
and `activation_dmd` (`trajectory`, `per_regime__*`, `windowed__*`, `tracks__*`,
`regimes__*`). **Homeless inbound** (escape note): `cka_to_final`
(`representation_similarity`); `offcircle_dev`, `spectral_radius`
(`spectral_persistence`, derived over existing DMD); `resid_post max|λ|`
(`output_spectral_escape`, the clean detector).

#### `FrequencyGroupTrajectory` — `(variant, group[, site])` · **rich** (9 + 16 + 14)
`group_freqs`, `group_sizes` (× **3 collision**: `freq_group_weight_geometry`,
`intragroup_manifold`, `neuron_group_pca`); `group_n_neurons`, `populated_groups`
(`parameter_dmd`); `projections`, `explained_variance[_ratio]`
(`parameter_trajectory`); × Site facet: full `parameter_dmd` regime/windowed
decomposition.

#### `NeuronTrajectory` — `(variant, neuron)` · thin (3)
`commitment_epochs`, `switch_counts` (`neuron_dynamics`), `neuron_group_idx`
(`neuron_group_pca`). Note `commitment_epochs` is really an **Event** wearing a
trajectory key (see below).

#### `FrequencyModeTrajectory` — `(variant, frequency)` · thin (1)
`key_frequencies` (`gradient_site`).

#### `VariantTrajectory` — `(variant)` · (10)
`neuron_group_pca` tensors, `parameter_dmd` (`reference_epoch`, `n_groups`),
`threshold` (`neuron_dynamics`). Facet `(v, row_id)`: `graduation_epochs`
(`input_trace_graduation` — also an Event), `window_epochs` (`gradient_site`).

---

## New objects the escape work forced into the open

These are designed/half-populated but have **no object** — attributes bolted onto
whatever coord was handy. (Source: escape-regime push-down note, 2026-06-16,
local.)

#### `DecisionQuery` — keyed by the `=`-query · **homeless, half-started**
Output-side health at the decision position. **REQ_151 already populated its
first attribute** (searchable attention-entropy field at the decision query) —
we started filling an object we hadn't named.
- inbound: `logit_gap`, `absorbed_fraction`, `log_Z`, `correct_margin`,
  `logit_cosine_to_ref` (`output_logit_health`); `resid_post max|λ|` in the
  escape zone (`output_spectral_escape`).

> `WeightGroup` (an earlier proposal) is **subsumed by `WeightMatrix`** — it was
> just the weight half of the overloaded `site`, now a first-class object above.
> `stream_weight_velocity`'s `rel_velocity` lands on `WeightMatrix`.

#### `Event` — a transition in a specific object's trajectory · **homeless, ubiquitous**
Currently every "when did X transition" scalar is buried inside another object:
`commitment_epochs` (on Neuron), `graduation_epochs` (on InputTrace),
`window_epochs` / `second_descent_onset` (anchors), `regimes__boundary_indices`
(on DMD), per-group velocity-peak epochs. The escape work's own cross-cutting
constraint — "escape-zone window `[onset+18k, last]`, anchored on
`second_descent_onset`" — is an Event object passed around as loose params.

**Key keying decision (from the p109 finding):** an Event must be keyed
`(variant, object-ref)`, **not** `(variant)` alone. The p109 late reorganization
is *silent in all 5 aggregate instability diagnostics* but visible
per-frequency/per-head. An event keyed only by Variant *is* the aggregate that
hides it. "These solo-freq heads de-sharpened at ~29k" is an event on specific
heads, not a Variant property.

## Forward-capacity objects (the taxonomy aims at the general transformer)

**Design stance (2026-06-19):** the program builds toward general multi-layer
transformers; the 1-layer modular-addition model is *foundation + calibration*
(and likely anomalous in how richly it rewards study — treat the depth of work
here as foundation-building, not the steady state). So the taxonomy is designed
toward the known future state, **not** trimmed to fit the toy model. Objects the
toy Variant doesn't instantiate are kept first-class and *gated*, never pruned.

Two gating axes (both "not universal," different reasons):
- **architecture-gated** — needs ≥2 layers / longer sequences. *Structural*, a
  property of the Variant (its `n_layers`, `n_ctx`).
- **family-context** — Fourier modes, frequency groups. *Semantic*, provided by
  the family (invariant #2).

**Tractability is a third axis, tied to the calibration→deployment inheritance
framework** (`project_analysis_inheritance_framework`; Tier 0–3). Some lenses
won't scale. The toy/calibration phase is where we identify **cheap proxy lenses
that correlate with expensive deep ones**, so at scale we run the proxy and
reserve the deep lens for narrow application. Attributes should carry a cost/scale
tier, not just an object home.

### Active forward agenda — instantiated in 1-layer (within-head), instruments mostly owned

`QKCircuit` / `OVCircuit` are *not* architecture-gated — a 1-layer model has them
(within-head composed virtual weights). They're empty only because we never
materialized the composed weight. The unsaddled conversation independently named
them, which is a real "haven't looked here" signal. The high-value/low-cost move:
**materialize composed `W_Q^T W_K` / `W_O W_V` and point existing instruments at
them.** Tags: **HAVE** (own it) · **CHEAP** (own instrument, repoint) · **NEW** ·
**DELTA** (our data already says more).

**`QKCircuit` — `(variant, epoch, head)` · composed `W_Q^T W_K`**
- Fourier spectrum of the *composed* QK — **CHEAP** (`weight_basis_projection`,
  repointed): dominant freq, power, doubled-vs-solo.
- cross-head QK frequency redundancy — **DELTA**: the flat doubled-freq pair
  (QK cos≈1) vs excursing solo-freq pair (`attention_head_pairing.ipynb`). **Not
  a `HeadPair` object** (fails the objecthood litmus) — a self-join query over
  `attention_head_frequency`. The schema work is the base table, not the pairing.
- rank / effective rank — **CHEAP** (`weight_spectra` SVD of composed QK).
- attention entropy at the decision query — **HAVE** (REQ_151, on
  `DecisionQuery`); the *de-sharpening event* is its trajectory.
- operand symmetry (a↔b) — **NEW**, family-context (addition commutativity).
- pattern consistency across inputs — **NEW** (`InputTrace × AttentionHead`).

**`OVCircuit` — `(variant, epoch, head)` · composed `W_O W_V`**
- Fourier spectrum of composed OV — **CHEAP** (`weight_basis_projection`).
- rank / effective rank — **CHEAP** (`weight_spectra`).
- per-head direct logit attribution — **NEW**; partial start in
  `output_logit_health` (`logit_gap`, `correct_margin`), homeless.
- OV↔QK frequency dissociation — **DELTA**: the read/write split (solo-freq QK
  heads vs doubled-freq MLP writes coming apart late).
- unembed / Fourier-mode alignment of OV output — **HAVE-ish**
  (`centroid_fourier_alignment` analog).

### Architecture-gated — kept first-class, instantiated by deeper Variants

Not in a 1-layer model; core to the destination. Held in the taxonomy so the
schema is built toward them (do **not** delete as "N/A").

- `InductionHead` / `PreviousTokenHead` / `DuplicateTokenHead` — discovered head
  motifs; need ≥2 layers + longer sequence. (Here, the earned head-type axis is
  instead the **frequency-pairing role** — the family-context analog.)
- cross-head / cross-layer composition (Q/K/V-composition scores; composed
  virtual weights *across* heads) — need depth.
- layer-wise residual drift — needs depth.
- `MLPLayer` as key-value memory at scale — partially active now: our
  `FrequencyGroup` / `neuron_frequency` is the toy-model instance of the KV-memory
  story (neurons keyed by detected frequency).

### Already ours, just needs naming

- `ResidualStream` = `ActivationSite[resid_post]`. The Elhage framing (one shared
  channel; attn+MLP as writes) is exactly the escape note's DMDc verdict.
  Recognizing it *as* the residual stream is a naming win, not new compute.

---

## Collisions to adjudicate

The duplicate-attribute problem the analyzer-ownership axis hides. Two analyzers
declaring the same field on the same coord-signature: is it the same attribute
computed two ways (a semantic collision to merge) or two different things sharing
a name (a rename)? The object makes the question answerable; today it's invisible.

- `(variant, epoch, site)`: `circularity`, `center_spread`, `fisher_mean`,
  `fisher_min`, `mean_radius`, `snr` — each by **both** `repr_geometry` and
  `freq_group_weight_geometry`. **RESOLVED by the ActivationSite/WeightMatrix
  split** (2026-06-19): `repr_geometry` → `ActivationSite`,
  `freq_group_weight_geometry` → `WeightMatrix`/`FrequencyGroup×WeightMatrix`.
  Never the same attribute; the overloaded `site` coord hid two objects.
- `(variant, epoch, frequency)`: `frequencies` × 3.
- `(variant, group)`: `group_freqs` × 3, `group_sizes` × 3.
- `(variant, epoch)`: `prime` × 3, `n_groups` × 2, `fisher_*` also here *and* on
  Site, `epochs` × 11 (benign — the axis, but every analyzer re-declares it as if
  it owned it).

The `frequency_structure` entry in the escape note —
*"neuron_frequency fields may already exist; extend rather than duplicate"* —
is this collision caught *in the act of about to happen again*. The author had to
manually remember to check, because the object wasn't there to make the home
obvious.

---

## DMD: a universal instrument parameterized by the object

DMD today is frozen into two analyzers (`activation_dmd` keyed `site`,
`parameter_dmd` keyed `group, site`). But DMD is an operation on an
`(object, time, features)` tensor — any object owning a slice through tensor data
could carry a DMD trajectory.

The escape work proves something stronger than mechanical reuse: **the object's
structure selects which DMD variant is valid.**

- A `Site` (`resid_post`) is *not* a free body — attn_write and mlp_write
  subspaces overlap **0.79**, and `resid_post = resid_pre + attn_write +
  mlp_write` is an additive identity. Correct instrument: **DMDc** (one state +
  feedback inputs), not joint DMD.
- A `FrequencyGroup` *is* independent — `W_in` column subspaces across groups
  overlap **0.008** (orthogonal Fourier modes). Correct instrument: **joint DMD
  across groups** (off-diagonal blocks = inter-frequency competition).

Same operation, opposite valid form; the discriminator (subspace independence) is
a property of the object. DMD is a universal instrument (invariant #1: the
instrument doesn't change shape for what it's pointed at) whose *parameterization*
is read off the object. Once objects carry that metadata, `spectral_persistence`,
joint-group DMD, and DMDc stop being three analyzers and become **DMD pointed at
three objects.**

---

## Why this shakes loose the Analyzer Atlas

`docs/analysis_atlas.md` organizes "what exists / what's missing" by *analyzer* —
and analyzers overlap, so the atlas never sorts. Under the object lens the same
two questions become legible:

- **What exists** = which objects have rich attribute coverage (Site,
  FrequencyGroup[Trajectory], SiteTrajectory).
- **What's missing** = the empty/near-empty/homeless objects (AttentionHead,
  QKCircuit, OVCircuit, DecisionQuery, WeightGroup, Event) and the thin ones
  (Neuron, FrequencyMode).

An object-keyed atlas answers "what do we know about an AttentionHead?" with one
lookup instead of a scan across analyzers. The analyzer's role inverts: it
becomes a *back-reference* on each attribute (who fills it), not the primary
index.

---

## How this lands in the current warehouse

The warehouse already has the two-layer shape this taxonomy targets — the move is
to tilt the dial, not rebuild.

- **Designed semantic tables** (`warehouse/mapping_semantic.py`): conformed,
  *object-shaped* tables that **multiple analyzers feed via claims**
  (`claims_for(analyzer) → SemanticClaim(table, …)`; the writer row-unions every
  feeder of a table — `_table_feeders`). Today: `pca_results`,
  `shape_characterizations`, `neuron_frequency_attribution`, `frequency_spectrum`,
  `variant_outcomes`, `pca_projections`. **This is REQ_002's visitor pattern
  already implemented at the storage layer** — analyzers contribute attributes to
  an object table; they don't own a result blob.
- **Generic analyzer-named fallback** (`warehouse/writer.py`): fields no claim
  takes fall through to a table named after the analyzer. *This* is the
  "materialized copy of analyzer output" residue.

**Target state = grow the semantic layer, shrink the generic fallback.** The
object taxonomy is the target list of semantic tables; the generic fallback is the
backlog of un-promoted attributes. Promoting one = writing a `SemanticClaim` that
reshapes an analyzer field into an object table. Mechanism exists; 6 tables use it.

Two implementation notes:

- **`GROUP_TYPE` already encodes the `ActivationSite`/`WeightMatrix` split.**
  `repr_geometry.circularity` lands with `group_type=CENTROID_GROUP` (activation);
  `freq_group_weight_geometry.circularity` lands as a weight/frequency group. The
  dump collision is *already* disambiguated by the discriminator. So the split is
  a choice: promote `group_type` to separate object tables, or keep it as a
  discriminator column. **Lean: discriminator now; separate tables only when the
  attribute *sets* diverge** (weight-only SVD spectra vs activation-only CKA).
- **Tensors stay blobs.** An object table is the *columnar* projection. Tensor
  attributes (`W_E`, DMD modes, SVD `u`/`vt`, activations) hang off the object via
  the tensor catalog (`TensorRef`) — referenced, not inlined. Precise statement:
  *an object is a stable identity with **scalar columns populated by analyzers,
  plus references to its tensor attributes**.*

**Change profile** = the additive-vs-breaking versioning axis. Additive (new claim
→ new column/rows on a stable table; new object → new semantic table) is the
common case; the rare break is semantic recomputation of an existing attribute
(developer-declared, → REQ_145). Mostly-additive = schema-stable.

## Mapping to the four drafts

- **REQ_001** (domain object schema) ← this doc is its repo-grounded form. The
  objects are the coord-signatures; the temporal types are confirmed; the
  transverse `FrequencyMode` and the `(variant, object-ref)` Event keying are the
  two corrections the findings demand.
- **REQ_002** (analyzers as visitors) is ~60% done — REQ_107's
  `AnalyzerSpec.outputs` already declares object+attribute (as coords+name). The
  additive change: read those declarations object-first and route attributes to
  named homes; adjudicate the collisions above.
- **REQ_003/004** (versioning + staleness) — fold the *developer-declared
  semantic break* into REQ_145; retire the parallel versioning surface.
