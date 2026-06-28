# MIScope Data Model — Master Synthesis

**Status:** Validation draft (2026-06-28). The canonical data model — supersedes and **retires**
`docs/data-domain-model.md` (V1) and `docs/datamodel.md` (V2), both deleted. Reconciles them with
`docs/object_taxonomy.md` (the repo-grounded findings) and the live Store tier
(`packages/miscope/src/miscope/warehouse/`).

This is the contract to validate **before** building. It is the logical data model — objects,
keys, attribute homes — *plus* the decision procedure that tells us how to add to it as the
research moves. Read Part 0 first (the admission rule, plus the two lenses — Medallion and
Observability — that frame the whole store), then Part I (the one live tension — the hybrid store —
and the object apparatus the sieve leaves behind), before the tables: the tables only make sense
once the sieve and the lenses are in hand.

---

# Part 0 — What the Model Admits (the sieve)

Before the tensions, the one rule that decides whether a thing belongs in this model at all. Anyone
arriving for the *reasoning* behind the schema should start here.

**The admission rule — a front gate, a side gate, a corollary.** Three ways a thing can relate to
the model; only the first two yield objects.

**Gate 1 — Given-real → object, `STABLE` (the front gate).** Admitted directly, *no litmus*: any
object whose realness is **given** by something we already trust. Three sources — the three object
families:

- **Infrastructure objects** (Architecture, TaskType, Task, Family, Variant, Checkpoint, Probe, and
  the method/recipe registry) — realness from **provenance/definition**. The run happened, or we
  declared the spec; an external witness (the file tree, W&B, the config) agrees it exists.
- **Architecture objects** (residual stream, attention head, weight matrices, MLP neuron) —
  realness from the **model definition**. They exist because the architecture says so. (Transformers
  are the object of study here; non-transformer models in this codebase exist to *test theories
  about transformers*, not as architectures studied in their own right.)
- **Derived / theorem objects** (QK/OV circuits, the full circuits, the direct path, the TaskType's
  irrep basis) — realness from **mathematical derivation**. They are closed-form functions of given
  primitives; the identity is true by construction.

Given-realness needs no measurement: a theorem object is an object *whether or not anything has
measured it* — which is why the Layer 4 circuits are objects while still `empty`. They enter
`STABLE`: the definition isn't ours to defend.

**Gate 2 — Earned-real → run the litmus (the side gate).** A thing whose realness is **not** given
must earn its way in, and the litmus (decision-procedure step 3 + promotion criterion step 5)
decides the *form* it takes:

- It carries its own **irreducible measurement** — **non-reconstructible by join** over other
  objects *and* **gauge-invariant in the operand alone** → admitted as an **`EVOLVING` object**,
  held provisionally, rigor still owed on whether it's real. → **FrequencyGroup** (centroid spread,
  manifold curvature — joint-geometry of the neuron set that no per-neuron join reconstructs),
  **FrequencyMode** (its own `committed`/nucleation reading under the irrep basis).
- It is **fully reconstructible by join/predicate** over other objects → **query, forever.** →
  **HeadPair** (a self-join over a per-head frequency table), **"switching"** (a query over the
  top-k across epochs).

**Gate 3 — Transverse → query, regardless of certainty (the corollary).** A phenomenon whose
*identity is itself a join* or co-occurrence across other objects stays a query however certain we
are it's real — the p109 late reorganization, a head-pairing, a frequency shared across sites.
Earning realness (by intervention or prediction) does **not** promote it: **realness ≠ objecthood.**
The rule is about *form*, not *truth*.

> **The gate is the maturity tier.** Front gate → `STABLE` (definition given, not ours to defend);
> side gate (admitted) → `EVOLVING` (definition is an open field question, may reify a false frame).
> This *is* the Maturity axis (Part I); Coverage and Status move independently of it.

Everything that clears no gate is **measurement**, not an object: a fact at a coordinate, keyed
`(instrument, operand, coordinate-signature)`, stored at an address *without an object commitment*
(Tension 1) — a poorly-applied instrument is just a measurement no object ever claims. The membrane
between measurement space and an object's own columns is the promotion criterion (step 5): a
measurement becomes an attribute only when it is a gauge-invariant function of the operand alone.

**What this excludes — behavior.** Behavioral / posited objects — *feature, circuit-as-subgraph,
induction head, learned algorithm, clock/pizza* — clear no gate *to objecthood*: their realness is
neither given (Gate 1) nor backed by an irreducible measurement of their own (Gate 2's object case
turns on exactly that), and their identity isn't even a clean join. We **reproduce the behavior with
a query**, but never mint a table for it. (See the algebra-vs-behavior harvest in
[canonical_object_harvest.md](canonical_object_harvest.md): *harvest the algebra, flag the
behavior.*)

## Two lenses that frame the whole store

The sieve decides *what* is admitted; two standard data-engineering lenses describe *how* what's
admitted is laid out and observed. Both are orienting vocabulary, used throughout the tables.

**Lens 1 — Medallion tiers (the physical layout).** The store is a classic medallion stack, and the
tiers map one-to-one onto the write/read split resolved in Tension 1:
- **Bronze** — Analyzer output: `npz`/`safetensors` artifact deposits (raw, recompute-expensive,
  tensor-bearing). The durable truth; rebuilt only by re-running analysis.
- **Silver** — conformed, *object-shaped* tables in the Warehouse (the schema in Part III). Cheap,
  rebuildable projections of Bronze; where predicate/filter/join/cross-model comparison happens.
  Materialized derived tables (REQ_141) and the Event table live here too — **Silver holds
  materialized queries, not only given-real objects.**
- **Gold** — the View Catalog (REQ_047: `EpochContext` → `BoundView`). Refined, consumer-facing
  lenses over Silver. Out of scope for the schema tables below, but it is the tier the dashboard and
  fieldnotes actually read; named here so the stack is complete.

**Lens 2 — Observability (the two observation modes).** Operationally this is an observability
system over training, and every admitted attribute is observed in one of two modes:
- **Metrics** — regularly-sampled numeric values over training time, i.e. *time series*. These are
  the `[S]` Snapshot and `[T]` Trajectory scalar columns (a neuron's `dominant_freq` across epochs,
  loss across epochs). A metric that clears the promotion criterion becomes a column on an object.
- **Events** — discrete state-change notifications at a *point* in training time: the `[E]` rows (a
  neuron commits, a head de-sharpens, second descent begins). **Events are derived *from* Metrics** —
  located by reading a metric's time series (e.g. loss velocity/acceleration crossing a threshold).
  In sieve terms an Event is *earned/derived*, hence query-form — but we **materialize** it (Silver)
  because it is the cross-variant alignment spine (Layer 7).

---

# Part I — The Live Tension & the Object Apparatus (read first)

Earlier drafts framed this Part as *two* tensions. Part 0 has since dissolved the second: the
admission rule already decides which domain objects are real and which stay queries, so "the domain
objects are part of the research" is no longer a standing tension — it is **settled by the sieve.**
What remains is one genuine tension (the hybrid store, below) plus the *operational apparatus* the
sieve leaves behind — the procedure for running the admission rule on a tempting finding, and the
three axes describing an admitted object's settledness.

## Tension 1 — We have a hybrid store (columnar + tensor)

We store two physically different kinds of data: **scalar/columnar** facts (a neuron's dominant
frequency, a site's circularity) and **dense tensors** (raw weight matrices, DMD modes, SVD
factors, attention patterns). A relational/columnar store is right for the first and wrong for
the second.

### Resolution: unify by *addressing*, not by *format*

The goal is a single consumer surface where the hybrid complexity is hidden behind the Store
boundary. That goal is correct **and is already mostly built** — the move is to finish it, not
to invent it. The principle:

> **An object is a stable identity with scalar columns (in the columnar plane) *plus* references
> to its tensor attributes (in the blob plane). Both planes are reached through the same object
> key.** The unification is one identity/addressing scheme over two stores — not one store.

- **Columnar plane** — conformed Parquet tables, queried directly via DuckDB
  (`miscope.query` / `variant.warehouse`). This is where predicate/filter/join/cross-model
  comparison happens.
- **Tensor plane** — dense arrays in the blob store, addressed by a `TensorRef` row in the
  tensor catalog (already built: `warehouse/tensor_catalog.py`, address-only refs, header-read
  shape, bytes-authoritative dtype). The Store resolves the reference and fetches on demand.

**What this rejects:** trying to make tensors queryable *as columns* (shoving arrays into the
relational store). You never want to `WHERE` over the contents of an SVD factor. Tensors are for
numerical computation; they are fetched by reference, not filtered. So "tensor as an object
datatype" is exactly right — realized as *a typed reference the Store resolves*, which is what
`TensorRef` is. The consumer says `object.attribute`; the Store decides whether that resolves to
a column read or a blob fetch.

### Resolution: the write path is decoupled from the read store (and that is correct)

The original instinct was to have Analyzers write *directly* to the final columnar format, so
"downstream Analyzers query a Store made of hybrid data" with no intermediate. The honest
engineering verdict (already settled in practice — `feedback_analyzer_output_completeness`,
REQ_144's deletion of the 814-line summary engine) is:

> **Analyzer → artifact deposit (npz/safetensors, recompute-expensive, tensor-bearing) →
> conformed columnar projection (cheap, rebuildable). The unified hybrid store is the *read*
> model; the artifact is the *write* deposit. These are not in conflict — they are two ends of
> the same pipe.**

This split *is* the Bronze→Silver hop of the medallion stack (Part 0): the artifact deposit is
Bronze, the conformed columnar projection is Silver, and the View Catalog (Gold) reads Silver.

Why decoupled wins, concretely:

1. **The conformed table is cross-analyzer.** Multiple analyzers feed one semantic table
   (row-union via *claims* — the visitor pattern). No single analyzer owns the final table, so no
   single analyzer can write it directly without losing collision resolution.
2. **Schema evolution is cheap.** A new column, a rename, or a semantic recompute re-runs the
   *materialize* pass over existing artifacts — not the *analysis* (which re-loads checkpoints and
   recomputes). The artifact is the durable truth; the columnar store is a disposable projection.
3. **One deposit carries both planes.** The npz artifact holds scalars *and* tensors together —
   the natural unit the analyzer produces. The split into (columnar table, tensor ref) happens at
   materialize time, behind the Store boundary, not in the analyzer.

**So the answer to "is Analyzer-writes-files → Warehouse-loads the right workflow?": yes — but
reframe it.** It is not a compromise of the unified-store dream. The unified store is achieved at
read time (one query surface, tensors resolved on demand). The decoupled write path is what makes
the read store cheap to rebuild as the schema (and the research) churns. JSON authored files are
being eliminated (REQ_144); the direction already converges on the end-state, via a two-layer
write/read split rather than direct-to-columnar.

**One guard:** Layer 8 (`Decomposition_Scalar`) is structurally a second generic store — same
shape as the analyzer-named fallback. A scalar stable enough to *name* belongs as a column on an
object table (via a claim), never as a `Decomposition_Scalar` row. Otherwise the generic store is
rebuilt under a new name and the dial never tilts.

## Applying the Sieve — the object-naming procedure & the three axes

> **Subsumed by Part 0.** This was once *"Tension 2 — the domain objects are part of the research."*
> The admission rule retired it: *given-real → object, earned-real → query* already decides whether a
> tempting finding is an object or a view, so there is no standing tension between "hard-code the
> unstable objects" and "refuse to name anything" — the sieve picks for us. What survives is purely
> operational, and all of it flows *from* Part 0: a procedure for **running** the admission rule on a
> finding, and three axes for **describing** how settled an admitted object is.

Objecthood stays a **decision we re-run**, not a guess we freeze; every object is marked with how
settled it is so the reader knows what's safe to build against.

#### The object-naming decision procedure

Run this whenever a finding tempts a new object. (This is the line of questioning that keeps
getting buried — it lives here now.)

1. **Stable-noun test.** When we examine this behavior, is there a noun for *what we're
   examining* that survives across findings and across models? If the noun changes every time the
   *method* changes, it isn't the object — name the slower-changing thing.
   *Objects are named after the mechanism; analyzers after the computation.*
2. **Identity test.** Does it have an intrinsic key (a coord tuple) that means the same thing
   across checkpoints and across variants? That key is what makes cross-model comparison possible.
3. **Measurement test (the objecthood litmus).** Does an analyzer write a *measured* attribute to
   it that is **not** reconstructible by a predicate/join over other objects? 
   - Yes → it's an **object** (entity; lives in the schema).
   - No — it's fully a join/predicate over other objects' attributes → it's a **query** (view;
     lives in query-time, not the schema). *Promote a query to an object only on evidence it
     recurs across architectures/families — never on one finding.*
4. **Temporal-type test.** Classify it: **[S]** Snapshot (keyed with `epoch`), **[T]** Trajectory
   (keyed by `variant`, no `epoch`), **[E]** Event (a *transition* in training time, keyed
   `(variant, object_ref)`).
5. **Attribute-home test.** For each thing we learn about it, decide its home: a **scalar column**
   on the object, a **child table** (open-ended sets like top-k), a **link/edge** to another
   object, or a **tensor reference**.
   - **Promotion criterion (which measurements become scalar columns).** A measurement graduates
     from a coordinate-keyed fact to an *intrinsic attribute on the object* only when it is a
     **gauge-invariant function of the operand alone**: (a) *self-derivable* — computable from the
     operand by itself, no probe / input distribution; (b) *invariant* — stable under the operand's
     arbitrary gauges (basis rotation, sign, ordering); (c) *non-reconstructible by join* (step 3).
     Spectral invariants — `effective_rank`, `operator_norm`, `spectral_radius`, copying score —
     pass and belong **on** the object. A decomposition therefore splits: the dense spectrum →
     **tensor reference**; its invariant summaries → **columns**.
   - **Two grades of invariant.** *Unconditional* (rank, norm — no instrument named) vs
     *instrument-conditional* (e.g. dominant frequency under the TaskType's irrep basis — invariant
     only given a fixed instrument). Both may be columns; the second is **task-gated and carries its
     instrument in provenance** — it may not pose as unconditional.
   - **What stays in measurement space.** *Probe-relative* readings (depend on an input
     distribution) belong to `ActivationSite`, not to a weight-derived operand — the
     WeightMatrix/ActivationSite split, re-derived from invariance. *Instrument-arbitrary* readings
     (change under an equally-valid instrument swap) are measurements, never attributes.
6. **Cross-model-comparison test (the payoff).** Can the attribute be selected across variants by
   **Task Parameters + Model Parameters** (e.g. *"primary frequency, model_seed, data_seed for all
   variants on the Task where prime=109"*)? If the keying doesn't permit that join, the object is keyed
   wrong — fix the key before building.

##### Worked example — the Neuron (the user's example, run through the procedure)

- *Stable noun?* **Neuron** — survives every method change. → object. ✔
- *Identity?* `(variant, epoch, layer_index, neuron_index)`. ✔
- *Measured attribute?* "a frequency explains its computational behavior" → `dominant_freq`,
  `max_frac` (scalar columns). ✔ object.
- *"Explained by a set of frequencies"* → **child table** (top-k frequencies per neuron), *not* a
  scalar. Today `neuron_frequency_attribution` carries only the `dominant` row; the schema move
  is to relax it to carry **rank** (top-k), keyed `(variant, epoch, neuron, rank)`.
- *"Switching frequencies"* → a **query** over the primary/top-k across epochs *and* an **[E]
  Event** when a switch is located in time (`switch_counts` is the trajectory tally;
  `commitment_epochs` is the event). Not a new object.
- *Cross-model?* `SELECT dominant_freq, model_seed, data_seed FROM neuron … WHERE prime=109` —
  works iff Neuron is keyed by variant and the freq attribution joins on `(variant, epoch,
  neuron)`. ✔ This is the whole point.

#### Three orthogonal axes (do not conflate them)

An object is described by three independent questions. The mistake is collapsing them into one
"how done is it?" column — they answer different things and move independently.

**1. Maturity — does the *definition* belong to us?** (about the *field*)
- **STABLE** — the definition is architecture-intrinsic or community-accepted. It **does not
  require proof from us**; we can study it freely without risk of giving heft to an object that
  doesn't exist or that we've defined poorly. An `AttentionHead` is STABLE even though we've
  barely studied it — the definition isn't ours to defend.
- **EVOLVING** — the object's *definition* is an open question in the field. Modeling it risks
  reifying a frame that may be wrong. We may still study it, but with rigor about whether it's
  real. A `FrequencyGroup` is EVOLVING — it may be a false frame; that it's richly populated
  doesn't make it real.

**2. Coverage — how much have we populated?** (about our *data*)
`rich → solid → thin → near-empty → empty`. **Adding attributes (scalar, list, or tensor) is a
coverage question, expected on any canonical object** — it is *not* a maturity change. Extending
a STABLE object with more attributes leaves it STABLE.

**3. Status — is it built in the Store?** (about our *codebase* — the internal "under development"
flag)
- **ACTIVE** — object table exists and is populated.
- **IN-DEV** — under development right now.
- **PLANNED** — designed in this doc, not yet built.
- **CANDIDATE** — held open, *not committed* to model; must clear the litmus / earn it on evidence
  before it becomes PLANNED.

> **Why these don't move together:** `AttentionHead` = STABLE · near-empty · PLANNED.
> `FrequencyGroup` = EVOLVING · solid · ACTIVE. `Event` = STABLE-structure · thin · PLANNED.
> "Build a REQ now?" is a *judgment over all three* (Part V), not a single tier.

#### When *not* to mint an object — capture the characteristic as an attribute

The litmus (step 3) keeps queries out of the schema. This is the EVOLVING-side companion: when a
phenomenon is **contested in the field**, prefer recording it as an *attribute* and letting it
stand, rather than minting an object that lends it false solidity.

> **Worked counter-example — clock vs. pizza.** There is an open argument about whether the
> "clock" and "pizza" algorithms are the same mechanism. We would **not** model `Clock` / `Pizza`
> as object types. The working hypothesis is that they likely point at the *same* underlying
> mechanism, and that a single final-snapshot view may just be catching a model as a pizza *or* a
> clock **at that moment** — i.e. it's more likely a **state** than a kind. If common states
> recur across a family (as our `blob → disc → saddle` path does in the frequency groups), then
> `State` becomes a candidate object — entering as **EVOLVING · CANDIDATE**, graduating toward
> STABLE only on community acceptance. Until then, the characteristic lives as an attribute.

Part II carries all three axes as conventions; the scan table in Part V is the one-page
"what's safe to build" view across them.

---

# Part II — Conventions

**Temporal type** — `[S]` Snapshot `(variant, epoch, …)` · `[T]` Trajectory `(variant, …)`,
no epoch · `[E]` Event, a transition keyed `(variant, object_ref, …)`, **never `(variant)`
alone**. Observability lens (Part 0): `[S]`/`[T]` are **Metrics** (time series); `[E]` are
**Events** (discrete state-changes derived *from* those metrics).

**Three orthogonal object axes** (Part I.2):
- **Maturity** (definitional) — `STABLE` (definition not ours / accepted) · `EVOLVING` (definition
  is an open field question; risks reifying a false frame).
- **Coverage** (data populated) — `rich · solid · thin · near-empty · empty`. Adding attributes is
  a coverage change, never a maturity change.
- **Status** (internal build) — `ACTIVE` (built, populated) · `IN-DEV` (building now) · `PLANNED`
  (designed, not built) · `CANDIDATE` (held open, not committed).

**Attribute status** (does the *data* exist today) — `EXISTS` · `NEW` · `CHEAP` (instrument
exists, repoint it) · `DELTA` (data already says more than we read).

**Gating** — `task` (semantic; provided by the **TaskType/Task** — e.g. group structure → irrep basis) ·
`arch` (structural; from the **Model Architecture** — needs ≥2 layers / longer context). Ungated
objects apply to any transformer. **The two gates are the two factors of the Family junction**
(Architecture × TaskType, Layer 1) — the gating taxonomy already encoded that decomposition. Orthogonal
to all three axes above (an `InductionHead` is STABLE but arch-gated and uninstantiated in a 1-layer
model).

**Attribute home** — every attribute lands in exactly one place: a **scalar column** on its
object's table (common case) · a **child table** row (open-ended sets, top-k) · a **link/edge**
to another object · a **`Decomposition_Scalar`** row (open-ended spectral tail only) · a
**tensor reference** into the blob store (dense arrays).

**Generic-association tables (the controlled exception).** `Event` (Layer 7) and `Decomposition`
(Layer 8) are the *only* tables keyed by a polymorphic `(object_type, object_key)` where `object_key`
is a **serialized** composite key, not a foreign key. They are the deliberate exception to the
named-object-table rule — justified because both range over heterogeneous object types and need one
alignment-/join-keyed surface. **Cost made explicit:** the database cannot enforce that a referenced
object exists; referential integrity becomes the *producer's* responsibility, not the schema's.
Discipline — (a) every extraction/decomposition claim that writes one of these tables validates its
`(object_type, object_key)` against a live object key before emitting the row; (b) a periodic
warehouse-health scan flags orphans. No third generic-association table appears without the same
discipline (Part VI #7).

**Store realization** (Part IV maps these): a logical object becomes a *semantic table*, a
*discriminator-keyed slice of a shared table*, or a *tensor-catalog projection* — the logical
split into two objects does **not** mandate two physical tables.

**Recipe addressing** — a parameterized method's identity is its name *and* its parameters (e.g. a
threshold), captured as a **recipe signature** and carried in keys as `recipe_id` (`Decomposition`,
Layer 8) and `method_recipe` (`Event`, Layer 7). Both dereference against **one Recipe Registry** of
method/recipe definitions — the REQ_138 run-set/recipe registry, *generalized* from analysis recipes
to detection-method recipes. A signature is never a bare opaque string: paired with its method/type name it resolves to the full,
introspectable definition, so a thresholded verdict (a `specialize` Event, a parameterized
`Decomposition`) always traces back to exactly how it was produced. **One registry, not one per
consumer** (Part VI build note). The registry holds *definitions*, not measurements of a model, so it
is an **Infrastructure object** (`MethodRecipe`, Layer 1) — given-real by provenance (we declared the
method), not sieve-earned. **Committed as a PLANNED object** with its build deferred until event
capture ramps (Part VI #6) — named now so the `Event`/`Decomposition` recipe FKs aren't dangling
strings.

---

# Part III — The Schema

## Layer 1 — Platform Objects · `STABLE · ACTIVE`

Defined by the platform, not the model. Uncontroversial. (Per-object `STABLE` tags omitted —
the whole layer is platform metadata.)

> **Family = Architecture × TaskType, on a type/instance lattice.** A Family is a *join* on two
> independent factors — a **Model Architecture** and a **TaskType** (the task-logic container, e.g.
> *Modulo Addition*) — each running its own **type → instance** ladder. The Family pairs the two *type*
> nodes; a Variant binds the two *instance* nodes:
>
> | | **type** (template) | →(instantiate)→ | **instance** (realized) |
> | --- | --- | --- | --- |
> | **model side** | `Architecture` (depth/width/ctx) | train (seeds) | `Variant` (trained net) |
> | **task side** | `TaskType` (basis + dataset logic) | resolve (`prime`) | `Task` (ModAdd mod 109) |
> | **pairing** | `Family` = Architecture × TaskType | | a Variant bound to its Task |
>
> Per-factor symmetry: **Architecture : Variant :: TaskType : Task** (instance-of-a-type on each side);
> the Family is the type-pairing, the (Variant, Task) binding the instance-pairing. The factors are
> independent — different architectures train on one TaskType (**already true here**), one architecture
> may (in future) train on a *set* of TaskTypes — so the Family owns no semantics of its own. This
> resolves the old "family object-type ownership" tension (Part VI #2): semantic context — group
> structure, the irrep basis, task parameters like `prime`, the master dataset — belongs to the
> **TaskType / Task**, never to the Family.

### Architecture · `STABLE` — the model-spec container (the model-side *type*)
The structural transformer spec — the **task-independent** config shared by every Variant in a Family.
Given-real by **provenance/definition** → Gate 1, STABLE. The provider of the `arch` gate (depth,
context, width). *Symmetric to `TaskType`:* Architecture is to a Variant what a TaskType is to a Task.
| Field | Type | Description |
| --- | --- | --- |
| architecture_id | int | Surrogate key (PK) |
| name | string | e.g. `1L_4H_d128` |
| n_layers | int | Depth (gates `arch` objects) |
| n_ctx | int | Context length (gates `arch` objects) |
| d_model | int | Residual width |
| n_heads | int | Attention heads per layer |
| d_mlp | int | MLP hidden width |

*(`d_vocab` is **task-coupled**, not an Architecture field: it equals the Task's token count
(= `prime` + special tokens), fixed where the Architecture meets the Task — the one boundary the two
otherwise-independent factors share.)*

### TaskType · `STABLE` — the task-logic container (the task-side *type*)
The reusable logic for a kind of task — e.g. **Modulo Addition**. Given-real by
**provenance/definition** → Gate 1, STABLE. It is the **semantic-context provider** of invariant #2:
the single body of logic that (a) constructs the **irrep / Fourier basis** and (b) generates the
**master dataset** (train / test / probes), both parameterized by `TaskType_Parameter`. Any Family
pairing an architecture with `modular_addition` shares this logic — the *only* difference between p109
and p113 is one parameter, `prime`. Its `structure` licenses the Task-scoped objects of Layer 5.
| Field | Type | Description |
| --- | --- | --- |
| tasktype_id | int | Surrogate key (PK) |
| name | string | e.g. `modular_addition` |
| structure | string | Algebraic structure provided, e.g. `cyclic_group` (`none` if unstructured) |
| description | string | Objective / construction logic |

### TaskType_Parameter · `STABLE` (was `Family_Parameter`)
The TaskType's declared parameters — e.g. `prime`. The old name `Family_Parameter` *already* read
"per-family **task** parameters" — the leak named in plain sight (and `prime` was never a
model_seed/data_seed-style universal).
| Field | Type | Description |
| --- | --- | --- |
| tasktype_id | int | FK → TaskType |
| name | string | Parameter name (e.g. `prime`) |
| datatype | string | Parameter datatype |

**Key:** PK `(tasktype_id, name)`

### Task · `STABLE` — a parameterized *instance* of a TaskType
A TaskType with its parameters **resolved** — e.g. *Modulo Addition mod 109*. **Task : TaskType ::
Variant : Family** (an instance of a type). Home of the *concrete* artifacts the TaskType logic
produces at this parameterization: the specific irrep basis (Z/109 ≠ Z/113) and the specific master
dataset. Shared by every Variant trained on it, whatever its model/data seed.
| Field | Type | Description |
| --- | --- | --- |
| task_id | int | Surrogate key (PK) |
| tasktype_id | int | FK → TaskType |

**Key:** PK `task_id`; unique `(tasktype_id, resolved-params)`.

### Task_Parameter_Value · `STABLE` (was `Variant_Parameter`)
The Task's **resolved** parameter values — e.g. `prime=109`. **The cross-model comparison spine:**
`WHERE prime=109` joins here → `Task` → `Variant` (comparing models on the *same Task*). Moved off the
Variant — prime identifies the *Task*, not the training run; the seeds are what vary per Variant.
| Field | Type | Description |
| --- | --- | --- |
| task_id | int | FK → Task |
| parameter_name | string | Matches `TaskType_Parameter.name` |
| value | string | Serialized; cast on read via datatype |

**Key:** PK `(task_id, parameter_name)`

### Family · `STABLE` — the (Architecture × TaskType) template (the *type* pairing)
Carries no semantics of its own; it **pairs** a model `Architecture` with a `TaskType`. Type-level: a
Family is to a Variant what a TaskType is to a Task.
| Field | Type | Description |
| --- | --- | --- |
| family_id | int | Surrogate key (PK) |
| name | string | Full family name |
| abbr | string | Abbreviated name |
| architecture_id | int | FK → Architecture (the structural factor) |
| tasktype_id | int | FK → TaskType (the semantic factor) |

### Variant · `STABLE` — a trained *instance* of a Family, bound to a Task
A realized training run. Identity is **irreducible** — different seeds are different models, not noisy
samples of one. Binds the model-side instance (this trained net) to the task-side instance (its
`Task`); its structural config comes from `Family → Architecture`.
| Field | Type | Description |
| --- | --- | --- |
| variant_id | int | Surrogate key (PK) |
| family_id | int | FK → Family (the template; carries the Architecture) |
| task_id | int | FK → Task (the resolved task instance it trained on) |
| name | string | e.g. `p113_seed999_dseed598` |
| abbr | string | Abbreviated name |
| model_seed | int | Weight-init seed — a **Model Parameter** (per-run) |
| data_seed | int | Train/test split seed — a **Model Parameter** (per-run) |

> **Integrity:** `Variant.family.tasktype_id == Variant.task.tasktype_id` — the template's TaskType
> and the bound Task's TaskType must agree.

> **Two parameter spaces (mirror images across the model/task split).** **Task Parameters** (`prime`,
> …) are declared by `TaskType_Parameter` and resolved on the `Task` — they identify *which task*.
> **Model Parameters** (`model_seed`, `data_seed`) live on the `Variant` — they identify *which
> training run* of a fixed (Family, Task). Seeds are architecture-universal (every run has them) and
> demonstrably consequential to outcomes, so they are standing Variant columns rather than a declared
> set; other training hyperparameters (learning rate, weight decay) join this space *if/when* they
> vary across Variants.

### Checkpoint · `STABLE`
A point in training time. Anchors every epoch-scoped object.
| Field | Type | Description |
| --- | --- | --- |
| variant_id | int | FK → Variant |
| epoch | int | Training step |
| safetensors_path | string | Weights on disk (relative) |
| wall_clock_time | float | Seconds elapsed (optional) |

**Key:** PK `(variant_id, epoch)`. *(Loss is its own object — Layer 3.)*

### Probe · `STABLE`
Input data for a forward pass. Activation results are meaningful only relative to the probe.
**Derived from the `Task`'s master dataset** (TaskType logic generates it; the Task realizes it), so
it hangs off the Task, not the Family.
| Field | Type | Description |
| --- | --- | --- |
| probe_id | int | Surrogate key (PK) |
| task_id | int | FK → Task |
| name | string | Descriptive name |
| description | string | Purpose / construction |
| n_samples | int | Sample count |
| probe_path | string | Serialized tensor on disk |

### MethodRecipe · `STABLE · empty · PLANNED` — the method/recipe registry (committed; build deferred)
A registered, parameterized **method definition** — a detector or decomposition recipe *with its
parameters resolved* (notably a `specialize` detector's threshold). Given-real by **provenance** (we
declared the method), so it is an Infrastructure object, not a sieve-earned one — it holds
*definitions*, not measurements of a model. `Event.(detection_method, method_recipe)` and
`Decomposition.(decomposition_type, recipe_id)` FK into it; it is the queryable surface that lets us
*compare detectors* and *list every threshold a `specialize` event has used*. **Committed now, built
later:** the model names it so those FKs aren't dangling strings, but the table itself ships when event
capture ramps (it gets unwieldy fast otherwise) — Part VI #6, folds into REQ_138.
| Field | Type | Description |
| --- | --- | --- |
| detection_method | string | Named method/detector (`loss_velocity`, `activation_specialization`, …) |
| method_recipe | string | Recipe signature of resolved params **incl. threshold** (empty = default) |
| definition | json | Full parameterization (params, thresholds, producing-logic ref) |
| description | string | Human-readable method description |

**Key:** PK `(detection_method, method_recipe)`.

---

## Layer 2 — Intrinsic Objects (architecture)

> **The `site` coord splits into two objects: `WeightMatrix` (intrinsic to params) vs.
> `ActivationSite` (probe-relative).** Earned on four independent seams that all fall on this
> boundary: attribute *kind* (intrinsic vs input-distribution-dependent), the SNR/energy guard
> (applies only to activation sites), the DMD variant (DMDc for the additive residual stream vs
> joint DMD for orthogonal weight subspaces), and the collision resolution below.
>
> **Store note (discriminator vs table):** `GROUP_TYPE` *already* encodes this split
> (`weight_matrix` / `activation_site` / `centroid_group`). The split is therefore a realization
> *choice*: keep it a discriminator column, or promote to two physical tables. **Lean:
> discriminator now; separate tables only when the attribute *sets* diverge** (weight-only SVD
> spectra vs activation-only CKA). See Part IV and the open defect in Part VI.

### WeightMatrix · `[S]` · `STABLE · rich · ACTIVE`
A parameter tensor at a checkpoint. `site` ∈ `{W_E, W_Q, W_K, W_V, W_O, W_in, W_out, W_U,
W_pos}`. (Embedding=`W_E`, Unembedding=`W_U`, MLP=`W_in`/`W_out` are site names, not tables.)
Subsumes the earlier `WeightGroup` proposal.
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, site | — | Key | — |
| power | float | weight_basis_projection | EXISTS |
| fractional_power | float | weight_basis_projection | EXISTS |
| dominant_frequency_pair | string | weight_basis_projection | EXISTS |
| rel_velocity | float | stream_weight_velocity | EXISTS |

**Key:** PK `(variant_id, epoch, site)`; FK → Checkpoint · **Tensors:** raw matrix, SVD `u`/`vt`,
Fourier coeff cubes (`magnitudes`, `phases`, `cos/sin_coeffs`, …) via tensor ref.

### ActivationSite · `[S]` · `STABLE · rich · ACTIVE`
Activation snapshot at a runtime location (`resid_post`, `mlp_out`, `attn_out`). Probe-relative.
**`ActivationSite[resid_post]` *is* the residual stream** (Elhage framing) — a naming win, not new
compute.
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, probe_id, site | — | Key | — |
| circularity | float | repr_geometry | EXISTS |
| center_spread | float | repr_geometry | EXISTS |
| mean_radius | float | repr_geometry | EXISTS |
| mean_dim | float | repr_geometry | EXISTS |
| pca_var_pc1..3 | float | repr_geometry | EXISTS |
| fisher_mean / fisher_min | float | repr_geometry | EXISTS |
| snr | float | repr_geometry | EXISTS |
| freq_norm | float | activation_frequency_norm | EXISTS |
| fourier_alignment | float | centroid_fourier_alignment | EXISTS |

**Key:** PK `(variant_id, epoch, probe_id, site)`; FK → Checkpoint, Probe.

### AttentionHead · `[S]` · `STABLE · near-empty · PLANNED`
**STABLE by definition** (architecture-intrinsic; the definition is the field's, not ours) but the
*least-covered* canonical object we have: `HEAD` appears in one coord-signature with 2 attributes,
and the attention `patterns` tensor still buries the head axis. Low coverage is not low maturity —
this is exactly the object that needs coverage REQs, with no risk of reifying a bad frame.
REQ_136 made `Coord.HEAD` first-class for weight-spectra / wbp; promoting the head axis *out of the
patterns blob* is a **coordinate change at the tensor boundary**, not a columnar claim.
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, layer_index, head_index | — | Key (layer DEFAULT 0) | — |
| dominant_frequency | int | weight_basis_projection | EXISTS |
| sv | float | weight_spectra | EXISTS |
| attention_entropy | float | REQ_151 / DecisionQuery | CHEAP |

**Key:** PK `(variant_id, epoch, layer_index, head_index)`; FK → Checkpoint · **Tensors:**
attention `patterns` (promote head axis to a coordinate).

### MLP_Neuron · `[S]` · `STABLE · solid · ACTIVE`
A single neuron. Structural identity; behavior captured by attributes + a top-k child table.
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, layer_index, neuron_index | — | Key (layer DEFAULT 0) | — |
| dominant_freq | int | neuron_frequency_attribution | EXISTS |
| max_frac | float | neuron_frequency_attribution | EXISTS |
| assignments | int | neuron_grouping (→ FrequencyGroup) | EXISTS |
| confidence | float | neuron_grouping | EXISTS |
| radial / tangential | float | rotational_dynamics (escape detector) | NEW |
| power | float | neuron_activation_spectrum | NEW |

**Key:** PK `(variant_id, epoch, layer_index, neuron_index)`; FK → Checkpoint.

#### MLP_Neuron_Frequency · `[S]` child table (top-k) · `STABLE · thin · ACTIVE` (rank: `PLANNED`) · `task`
"A neuron is explained by a *set* of frequencies." The open-ended membership the scalar
`dominant_freq` can't hold. Today's `neuron_frequency_attribution` semantic table carries only the
`dominant` row — **relax it to carry rank** (this is the schema move the worked example produces).
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, neuron, rank | — | Key | — |
| frequency | int | neuron_frequency_attribution | EXISTS (dominant only today) |
| frac_explained | float | neuron_frequency_attribution | EXISTS |

**Key:** PK `(variant_id, epoch, neuron, rank)` · `rank=0` is the dominant frequency (the current
`dominant=True` row). Edge → `FrequencyMode.frequency`.

---

## Layer 3 — Performance Metrics · `STABLE · ACTIVE`

### Loss · `[S]`
| Field | Type | Description |
| --- | --- | --- |
| variant_id | int | FK → Variant |
| loss_type | int | TRAIN=0, TEST=1 |
| epoch | int | — |
| value | float | Loss value |

**Key:** PK `(variant_id, loss_type, epoch)`

---

## Layer 4 — Derived / Virtual Objects · `STABLE (definition) · empty · PLANNED`

Composed from intrinsic weights. **Not arch-gated** — a 1-layer model has within-head QK/OV
circuits. Empty today only because the composed weight was never materialized. The high-value
move: materialize composed `W_Q^T W_K` / `W_O W_V` and point existing instruments at them (mostly
`CHEAP`). The **full** circuits (`W_U W_O W_V W_E`, `W_E^T W_Q^T W_K W_E`) and the **direct path**
(`W_U W_E`) extend the residual-space pair end-to-end through the embeddings — added from the
literature harvest ([canonical_object_harvest.md](canonical_object_harvest.md), Source 1). Their
attributes are **gauge-invariant scalars of the operand** (Part I step 5), not probe-relative
measurements.

> **Maturity nuance — "the QK circuit" is STABLE; "a circuit" is EVOLVING.** The *general*
> mech-interp notions — `feature`, `circuit` (as a computational subgraph), `learned algorithm`,
> `concept` — are genuinely EVOLVING concepts in the field; we do **not** mint objects for them.
> But the **QK circuit** and **OV circuit** specifically are precise, community-accepted composed
> weights (`W_Q^T W_K`, `W_O W_V` — the Elhage framework). Their *definition* is borrowed and
> mathematically exact, so the *objects* are STABLE. What's interpretive is what their attributes
> *mean* — which is the research, carried as attributes, not as object identity.

### QKCircuit · `[S]` · `STABLE · empty · PLANNED` — composed `W_Q^T W_K` (*what to look at*)
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, layer_index, head_index | — | Key (layer DEFAULT 0) | — |
| dominant_frequency | int | weight_basis_projection (repointed) | CHEAP |
| power | float | weight_basis_projection | CHEAP |
| effective_rank | float | weight_spectra | CHEAP |
| operand_symmetry | float | NEW (task: a↔b commutativity) | NEW |

**Key:** PK `(variant_id, epoch, layer_index, head_index)`; FK → AttentionHead.
*HeadPair (heads sharing a frequency) is a **query** (self-join over a per-head frequency table),
not an object — fails the litmus; promote only on cross-architecture evidence.*

### OVCircuit · `[S]` · `STABLE · empty · PLANNED` — composed `W_O W_V` (*what to do with it*)
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, layer_index, head_index | — | Key (layer DEFAULT 0) | — |
| dominant_frequency | int | weight_basis_projection | CHEAP |
| effective_rank | float | weight_spectra | CHEAP |
| logit_attribution | float | output_logit_health (partial) | NEW |

**Key:** PK `(variant_id, epoch, layer_index, head_index)`; FK → AttentionHead.

### FullOVCircuit · `[S]` · `STABLE · empty · PLANNED` — composed `W_U W_O W_V W_E` (*source token → output-logit map*)
The square `[vocab, vocab]` end-to-end OV path. For modular arithmetic it exposes the additive
structure directly. **Highest-value harvest target — build first.**
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, layer_index, head_index | — | Key (layer DEFAULT 0) | — |
| copying_score | float | ov_eigenspectrum (Σ λ₊ / Σ\|λ\|) | NEW |
| effective_rank | float | weight_spectra (repointed) | CHEAP |
| operator_norm | float | weight_spectra | CHEAP |
| dominant_frequency | int | weight_basis_projection (task-conditional) | CHEAP |

**Key:** PK `(variant_id, epoch, layer_index, head_index)`; FK → OVCircuit, AttentionHead ·
**Tensors:** the `W_U W_O W_V W_E` matrix + eigenvalues via tensor ref.

### FullQKCircuit · `[S]` · `STABLE · empty · PLANNED` — composed `W_E^T W_Q^T W_K W_E` (*which token-pairs the head binds*)
The square `[vocab, vocab]` end-to-end QK bilinear form. Positional variants substitute `W_pos`.
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, layer_index, head_index | — | Key (layer DEFAULT 0) | — |
| effective_rank | float | weight_spectra (repointed) | CHEAP |
| operator_norm | float | weight_spectra | CHEAP |
| dominant_frequency | int | weight_basis_projection (task-conditional) | CHEAP |

**Key:** PK `(variant_id, epoch, layer_index, head_index)`; FK → QKCircuit, AttentionHead ·
**Tensors:** the `W_E^T W_Q^T W_K W_E` matrix + eigenvalues via tensor ref.

### DirectPath · `[S]` · `STABLE · empty · PLANNED` — composed `W_U W_E` (*0-layer bigram logit term*)
The embedding→unembedding path: the baseline against which head contributions are read in the logit
decomposition. One per checkpoint (no head axis).
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch | — | Key | — |
| effective_rank | float | weight_spectra (repointed) | CHEAP |
| operator_norm | float | weight_spectra | CHEAP |

**Key:** PK `(variant_id, epoch)`; FK → Checkpoint · **Tensors:** the `W_U W_E` matrix via tensor ref.

---

## Layer 5 — Task-Scoped Objects (group-structured tasks) · `task`-gated

> **Resolved (was the Part VI "family object-type ownership" open decision).** These were called
> "family-scoped," which strained invariant #2 ("families are context providers, not view owners").
> The **Family = (Architecture × TaskType)** decomposition (Layer 1) fixes it: the semantic context
> here is provided by the **TaskType** (its group structure), realized on the **Task**, not the Family.
> Nothing is "family-owned" — these are **Task-scoped**, and the claims registry gains a **TaskType**
> dimension, not a family one. This *honors* invariant #2 (TaskTypes/Tasks are the real context
> providers) rather than bending it.

### IrrepBasis · `STABLE` · TaskType instrument, realized per Task (not a measured object)
The canonical basis for group-structured tasks — for Z/p, the DFT basis
`{1, cos(2πk·/p), sin(2πk·/p)}`, which **is** the irreducible representations of the cyclic group.
A **theorem-given** primitive whose *construction* is provided by the **TaskType's** logic (from its
group structure) — not the Family ([canonical_object_harvest.md](canonical_object_harvest.md),
Source 3) — and whose *concrete* basis is realized on the **Task** instance: Z/113 and Z/109 are
different groups with different bases. The basis is STABLE; **which** modes the model populates is the
EVOLVING finding carried by `FrequencyMode` below. Realization: a TaskType-supplied instrument (the
Fourier transform already in use), its concrete basis attached to the Task, named here so the STABLE
basis and the EVOLVING selection are never conflated. The frequency `k` that keys `FrequencyMode` is
an index *into this basis*; any "dominant frequency" column elsewhere is an *instrument-conditional*
invariant relative to it (Part I step 5).

> **Invariant #1 stays intact.** The *transform* — the Fourier/irrep projection — is a **universal
> instrument**: it runs on any matrix and does not change shape for the task. What the Task provides is
> only the **concrete basis instance** (the Z/p irreps for this `prime`), as *context* per invariant
> #2. "IrrepBasis is a TaskType instrument" is shorthand for "the universal Fourier transform,
> parameterized by the Task's group" — the Task supplies the *parameter*, not a bespoke view.
> Invariant #1 (instruments are universal) and invariant #2 (Tasks provide context) are both honored
> here, not traded against each other.

### FrequencyMode · `[S]` · **side-gate object** · `EVOLVING · thin · ACTIVE`
The central object of group-structured tasks. Admitted via the **side gate** on its *own* per-checkpoint
`committed`/nucleation reading under the irrep basis — an irreducible measurement of the mode, not a
join — so it is an object, not a query. But it is **richly transverse in its edges**: most of its
*descriptive* payload lives on other objects' rows, and `frequency` is the **join target** tying
neurons, groups, and sites together. If that own-measurement doesn't survive scrutiny it collapses
to a pure query — which is exactly what `EVOLVING` flags.
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, frequency | — | Key | — |
| active | bool | fourier_frequency_quality | EXISTS |
| energy | float | gradient_site (× site) | EXISTS |
| committed | bool | nucleation | EXISTS |

**Key:** PK `(variant_id, epoch, frequency)`; FK → Checkpoint · **Join edges:**
`MLP_Neuron_Frequency.frequency`, `FrequencyGroup.group_freq`, `ActivationSite × frequency`
energy. **Missing edge to build:** time-varying neuron→freq membership (the `reference_epoch`
pinning is the p101 artifact — the missing edge *is* the bug).

### FrequencyGroup · `[S]` · **side-gate object** · `EVOLVING · solid · ACTIVE`
A set of neurons sharing a dominant frequency — **earned-real, admitted via the side gate.** Its
object-hood rests on **irreducible** measurements of the neuron *set*: `mean_spread` (centroid PCA)
and `r2_linear/quadratic/curvature` (manifold shape) are joint-geometry readings that **no
aggregate-join over per-neuron `MLP_Neuron_Frequency` rows reconstructs**. That is what admits it —
not its coverage. It stays `EVOLVING` because the open rigor is precisely whether those
joint-geometry measurements survive scrutiny; if they reduce to a per-neuron join after all, it
collapses to a query. High coverage does not make it real.
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, epoch, group | — | Key | — |
| group_freq | int | neuron_grouping (→ FrequencyMode) | EXISTS |
| n_per_group | int | neuron_grouping | EXISTS |
| mean_spread | float | neuron_group_pca | EXISTS |
| r2_linear / r2_quadratic / r2_curvature | float | intragroup_manifold | EXISTS |

**Key:** PK `(variant_id, epoch, group)`; FK → Checkpoint. *FrequencyMode and FrequencyGroup are
two views of nearly the same mechanism; `group_freq` is the edge that relates them.*

---

## Layer 6 — Trajectory Objects · `[T]`

Keyed `(variant, …)`, no `epoch`. Home of the DMD spectra.

### ActivationSiteTrajectory · `[T]` · `STABLE · rich · ACTIVE` — DMD home
| Field | Type | Source | Status |
| --- | --- | --- | --- |
| variant_id, site | — | Key | — |
| explained_variance_ratio | float[] | global_centroid_pca | EXISTS |
| dmd_eigenvalues | tensor | activation_dmd | EXISTS |
| cka_to_final | float | representation_similarity | EXISTS |
| spectral_radius | float | spectral_persistence | EXISTS |

**Key:** PK `(variant_id, site)`; FK → Variant. *(DMDc here — residual stream is additive,
not a free body. See Part I / the DMD note.)*

### FrequencyGroupTrajectory · `[T]` · `EVOLVING · solid · ACTIVE`
`group_freqs`, `group_sizes`, per-group DMD (joint DMD across groups — orthogonal Fourier modes).
**Key:** `(variant_id, group[, site])`

### NeuronTrajectory · `[T]` · `STABLE · thin · ACTIVE`
`switch_counts`, `neuron_group_idx`. **Key:** `(variant_id, neuron_index)`.
*(`commitment_epochs` is really an Event — see Layer 7.)*

### FrequencyModeTrajectory · `[T]` · `EVOLVING · thin · ACTIVE`
`key_frequencies` (gradient_site). **Key:** `(variant_id, frequency)`

---

### Layer 7 — Events · `[E]` · `STABLE (structure) · thin · PLANNED` (new capability)

The **Events** mode of the observability lens (Part 0): a discrete state-change at a *point* in
training time, derived from the Metric time series of Layers 2–6. Three properties fix what an Event
is and where it lives:

- **Events are derived annotations on the timeline, never the primary store.** An Event is *earned*
  (Part 0) — computed from metrics, not given by provenance/architecture/theorem — so it is
  query-form, **materialized** as a Silver derived table (REQ_141-style) for reuse, not a given-real
  object. The `STABLE (structure)` tag describes the *schema*, not the instances.
- **Events are method- and interpretation-dependent.** The same transition is located differently
  under different detectors, so an Event **carries its detection method** (as `Training_Window`
  already does via `boundary_method`). Worked example: *second-descent onset* is the epoch where
  test-loss **velocity** (1st derivative) and **acceleration** (2nd derivative) cross chosen
  thresholds; *late-instability onset* is the analogous crossing on the train/test curves. Different
  thresholds → different epochs → different, equally-legitimate rows.
- **Events are the alignment targets for cross-variant comparison.** Absolute epoch is not comparable
  across variants; *events* are. To compare two runs you register them onto a shared event (align
  both at `second_descent_onset`), then read metrics in event-relative time. This is why Events are
  **not scattered as attributes on objects** (the open question): a `commitment_epoch` buried on each
  object is un-queryable across object types and across variants — exactly the aggregate that hid the
  p109 reorganization. They need one uniform, alignment-keyed table; the `object_ref` in the key links
  each Event *back* to the object it annotates without dissolving into it.

**Keyed `(variant, object_ref, event_type)`, not `(variant)` alone** — the p109 finding made this
structural: a late reorganization is silent in every aggregate diagnostic but visible
per-head/per-frequency. An Event keyed only by Variant **is** the aggregate that hides it. A *global*
event such as second-descent onset uses `object_type='loss'`/`'global'` — the loss curve is its
operand.

> **Store note:** no Event table exists today — events are buried as epoch-valued scalars on other
> objects (`commitment_epochs`, `graduation_epochs`, `regimes__boundary_indices`, velocity-peak
> epochs). Lifting them out is a *new derived table + extraction claims*, not a reshape of existing
> columns. Highest-value novel piece; own requirement. The loss-derivative detector above
> (velocity/acceleration + thresholds) is the natural first extraction claim — it anchors late
> instability *and* second descent across variants from the one curve every variant has.

### Event · `[E]`
| Field | Type | Description |
| --- | --- | --- |
| variant_id | int | Key |
| object_type | string | Key — `neuron, head, frequency, group, loss, global, …` |
| object_key | string | Key — serialized object key |
| event_type | string | Key — `commitment, graduation, specialize, de_sharpen, second_descent_onset, instability_onset` |
| detection_method | string | Key — *which* detector (`loss_velocity, activation_specialization, dmd, visual`); the method's stable identity |
| method_recipe | string | Key — signature of *this detector's parameters only* (thresholds, windows; REQ_138; empty = the detector's default recipe) |
| epoch | int | When it occurred |
| notes | string | Optional |

**Key:** PK `(variant_id, object_type, object_key, event_type, detection_method, method_recipe)`;
FK → Variant · Absorbs `commitment_epochs`, `graduation_epochs`, velocity-peak epochs, DMD
`regimes__boundary_indices`. `detection_method` + `method_recipe` in the key make the method- and
**threshold**-dependence structural — the same transition under two detectors, or one detector at two
thresholds, is two comparable rows, not a collision. `method_recipe` reuses the REQ_138
recipe-addressing already carried on `Decomposition.recipe_id` — don't build a parallel surface.

> **`detection_method` and `method_recipe` are independent — both stay in the key.**
> `detection_method` is the detector's identity (the *what*); `method_recipe` signs only its
> *parameters* (the *how-tuned*), scoped under that detector. Neither determines the other:
> `loss_velocity` at two threshold sets is two recipes under one method; `loss_velocity` vs `dmd` is
> two methods. This keeps the common query cheap — *"all `loss_velocity` events"* is `WHERE
> detection_method='loss_velocity'`, no signature resolution — while the recipe still pins the exact
> threshold. The Recipe Registry (`MethodRecipe`, Layer 1) resolves the `(detection_method,
> method_recipe)` *pair* to the full definition; the method name is carried explicitly so a recipe
> signature is never the only handle on which detector ran.

> **Worked example — "specialization" (the threshold problem, resolved).** *"When does a neuron
> specialize?"* has been unanswerable because specialization is a **threshold judgment, not a
> fact**: a neuron crosses some `max_frac`/confidence bar at an epoch that *moves with the bar*. The
> Event table is its home. `event_type='specialize'`, `object_type='neuron'`, and the threshold
> lives in `method_recipe` — so `specialize @ frac≥0.8` and `specialize @ frac≥0.9` are two
> traceable rows, each resolving back to its definition. We carry *several* specialization events at
> once, explicit about their methods, **without ever letting a thresholded verdict harden into a
> given-real attribute on the Neuron.** The method-dependence is recorded, not hidden — and not
> promoted. (This is the Event analogue of the sieve's discipline: a detector verdict is an
> annotation, never an object's own state.)

**Event vs Window — instant vs coordination band.** An Event is a per-object *instant*; a
`Training_Window` is the *interval* a cluster of Events falls within. The distinction is
load-bearing because network changes during a reorganization are **coordinated but not
simultaneous** — different heads/neurons/frequencies follow their own sub-timelines, all moving
*around* a point rather than at it. Pinning them to one epoch would be false; treating them as
scattered would miss the coordination. The Window holds the range that captures it — and **its width
is itself a measurement**: some variants coordinate near-simultaneously (tight window), others take
thousands of epochs to complete the reorganization, if they complete at all (loose / open-ended
window). That extent is an earned measurement of *coordination tightness*, not bookkeeping.

### Training_Window · `[E-adjacent]` · `STABLE · solid · ACTIVE`
| Field | Type | Description |
| --- | --- | --- |
| variant_id | int | FK → Variant |
| window_name | string | `first_descent, plateau, second_descent, terminal_training` |
| boundary_method | string | `visual, loss_derivative, dmd` |
| epoch_start | int | Inclusive |
| epoch_end | int | Inclusive; NULL if terminal |

**Key:** PK `(variant_id, window_name, boundary_method)`. *Quarantined window layer
(`Coord.WINDOW`, one-way DAG) per REQ_144.*

---

## Layer 8 — Open-Ended Decomposition Tail · `STABLE (mechanism) · ACTIVE`, use sparingly

For variable-shape outputs (SVD/DMD/Fourier/PCA/eigen). **Stable attributes belong on object
tables (Layers 2–7); this layer is *only* the open-ended tail** — which is what stops the generic
store from becoming a second blob mirror.

> **Guard (Tension 1):** `Decomposition_Scalar` is the same shape as the generic analyzer-named
> fallback. Anything nameable and stable is a column on an object table via a claim. Reserve this
> layer for genuinely variable-shape spectral scalars and the tensor refs. V1's mistake was
> making `Decomposition` the *central* join table; here it is the *tail*.

### Decomposition
| Field | Type | Description |
| --- | --- | --- |
| decomposition_id | int | Surrogate key (PK) |
| variant_id | int | — |
| epoch | int | NULL for trajectory decompositions |
| object_type | string | Which object produced this |
| object_key | string | Serialized source-object key |
| decomposition_type | string | `svd, dmd, fourier, pca, eigendecomposition` |
| recipe_id | string | Parameterization signature (empty = default) |

**Unique:** `(variant_id, epoch, object_type, object_key, decomposition_type, recipe_id)`

### Decomposition_Scalar
`(decomposition_id, field_name) → value`. Open-ended spectral scalars only.

### Decomposition_Tensor → **realized as the tensor catalog (`TensorRef`)**
`(decomposition_id, field_name) → (tensor_path, shape, dtype)`. This *is*
`warehouse/tensor_catalog.py` — address-only, header-read shape, bytes-authoritative dtype. Do not
build a parallel structure.

---

# Part IV — Store-Tier Realization (model → build bridge)

How each logical object lands in the warehouse. The warehouse already has the two-layer shape this
targets — the move is to **tilt the dial** (grow the semantic layer, shrink the generic fallback),
not rebuild.

| Logical construct | Store mechanism | Status |
| --- | --- | --- |
| Object scalar column | `SemanticClaim` reshaper → conformed semantic table (`mapping_semantic.py`) | mechanism exists; 6 tables use it |
| WeightMatrix / ActivationSite split | `GROUP_TYPE` discriminator (one table) → separate tables only if attr sets diverge | discriminator exists (see defect, Part VI) |
| Child table (top-k freq) | Relax `neuron_frequency_attribution` claim to carry `rank` | small claim change |
| Tensor attribute | `TensorRef` row in tensor catalog, resolved on demand | built (`tensor_catalog.py`) |
| Event | **New** semantic table + extraction claims (scalar-epoch → Event rows) | new build |
| AttentionHead head axis | Promote head out of `patterns` blob to a `Coord.HEAD` coordinate | partial (REQ_136) |
| FrequencyMode neuron→freq edge | Time-varying membership table (kills `reference_epoch` pinning) | new build |
| Un-promoted attribute | Generic analyzer-named fallback (`writer.py`) | the backlog to drain |
| Versioning / staleness | Fold into REQ_145 signatures + one **developer-declared semantic break** | do not build a parallel surface |

**The write/read split, concretely:** Analyzer → npz/safetensors artifact (deposit) → materialize
pass reads `AnalyzerSpec.outputs`, routes each field via its claim to a semantic table or the
fallback, and registers tensor fields as `TensorRef`s → `miscope.query` / `variant.warehouse`
serve the unified read surface. Promoting an attribute = writing a claim; it never re-runs
analysis, only the cheap materialize pass.

**Make the unified read-surface tangible (consumer-side gap).** The hybrid store is most abstract
from the *consumer* seat — you can't feel "one surface, two planes" until you watch a query hand
back a tensor reference and resolve it. The cheap fix: add an example to
[demos/demo_query_surface.ipynb](demos/demo_query_surface.ipynb) that runs a query returning a
`TensorRef` (a columnar row) and then loads the referenced tensor on demand — the columnar→blob
hop, end to end, behind the one Store boundary. That single cell is the clearest statement of the
Tension-1 resolution.

---

# Part V — Build-readiness scan (one page, three axes)

The validation aid. **Maturity** (is the definition safe to build against?) · **Coverage** (how
much data exists?) · **Status** (built yet?). "Build a REQ now?" is the judgment over all three —
note how it tracks Maturity+Status, *not* Coverage (low coverage on a STABLE object is precisely a
build target, not a blocker).

| Object | Layer | Temporal | Maturity | Coverage | Status | Build a REQ now? |
| --- | --- | --- | --- | --- | --- | --- |
| Family (Arch × TaskType template) | 1 | — | STABLE | — | ACTIVE | ✅ yes — repoint FK to `tasktype_id` |
| TaskType / TaskType_Parameter | 1 | — | STABLE | thin | PLANNED | ✅ yes — the task-logic container |
| Task / Task_Parameter_Value | 1 | — | STABLE | thin | PLANNED | ✅ yes — the resolved task instance |
| Variant (+ Model Params) | 1 | — | STABLE | rich | ACTIVE | ✅ yes — add `task_id` FK |
| Architecture | 1 | — | STABLE | solid | PLANNED | ✅ **commit now** — many-to-many already real |
| MethodRecipe (registry) | 1 | — | STABLE | empty | PLANNED | 🔨 commit object; build with Event capture |
| Checkpoint, Probe | 1 | — | STABLE | — | ACTIVE | ✅ yes (Probe → Task) |
| WeightMatrix | 2 | S | STABLE | rich | ACTIVE | ✅ yes (fix discriminator first) |
| ActivationSite | 2 | S | STABLE | rich | ACTIVE | ✅ yes |
| MLP_Neuron (+ Frequency child) | 2 | S | STABLE | solid | ACTIVE | ✅ yes (rank-relax the child) |
| AttentionHead | 2 | S | STABLE | near-empty | PLANNED | ✅ yes — coverage REQ (head-axis coord) |
| Loss | 3 | S | STABLE | — | ACTIVE | ✅ yes |
| QKCircuit / OVCircuit | 4 | S | STABLE | empty | PLANNED | ✅ yes — materialize composed weight |
| FullOVCircuit | 4 | S | STABLE | empty | PLANNED | ✅ **build first** — additive structure + copying score |
| FullQKCircuit | 4 | S | STABLE | empty | PLANNED | ✅ yes — token-pair binding |
| DirectPath | 4 | S | STABLE | empty | PLANNED | ✅ yes — bigram baseline |
| FrequencyGroup | 5 | S | **EVOLVING** | solid | ACTIVE | ⚠️ study, but rigor on whether it's real |
| FrequencyMode | 5 | S | **EVOLVING** | thin | ACTIVE | ⚠️ build the edges, not a leaf |
| IrrepBasis (Task instrument) | 5 | — | STABLE | — | (instrument) | ✅ name it — Task-provided basis under FrequencyMode |
| ActivationSiteTrajectory | 6 | T | STABLE | rich | ACTIVE | ✅ yes |
| FrequencyGroupTrajectory | 6 | T | **EVOLVING** | solid | ACTIVE | ⚠️ inherits FrequencyGroup |
| NeuronTrajectory | 6 | T | STABLE | thin | ACTIVE | ✅ yes |
| FrequencyModeTrajectory | 6 | T | **EVOLVING** | thin | ACTIVE | ⚠️ |
| Event (derived) | 7 | E | STABLE (structure) | thin | PLANNED | 🔨 new derived table — own requirement |
| Training_Window | 7 | E | STABLE | solid | ACTIVE | ✅ yes |
| Decomposition tail | 8 | — | STABLE (mech) | — | ACTIVE | ✅ tensor ref built; scalar tail sparingly |
| `State` (clock/pizza-as-state) | — | T/E? | **EVOLVING** | empty | CANDIDATE | ❌ hold — earn it on recurring states |
| `HeadPair` | (query) | — | — | — | — | ❌ it's a **query**, not an object |
| InductionHead, PrevToken, … | 2 | S | STABLE | empty | PLANNED | ⛔ `arch`-gated — needs ≥2 layers |

*Reading the axes apart: `AttentionHead` and `FrequencyGroup` sit on opposite diagonals — STABLE
but barely-covered vs. EVOLVING but well-covered. The first is a safe build; the second a careful
one. That distinction was invisible in the old single-tier column.*

---

# Part VI — Open Decisions & Defects (resolve before building)

1. **Discriminator mislabel (defect, linchpin of the L2 split).** In
   [mapping_semantic.py:239-245](packages/miscope/src/miscope/warehouse/mapping_semantic.py#L239-L245),
   `freq_group_weight_geometry.circularity` (weight-side → `WeightMatrix`) is stamped
   `GroupType.ACTIVATION_SITE`, while `repr_geometry.circularity` (activation-side) is
   `CENTROID_GROUP`. The enum has an unused `WEIGHT_MATRIX` value. The "collision resolves for
   free via the discriminator" claim depends on this stamp being correct — **fix first.**
2. **TaskType / Task as the semantic factor (resolves the old "family object-type ownership").** The
   **Family = (Architecture × TaskType)** decomposition on a type/instance lattice (Layer 1) dissolves
   the tension: semantic context (group structure, irrep basis, task parameters, master dataset)
   belongs to the **TaskType / Task**, so the claims registry (`CLAIMS`, flat global dict) gains a
   **TaskType** dimension, not a family one — *honoring* invariant #2 (TaskTypes/Tasks are the context
   providers) rather than bending it. **Follow-ups:** (a) ✅ **done this pass** — `PROJECT.md` invariant
   #2 + Domain Concepts corrected to credit TaskTypes/Tasks and name the Family-as-junction; (b) ✅
   **Architecture committed this pass** — the many-to-many is already real (multiple architectures on
   the shared ModAdd TaskType), so `Architecture` is now an explicit Layer 1 object (Family →
   `architecture_id`; `n_layers`/`n_ctx`/widths moved off Variant onto it; `d_vocab` flagged
   task-coupled). A one-architecture-on-a-TaskType-*set* case would extend it further but isn't needed
   yet; (c) at build, confirm `Probe` and the train/test split (`data_seed`) hang off the
   **Task**/Variant correctly, and reconcile the `data/{family}/` on-disk layout with the new
   `TaskType`/`Task` split.
3. **Discriminator vs separate tables (L2).** Default to discriminator column; promote to two
   physical tables only when weight-only and activation-only attribute sets visibly diverge.
4. **Versioning surface.** `recipe_id` (L8) + object-scoped staleness must fold into REQ_145; the
   only genuinely new idea to lift is the *developer-declared semantic break* (recomputed-and-now-
   incomparable vs recomputed-still-comparable). Do not stand up a parallel versioning surface.
5. **Decomposition_Scalar discipline.** Enforce "nameable + stable ⇒ object column, not a tail
   row" in review, or the generic store regrows under a new name.
6. **Recipe Registry (method definitions) — build folds into REQ_138.** `Event.method_recipe` and
   `Decomposition.recipe_id` both dereference to a single registry of parameterized method/recipe
   definitions. This is the REQ_138 recipe-addressing surface **generalized** to register detection
   methods (event detectors), not only analysis recipes — *not* a new parallel registry. The model
   *names* it (Part II, Recipe addressing) so the resolution target isn't a dangling string; the
   build — schema, population, and the join from an `Event`/`Decomposition` row to its definition —
   is REQ_138's. Open sub-question: whether some detection methods are **Task**-scoped (e.g.
   frequency-based `specialize` reads through the irrep basis the Task provides), which ties to
   decision #2. **Committed this pass** as the PLANNED `MethodRecipe` Infrastructure object (Layer 1),
   keyed `(detection_method, method_recipe)`, that `Event`/`Decomposition` rows FK into — the model
   *names* it now so those FKs resolve, but the **build is deferred until event capture ramps** (events
   make the registry unwieldy fast, so it ships alongside the Event table, folding into REQ_138).
7. **Generic-association integrity (Event, Decomposition).** The polymorphic `(object_type,
   object_key)` key — with a *serialized* `object_key` — gives up DB-enforced referential integrity.
   Before building the Event extraction claims (and when next touching `Decomposition`), add (a)
   producer-side validation that the referent object key exists, and (b) an orphan-scan in the
   warehouse health check. These are the two sanctioned generic-association tables (Part II); the
   Layer 8 guard already says as much for `Decomposition_Scalar` — this generalizes it to the keys.

---

# Part VII — Changelog (from V1 / V2)

- **From V1 (`data-domain-model.md`):** `site` split into `WeightMatrix` + `ActivationSite`
  (resolves the double-declared collision); `FrequencyMode`/`FrequencyGroup` added (V1 was
  family-agnostic and dropped the central objects); `Decomposition` demoted from central join
  table to open-ended tail; `Event` introduced keyed `(variant, object_ref)`, replacing
  Variant-keyed events / `is_cross_epoch` flags; QK/OV circuits given real attributes.
- **From V2 (`datamodel.md`):** added Part I (the two tensions + resolutions) and Part II's
  decision procedure as first-class — the object-naming line of questioning now has a permanent
  home; added the `MLP_Neuron_Frequency` top-k child table (from the worked example); added the
  Store-realization bridge (Part IV) and the build-readiness scan (Part V); recorded the
  discriminator mislabel defect; made the hybrid-store write/read split explicit and answered the
  direct-write question.
- **Maturity model corrected (2026-06-27 review):** split the single "stability tier" into **three
  orthogonal axes** — Maturity (definitional: STABLE = definition not ours/accepted, EVOLVING =
  contested frame), Coverage (data populated), and Status (internal build lifecycle: ACTIVE /
  IN-DEV / PLANNED / CANDIDATE). `AttentionHead` → STABLE (was EVOLVING); `FrequencyGroup` →
  EVOLVING (was SOLID). Added the clock/pizza counter-example (capture-as-attribute / `State` as a
  candidate) and the "QK circuit (STABLE) vs. circuit-as-concept (EVOLVING)" distinction. Added
  the `demo_query_surface.ipynb` consumer-side example action.
- **Admission rule + literature harvest (2026-06-27 session):** added **Part 0 — What the Model
  Admits (the sieve)**: *given-real → object, earned-real → query*, the three given-realness sources
  (provenance / architecture / theorem), and the corollary *realness ≠ objecthood*. Added the
  **promotion criterion** (gauge-invariant function of the operand alone) to the decision
  procedure's attribute-home test (step 5), with the two grades of invariant
  (unconditional vs instrument-conditional). Added the harvested theorem objects to Layer 4
  (`FullOVCircuit`, `FullQKCircuit`, `DirectPath`) and named `IrrepBasis` as the STABLE
  family instrument underlying `FrequencyMode` in Layer 5. New companion:
  [canonical_object_harvest.md](canonical_object_harvest.md) (algebra-vs-behavior sieve over
  Mathematical Frameworks / Olsson / Nanda / superposition / SAE lines).
- **Reframe — sieve subsumes Tension 2; Medallion + Observability lenses (2026-06-28):** retired
  *"Tension 2 — the domain objects are part of the research"*; Part 0's admission rule already
  settles it, so Part I now carries the one live tension (hybrid store) plus the object apparatus
  (procedure + three axes) the sieve leaves behind. Added two orienting lenses to Part 0: the
  **Medallion** stack (Bronze = `npz` analyzer deposits, Silver = conformed object-shaped Warehouse
  tables + materialized derived/Event tables, Gold = the REQ_047 View Catalog) mapped onto Tension 1's
  write/read split, and **Observability** (Metrics = `[S]`/`[T]` time series; Events = `[E]` derived
  state-changes). Sharpened Layer 7: Events are **derived annotations on the timeline** (earned →
  materialized query, not given-real), **method-dependent** (added `detection_method` to the key; the
  loss velocity/acceleration + threshold detector anchors second descent & late instability across
  variants), and the **cross-variant alignment** spine — resolving the events-as-attributes question
  against scattering them onto objects.
- **Admission rule refined to a three-gate frame (2026-06-28):** the binary *given → object,
  earned → query* was too coarse — it contradicted FrequencyGroup/FrequencyMode already living as
  `EVOLVING` objects. Restated as **Gate 1** given-real → object `STABLE` (front gate, no litmus;
  theorem objects are objects even when `empty`), **Gate 2** earned-real → litmus (side gate): an
  *irreducible* (non-join, gauge-invariant) measurement admits it as an `EVOLVING` object (→
  FrequencyGroup, FrequencyMode), else it's a query forever (→ HeadPair, "switching"), **Gate 3**
  transverse → query regardless of certainty (the prior corollary). **The gate an object enters is
  its Maturity tier** (front → STABLE, side → EVOLVING), unifying the admission rule with Part I's
  Maturity axis. FrequencyGroup/FrequencyMode re-tagged *side-gate object*. Added the **Event vs
  Window** framing: Window is the coordination *band* a cluster of Events falls within, and its
  *width* is an earned measurement of coordination tightness. Event method extended with
  `method_recipe` (REQ_138 recipe-addressing) so the detector's **threshold** is part of the key —
  resolving the long-open *"when does a neuron specialize?"* as multiple traceable `specialize`
  events, each tied to its threshold definition, none promoted to given-real. Named the **Recipe
  Registry** that `method_recipe`/`recipe_id` resolve against (Part II *Recipe addressing* +
  Part VI build note #6 → folds into REQ_138, not a parallel surface). Recorded its **promotion
  path**: reference-metadata-in-code today, promotable to an Infrastructure object (Gate 1) if a
  queryable method surface is wanted.
- **Family decomposed into Architecture × TaskType, on a type/instance lattice (2026-06-28):** named
  the long-missing task side and split it by type/instance — **`TaskType`** (the task-logic container:
  builds the irrep basis *and* the master dataset, parameterized by `prime`) and **`Task`** (a resolved
  instance, e.g. ModAdd mod 109). **Task : TaskType :: Variant : Family.** `Family_Parameter` →
  `TaskType_Parameter` (the name already read "per-family *task* parameters"); `Variant_Parameter` →
  `Task_Parameter_Value` (prime identifies the *Task*, not the run; still the cross-model comparison
  spine). Family recast as the (Architecture × TaskType) *template* pointing to a TaskType (FK
  `tasktype_id`); Variant gains a `task_id` binding plus the explicit **Model Parameters / Task
  Parameters** distinction (`model_seed`/`data_seed` are per-run Model Parameters; `prime` is a Task
  Parameter). Irrep basis + all Layer 5 objects moved to TaskType/Task (Z/113 ≠ Z/109 ⇒
  Task-instance-level); Layer 5 renamed **Task-Scoped Objects**; the `family` semantic gate recast as
  **`task`** (the two gates = the two junction factors); `Probe` repointed to `Task`. Resolves Part VI
  #2 (claims registry gains a *TaskType* dimension, honoring invariant #2).
- **Consolidation pass + commits (2026-06-28):** swept `family`→`task`/`TaskType` wording across Part 0
  (Gate 1 infra list + theorem basis), the decision procedure (steps 5–6), Layer 4 (`operand_symmetry`,
  the `*-conditional` dominant-frequency notes), and the Layer 5 header. **Committed `Architecture`** as
  an explicit Layer 1 object (the many-to-many is already real — multiple architectures on ModAdd):
  Family → `architecture_id`, structural config (`n_layers`/`n_ctx`/widths) moved off Variant, with the
  `d_vocab` task-coupling note. **Committed `MethodRecipe`** (the method/recipe registry) as a PLANNED
  Infrastructure object — named now so Event/Decomposition recipe FKs resolve, built when event capture
  ramps. Updated `PROJECT.md` invariant #2 + Domain Concepts to credit TaskTypes/Tasks as the context
  providers and name the Family-as-junction (Architecture/TaskType/Task added as concepts).
- **Merged parallel sweep (2026-06-28):** reconciled a second instance's validation pass. Edit A
  (invariant #2 / glossary) was already covered by our consolidation (superset — we also added the
  `Architecture` concept), keeping only its "instrument doesn't change shape … *or the task*" nuance.
  Adopted three additive sharpenings: **generic-association integrity** — `Event` + `Decomposition` are
  the two controlled tables keyed by a *serialized* polymorphic `(object_type, object_key)`, which
  forfeits DB referential integrity → producer-side validation + orphan-scan discipline (Part II
  convention + Part VI #7); **`detection_method` vs `method_recipe` independence** — method *identity*
  vs *parameters-only* signature, both in the key, the pair resolving via `MethodRecipe` (keeps "all
  events from detector X" cheap); **IrrepBasis vs invariant #1** — the Fourier transform is the
  universal instrument, the Task only supplies the basis *parameter*, so #1 and #2 are both honored.
- **Retired V1 + V2 files** — `data-domain-model.md` and `datamodel.md` deleted; this is canonical.
