# REQ_128: Analyzer Input Provisioning — Lazy Accessor vs. Eager Materialization

**Status:** Active — design decided 2026-05-28. Conditions of satisfaction below; ready for implementation after REQ_102 (see Ordering).
**Priority:** High — foundational. Establishes the input contract that **all** analyzers (new and rewritten) implement going forward; the eager-materialization pattern is also a standing memory-pressure source that compounds as analyzers chain and as artifacts grow more granular.
**Branch:** `feature/req-128-analyzer-input-provisioning` (off `develop`). Design exploration originated on `feature/generic_analyzer` (parked).
**Dependencies:**
- REQ_106 (Analysis Layer Architecture — defines the declared-dependencies discipline this REQ extends to the *loading* of those dependencies).
- REQ_109 (primitive layer — the granular analyzers that make whole-artifact loads expensive consume it).
- REQ_122 (storage-encapsulation invariant — constrains the solution space: analyzers must reach data through accessors, never raw paths).

**Ordering (decided 2026-05-28):** **REQ_102 (Analyzer Deprecation) lands first**, then REQ_128. REQ_102 is subtractive — it retires ~9 superseded analyzers. Doing it first shrinks REQ_128's rewrite surface (no point porting analyzers that are about to be deleted) and lets REQ_128's rewrite of the *survivors* subsume REQ_102's REQ_106 layering-audit concern for data-plane access (the `inputs.artifacts_dir` + roll-your-own-`ArtifactLoader` pattern is exactly what this REQ removes). REQ_102 is fully unlocked (REQ_111/117/126 in staging); it does not depend on the new input contract.

**Relationships (not hard dependencies):**
- REQ_127 (Downstream Visualization Migration — *complete*). The **consumer-side analog** of this problem: REQ_127 fixed an OOM where view loaders stacked whole granular artifacts via `load_epochs(...)` and resolved it with `fields=` selective loading. This REQ is the **producer/chaining-side** version of the same tension.
- REQ_129 (v1.0.0 Dead-Code Cull — *stub*). Sibling cull track. REQ_128 removes dead code **on the input-provisioning path** (it is co-located with the rewrite); REQ_129 captures the broader, unrelated v1.0.0 scaffolding cull so this REQ stays atomic.
- The Plan/Planner / Freshness infrastructure (from the ETL refactor) is the adjacent system that decides *what* to compute; this REQ is about *how the data flows in* once computation is scheduled.

**Attribution:** Engineering Claude (drafted 2026-05-27, surfaced during REQ_127's memory investigation and recalling the `feature/generic_analyzer` ETL exploration; design decided with the user 2026-05-28).

---

## Problem Statement

The analysis pipeline materializes analyzer inputs **eagerly**. For each per-epoch analyzer, `pipeline._materialize_per_epoch_inputs` builds a `ResolvedInputs` holding `model`, `cache`, `logits`, `probe`, and — for every `ArtifactInput` the analyzer's Spec declares — the **fully loaded** upstream artifact (`loader.load_epoch(dep, epoch)` / `load_summary` / `load`, with no field filter). A chained analyzer (e.g. `centroid_fourier_alignment` depending on `repr_geometry`) therefore receives the entire upstream artifact dict in memory, whether or not it reads all of it.

This couples two things that should be separable:

1. **What an analyzer depends on** (declared, static — good).
2. **When and how much of that dependency is resident in memory** (currently: all of it, for the lifetime of the `ResolvedInputs` object).

Two consequences follow:

- **Whole-artifact loads even for partial reads.** Post-REQ_126 granular analyzers emit large per-site cubes (e.g. `mlp_out_power` at `(d_mlp, n_freq, n_freq)`). A chained analyzer that needs one small key still triggers a load of the whole artifact. This is the *same* failure mode REQ_127 hit on the consumer side and fixed with `load_epochs(fields=...)` — but the producer/chaining side has no equivalent lever today.
- **Memory lifetime tied to an object, with no clean release.** REQ_127 Phase A.2f added a defensive fix in `_run_single_epoch` (null `inputs`/`result` before `del model, cache, ...` so `torch.cuda.empty_cache()` can reclaim). That treats the *symptom* (the last analyzer's `ResolvedInputs` keeps model/cache references alive past the explicit cleanup) rather than the *structure* (eager materialization makes the resident set as large as everything declared, released only when the inputs object dies).

The `feature/generic_analyzer` ETL exploration surfaced the core design question directly: **how should data be passed to analyzers?** If everything is pre-loaded and handed to the input object, it can be a lot of data with no clean release path — especially for chained analyzers. One option considered there was to pass **file paths** that each analyzer loads on demand.

The opportunity: pick a provisioning model that lets analyzers load **only what they need, when they need it, in a releasable scope**, without eroding the declared-dependency discipline or the storage-encapsulation invariant.

---

## Context: the design axes (framing, not decisions)

Three provisioning models, with the tension between ergonomics and control:

1. **Eager dict (current).** The pipeline pre-loads each declared dependency and hands the analyzer a ready dict (`inputs.artifacts["repr_geometry"]`). Dead simple for analyzer authors; centralized loading. But loads everything declared, and memory lifetime is the inputs object's lifetime.

2. **Raw file paths.** The analyzer receives paths and loads on demand. Gets on-demand release and selective reads — but **violates the storage-encapsulation invariant** (PROJECT.md constraint 3: only storage primitives compose paths; everything else uses accessors). An analyzer composing paths reaches past the API. Likely off the table for this reason, but recorded because it was the original idea on `feature/generic_analyzer`.

3. **Lazy accessor handle (recommended starting point).** The analyzer receives a scoped `ArtifactLoader` (or a thinner accessor) and calls `load_epoch(dep, epoch, fields=[...])` on demand. This is the invariant-preserving version of the path-passing idea: it gives the same on-demand, selective, releasable loading, but the analyzer still reaches data only through the accessor. The cost is ergonomics — a `.load()` call instead of a ready dict — and the need for a convention (or Spec-level validation) that an analyzer only loads what its Spec declares, so the declared-dependency discipline doesn't erode into ad-hoc reads.

**Decision (2026-05-28): Model 3, taken to its full form** — a scoped, lazy accessor with streaming as the spine and `fields` required. Models 1 and 2 are rejected (eager retains the memory problem; raw paths violate the storage-encapsulation invariant). The minimal "fields-only eager" half-measure floated in the original Notes is **not** taken — the user authorized rewriting all analyzers, which removes the back-compat constraint that made the half-measure attractive. See **Design** below.

### Codebase audit (2026-05-28, on `develop`)

Evidence that grounded the decision and sizes the work:

- **The lazy-accessor pattern already won, informally.** All 11 cross-epoch analyzers already do `loader = ArtifactLoader(inputs.artifacts_dir); loader.load_epochs(name, fields=[...])` — on-demand, selective, scoped-by-convention. **41** references to `inputs.artifacts_dir` across analyzers.
- **The eager cross-epoch / summary materialization is dead.** `_materialize_cross_epoch_inputs` → `_best_effort_load_all_epochs` eagerly calls `loader.load_epochs(name)` **with no `fields=`** (full stack) into `cross_epoch_artifacts` / `summary_artifacts`. **Zero** analyzers read those fields. This is the `all_epochs` granular-cube path — the prime memory suspect — loading the whole stack into a container nobody reads.
- **The eager per-epoch dict has 5 readers** (`inputs.artifacts[...]`, all `scope="epoch"`). `ArtifactLoader.load_epoch` (single epoch) **lacks** a `fields=` lever today; that gap closes here.
- **Declaration breakdown:** 16 analyzers declare `ArtifactInput`; 14 `all_epochs` (all on `output_scope="cross_epoch"`), 5 `epoch`, 0 `summary`.
- **The contract is already unified.** **0 of 34** analyzers author the legacy `category=` Spec style; 32 are unified, 2 have no Spec. The REQ_120/121 coexistence layer (`category` authoring, `is_unified`, the `if not spec.inputs` legacy branches) is dead and co-located — culled here.
- **Model side is small and separable:** 7 analyzers read `inputs.cache`, 7 read `inputs.model`. Out of scope for this REQ (see Constraints).

**Resolutions to the original open questions:**

- *Selective fields:* `fields` is **required** on every accessor call (with an explicit `"all"` escape hatch), not an optional per-`ArtifactInput` declaration. Selection moves to the call site, where the analyzer knows what it reads.
- *Declared-dependency enforcement:* the accessor is **scoped to the analyzer's declared `ArtifactInput`s** and **raises** on any undeclared name. Discipline becomes structural, not conventional — recovering what the `artifacts_dir` pattern silently lost.
- *Per-epoch cleanup band-aid:* removed. Lazy provisioning means artifact references are local to `analyze()` and released on return; the A.2f reference-nulling is subsumed for the artifact side.
- *Cross-epoch / summary scopes:* `stream()` (one epoch resident) for reducers; `load_stack(fields=...)` for true whole-matrix consumers (SVD/PCA/DMD). The dead eager cross/summary path is deleted.
- *Migration cost:* bounded by the survivor set after REQ_102 (~25 analyzers). Big-bang rewrite, no coexistence shim (pre-v1.0.0, no external back-compat).

---

## Design (decided 2026-05-28)

### The split being fixed

The current contract conflates **declaration** (what an analyzer depends on — static, on the Spec) with **provisioning** (when and how much is resident — currently eager, all-of-it, for the lifetime of `ResolvedInputs`). The redesign separates them: the Spec still declares; a scoped accessor provisions lazily.

### Storage alignment (why streaming, not just a perf knob)

The per-epoch artifact layout from REQ_021f — `artifacts/{analyzer}/epoch_{NNNNN}.npz`, one file per epoch — is a **streaming-native store**. Eager `load_epochs(name)` fought that grain by globbing every file into one array to synthesize a shape the storage never had. Streaming-first reads the store the way it is written: one epoch at a time. This is removing an impedance mismatch, not adding a cache — and it is the seam where async slots in later (`async for` over an iterator, no analyzer-contract change).

### The contract

`ResolvedInputs` slims to: `epoch` (or `epochs` for cross-epoch), the model side (`model` / `cache` / `logits` / `probe`, unchanged), and a single scoped accessor (`deps`). The eager `artifacts` / `cross_epoch_artifacts` / `summary_artifacts` dicts and the raw `artifacts_dir` string all collapse into `deps`.

`deps` is scoped to the analyzer's declared `ArtifactInput`s and exposes four verbs, each honest about its memory shape, all with **required `fields`** (verb names locked 2026-05-28):

| Verb | Shape | Memory | For |
|---|---|---|---|
| `deps.load_epoch(name, epoch, *, fields=[...])` | one epoch's dict at the given epoch | one epoch | per-epoch readers; also reference-epoch reads inside cross-epoch analyzers |
| `deps.stream(name, *, fields=[...])` | iterator yielding `(epoch, dict)` | one epoch at a time | reducers / trajectory metrics; async-upgradeable (the spine) |
| `deps.load_stack(name, *, fields=[...])` | stacked `(n_epochs, …)` array | whole stack (named so the cost is legible at the call site) | SVD / PCA / DMD that need the full matrix resident |
| `deps.load_cross_epoch(name, *, fields=[...])` | the upstream's single `cross_epoch.npz` dict | the one artifact | upstreams that are themselves cross-epoch analyzers (neuron_dynamics, neuron_group_pca, global_centroid_pca) |

The first three verbs are for **per-epoch** upstreams; `load_cross_epoch` is for **cross-epoch** upstreams. Calling a per-epoch verb on a cross-epoch upstream (or vice versa) raises a layout-mismatch error. `load_epoch` takes an **explicit epoch** so it serves both per-epoch analyzers' current-epoch reads and cross-epoch analyzers' reference-epoch reads.

- **`fields` required**, with `fields="all"` (or an `ALL` sentinel) as the explicit "I need everything" escape hatch. Requested fields are **validated against the upstream artifact's actual on-disk field set** — read cheaply from the `.npz` index (`np.load(path).files`, no array load) — so a misspelling raises early, not at `np.stack`. (Declared *output schema* — semantic field descriptions, registry-load enforcement, drift detection — is **out of scope here and owned by REQ_107**; see Design lock below.)
- **Scope-enforced:** `deps.<verb>("undeclared_name", …)` raises.
- Underneath, `deps` is a thin wrapper over `ArtifactLoader` (still the storage primitive — invariant intact); the pipeline constructs it per-analyzer from `spec.inputs`.
- `model` / `cache` / `logits` stay eager (they come from the forward pass the pipeline must run anyway; `ModelInput.needs_cache` already gates it).

### Design lock (2026-05-28, post re-audit)

A fresh re-audit against the **current 26-analyzer survivor surface** (REQ_102 partially landed; deferments → REQ_130/131) confirmed the thesis and refined it:

- **Surface partitions cleanly into two patterns.** 4 analyzers read a `scope="epoch"` upstream via the eager `inputs.artifacts[name]` dict (neuron_grouping, weight_basis_projection, centroid_fourier_alignment, fourier_frequency_quality). 10 analyzers read `scope="all_epochs"` upstreams by rolling their own `ArtifactLoader(inputs.artifacts_dir)`. The eager `cross_epoch_artifacts` / `summary_artifacts` dicts have **zero readers** — pure waste (worst case: `parameter_dmd` triggers a full stack of all 9 `parameter_snapshot` weight matrices × all epochs, immediately discarded, then re-loads with `fields=["W_in","W_out"]`). This is the prime memory suspect, confirmed.
- **`ArtifactInput.scope` is dropped.** The planner never reads it (`derive_required_artifacts` uses only the name; `derive_category` uses input *types* + `output_scope`); only the deleted eager `_materialize_*` paths consumed it. `ArtifactInput` collapses to `ArtifactInput("name")` — a pure dependency declaration; the verb at the call site determines access shape. The `"summary"` scope (0 users) and the `ArtifactScope` literal disappear.
- **Field validation uses npz keys, not the manifest.** The manifest is left untouched. Declared output-schema infrastructure (explicit field/dtype/description declarations, registry-load enforcement, drift detection) belongs to **REQ_107**, which constrains schema to live *in code, not a parallel manifest* (REQ_107 CoS lines 34/37, Constraint 61). REQ_128 needs only a runtime "does this field exist in the data I'm loading?" check, satisfied cheaply by `np.load(path).files`. Future seam: once REQ_107 lands, the accessor may additionally cross-check requested fields against the *declared* schema. (If npz-key availability proves problematic at implementation time, reassess against this finding.)
- **Undeclared-read reconciliation is in scope.** Migration reconciles each analyzer's `ArtifactInput` declarations with its *actual* reads. Confirmed instance: `activation_dmd` declares `repr_geometry` but actually reads `global_centroid_pca` via `load_cross_epoch` (the declaration is a planner-ordering fiction — see its line-58 comment); it must declare `global_centroid_pca`. Scope enforcement surfaces these.
- **REQ_130/131 collision handled by keeping old upstream names.** The 4 analyzers still pointing at to-be-deprecated upstreams (`neuron_freq_norm` ← neuron_dynamics / neuron_group_pca / freq_group_weight_geometry; `dominant_frequencies` ← fourier_frequency_quality) are rewritten onto the accessor **keeping the old upstream name** this pass; REQ_130/131 re-point afterward. Keeps REQ_128 atomic to "the input contract changed."

This lock **supersedes** any manifest-based `fields`-validation language elsewhere in this doc (the `manifest.json` validation bullets in the Resolutions and CoS sections).

---

## Conditions of Satisfaction

### Accessor & contract

- [ ] A scoped dependency accessor type exists, constructed by the pipeline per-analyzer from `spec.inputs`, wrapping `ArtifactLoader` (the storage primitive — no path composition leaks into the accessor's callers).
- [ ] `ResolvedInputs` carries the accessor and drops the eager `artifacts`, `cross_epoch_artifacts`, `summary_artifacts` dicts and the raw `artifacts_dir` string.
- [ ] The accessor exposes `load_epoch`, `stream`, `load_stack`, and `load_cross_epoch` (the four shapes must exist), each requiring `fields`.

### `fields` discipline

- [ ] `fields` is a required argument on every accessor verb; there is no implicit "load everything" default.
- [ ] An explicit escape hatch (`fields="all"` or an `ALL` sentinel) loads all fields, legibly.
- [ ] Requested fields are validated against the upstream artifact's actual on-disk field set (read from the `.npz` index, no array load); an unknown field raises a clear error naming the artifact and the bad field. (Declared-schema validation is REQ_107, not here.)

### Streaming spine

- [ ] `stream(name, fields=[...])` yields one epoch at a time and holds at most one epoch's data resident (verified by a memory assertion or peak-RSS check on a multi-epoch artifact).
- [ ] `load_stack` is the *only* verb that materializes the full `(n_epochs, …)` array; cross-epoch analyzers that can reduce epoch-by-epoch use `stream`.
- [ ] The iterator interface is shaped so a later async variant requires no analyzer-contract change (documented; not implemented here).

### Scope enforcement

- [ ] Calling any accessor verb with an analyzer name not in the analyzer's declared `ArtifactInput`s raises.
- [ ] Each migrated analyzer's `ArtifactInput` declarations match its actual reads (undeclared upstreams — e.g. `activation_dmd` → `global_centroid_pca` — are declared, not left as runtime reads).
- [ ] A test (REQ_106-style) asserts the scoping holds for the registered analyzer set.

### Cull (input-provisioning path only)

- [ ] `_materialize_cross_epoch_inputs`, `_best_effort_load_all_epochs`, and the eager `_materialize_per_epoch_inputs` artifact-loading branches are removed/replaced by accessor construction.
- [ ] The REQ_127 Phase A.2f reference-nulling band-aid (`inputs = result = summary = None` before `del`) is removed once lazy provisioning makes it unnecessary, with a note confirming GPU/CPU memory still releases per epoch.
- [ ] The dead legacy `category`-authoring scaffolding co-located on the input path is removed: `AnalyzerSpec.category` authoring + `is_unified` + `effective_*` legacy branches + the `if not spec.inputs` paths in the pipeline. (Unrelated dead code → REQ_129.)

### Analyzer migration (all survivors)

- [ ] Every surviving analyzer (post-REQ_102) that reads upstream artifacts is rewritten to the accessor contract: no `inputs.artifacts[...]`, no `inputs.artifacts_dir`, no self-constructed `ArtifactLoader`.
- [ ] Each rewritten analyzer requests only the fields it uses (no blanket `"all"` unless genuinely needed; flagged in review if used).
- [ ] `ArtifactInput` declarations remain the single source of truth for what each analyzer may load.

### Validation

- [ ] Parity: re-analysis of the canon reference set (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598) produces artifacts matching the pre-change outputs within `rtol=1e-3` (per [[feedback_req126_float64_parity]]). Shape-of-behavior changes are findings, not noise.
- [ ] Memory: re-analyze `p101/s999/ds598` and record **RSS vs. system page-cache separately** (`/proc/<pid>` vs system cache) before/after. The redesign is justified architecturally regardless; this measurement *calibrates the memory claim* and tells us whether residual climb is WSL2 page-cache (environmental) rather than Python retention. A clean architectural follow-up on remaining memory pressure, if any, is a separate track.

---

## Constraints

**Must:**
- Reach all stored data through the accessor / `ArtifactLoader`; no raw path composition outside the storage primitive (PROJECT.md constraint 3).
- Keep `ArtifactInput` (Spec) as the authority on dependencies; the accessor enforces, it does not widen.
- Existing on-disk artifacts remain readable unchanged — this changes *how inputs flow in*, not the storage format.

**Must avoid:**
- A blanket "load everything" default — `fields` is required precisely to prevent the silent whole-artifact loads that caused the problem.
- Re-introducing an unscoped loader handle that lets an analyzer read undeclared dependencies.
- Bundling unrelated v1.0.0 dead-code removal into this REQ (→ REQ_129) — keep this atomic to "the input contract changed."

**Flexible:**
- Verb names (`stream`/`collect`/`materialize`, etc.) and whether `load_epoch` is a distinct verb or a degenerate `stream`.
- Whether the accessor is a new thin type or a scoped subclass/wrapper of `ArtifactLoader`.
- Migration order across survivors. Default: simplest reducers first, then the `load_stack` (SVD/PCA/DMD) consumers.

**Out of scope (boundary preserved for later):**
- **Model-side provisioning.** `model` / `cache` / `logits` stay on the outer-loop forward pass. The accessor must not foreclose extending the same epoch-stream treatment to the model side later (the eventual "one stream of epochs, each yielding model + cache + upstream artifacts" north star) — but that unification is explicitly *not* built here.

---

## Notes

- **This is the producer-side twin of REQ_127.** That REQ's `feedback_granular_analyzer_load_epochs_fields` lesson (consumer loaders must use `fields=`) is the same insight applied to view loaders; REQ_128 applies it structurally to the *analyzer input* path rather than per-call.
- **The "minimal fields-only eager" half-measure was considered and rejected (2026-05-28).** It would have let `ArtifactInput` declare fields and kept eager materialization — attractive only under a back-compat constraint. The user authorized rewriting all analyzers (no external back-compat pre-v1.0.0), so the fuller lazy + streaming accessor is taken instead; it solves both the *amount* (fields) and the *lifetime* (lazy/streaming) halves of the problem rather than just the first.
- The REQ_127 Phase A.2f pipeline band-aid (`inputs = result = summary = None` before `del`) is subsumed and removed here (see CoS → Cull).

### Field evidence (2026-05-27, observed during REQ_127 close-out)

Re-analyzing `p101/s999/ds598` (full pipeline, now running the new granular
analyzers per the updated family.json) reproduced the memory problem on the
**analysis side**, distinct from the view-side OOM REQ_127 fixed: disk I/O was
infrequent while process memory climbed and did **not** release until the WSL
instance was restarted. Two candidate contributors to separate during scoping:
(1) eager materialization of the larger new artifacts for chained analyzers
(the structural concern this REQ targets); (2) WSL2's tendency to hold
page-cache memory and not return it to the host during heavy file I/O, which
can present as a non-releasing climb independent of any Python-level retention.
A scoping pass should measure RSS-vs-page-cache separately (e.g. `/proc/<pid>`
RSS vs system cache) before attributing the climb, so the fix targets the real
cause rather than an environmental artifact. Note the secondary-analyzer
per-epoch loop in `pipeline.py` (where chained analyzers like
`centroid_fourier_alignment` run) lacks even the A.2f reference-release
cleanup that `_run_single_epoch` got — a concrete first place to look.
