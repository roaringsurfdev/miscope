# REQ_110: Lakehouse Surface (Tabular Output, DuckDB Query, Publication Bundles)

**Status:** Draft
**Priority:** High — second of the two consolidation streams; the data surface for v1.0 publication.
**Branch:** TBD
**Supersedes:** REQ_101 (DataFrame Support), REQ_108 (Publication Surface).
**Dependencies:** REQ_109 (Measurement Primitives — typed measurement results are the upstream shape that flatten into tabular form); REQ_106 (DataView first-class status, layering principle); REQ_107 (registry — adjacent, not merged; supplies the discoverability layer this REQ's tables are queryable through); REQ_103 (the package + repo + docs surface that hosts the publication tooling).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Today, analyzers emit `.npz` artifacts and JSON summaries. That's the right shape for in-memory analysis driven by the dashboard or notebooks: load one variant's artifacts, slice them, render a figure. It's the **wrong** shape for two adjacent use cases the project has accumulated demand for:

1. **Cross-variant queries.** Questions like "show me all variants where attention Fourier alignment committed after grokking" or "list every variant with homeless_fraction > 0.2 across data seeds." Today these require Python-level aggregation across `variant_registry.json` and per-variant summaries. A SQL `WHERE` clause is the natural shape.
2. **External review.** A researcher reviewing a fieldnotes claim wants to verify the data underlying a chart without installing the platform and re-running the pipeline. The conventional path (publish raw `.npz`, expect re-run) has too much friction; the data is also the wrong shape — review wants the *derived* metrics, not the raw weights.

The opportunity surfaced during the discovery phase: **lean into a small-scale lakehouse pattern.** Analyzers (the ones that genuinely transform) emit Parquet files alongside (or instead of) `.npz`. Parquet is columnar, self-describing, language-agnostic, and HTTP-range-readable. DuckDB queries Parquet over local paths or HTTP, with no daemon and a single Python wheel. DuckDB-WASM does the same in the browser, making published data inline-queryable inside fieldnotes articles.

This REQ defines the tabular output surface end-to-end: the in-memory DataFrame contract, the on-disk Parquet contract, the cross-variant query layer, and the publication workflow that attaches curated bundles to GitHub Releases.

---

## Conditions of Satisfaction

### Tabular schema design

The schema design is the load-bearing decision. Long-format with explicit discriminator columns is the canonical shape.

- [ ] **Long-format canonical.** All analyzer-emitted tables and DataView outputs are long-format with explicit dimension columns and value columns. Wide-format is a presentation concern produced by `to_wide(...)` at the consumer.
- [ ] **Standard dimension columns** (present in every long-format table where applicable):
  - `variant_id` — variant identifier (always present; cross-variant joins are a primary use case).
  - `epoch` — checkpoint epoch (present for per-epoch and trajectory tables; absent for cross-epoch summary tables).
  - `group_type` — discriminator for what `group` refers to. Enum values:
    - `weight_matrix` — single weight matrix (e.g., `W_E`, `W_in`).
    - `weight_component` — combination of weight matrices for a component (e.g., `MLP`, `Attention`, `All`).
    - `activation_site` — activation site (e.g., `mlp_out`, `attn_out`, `resid_post`).
    - `frequency_group` — frequency group within a weight matrix.
    - `single_centroid` — a single class centroid.
    - `centroid_group` — class centroids for an activation site.
  - `group` — the group identifier itself (string; semantics determined by `group_type`).
  - `operation_type` — discriminator for the measurement / transform that produced the row. Enum values map to the three PCA modes from REQ_109 plus the analytical extensions:
    - `pca` — base PCA on a single sample set (per-epoch).
    - `pca_summary` — trajectory PCA: one basis fit across a stack of sample sets (also called *summary PCA*).
    - `pca_rolling` — windowed PCA across epochs.
    - `velocity`, `acceleration` — derivative measures.
    - `circularity`, `fourier_alignment`, `curvature`, `sigmoidality`, `lissajous`, `procrustes`, `arc_length`, `loop_area`, `self_intersection`, `jerk` — shape characterizations from REQ_109.
- [ ] **Discriminator-driven schema.** The `group_type` and `operation_type` columns are explicit (not parsed from the `group` string or table name). This makes `WHERE group_type = 'activation_site' AND operation_type = 'pca_summary'` a clean filter.
- [ ] **Schema co-located with DataView definition** (per REQ_106). Each DataView declares its long-format schema (column names, dtypes, semantic descriptions). The schema is part of the published API.

### PCA tables (the worked example)

PCA appears across the platform in many flavors. The user's overview defines the canonical shape: two tables per PCA operation.

- [ ] **`pca_results`** — one row per (operation, principal component):
  - `variant_id`, `epoch`, `group_type`, `group`, `operation_type`,
  - `pc_index` — principal component index (0-based),
  - `singular_value`, `eigenvalue`, `explained_variance`, `explained_variance_ratio`,
  - `participation_ratio`, `spread` — operation-level scalars (denormalized for query convenience),
  - `basis_vector` — the eigenvector as an array column (parquet supports this natively).
- [ ] **`pca_projections`** — one row per (operation, sample, principal component):
  - `variant_id`, `epoch`, `group_type`, `group`, `operation_type`,
  - `row_id` — sample identifier (e.g., neuron index, centroid label),
  - `pc_index`,
  - `projection_value`.
- [ ] All PCA-emitting analyzers populate both tables. Existing renderer-side PCA migrates to consume from these tables (via DataView loaders).

### Fourier tables

Defined to a similar level of structure as PCA:

- [ ] **`frequency_spectrum`** — one row per (variant, epoch, site, frequency):
  - `variant_id`, `epoch`, `site_type` (weight vs activation), `site`, `frequency`, `magnitude`, `derivation` (provenance string).
- [ ] **`learned_frequencies`** — one row per (variant, epoch, site, frequency, threshold, method):
  - `variant_id`, `epoch`, `site_type`, `site`, `frequency`, `threshold_name`, `commitment_method`, `is_committed` (bool), plus any method-specific scalars (e.g., `frac_explained` for `NEURON_DOMINANT`).
- [ ] **`neuron_frequency_attribution`** — one row per (variant, epoch, neuron, frequency):
  - `variant_id`, `epoch`, `neuron_idx`, `frequency`, `frac_explained`, `dominant` (bool).
  - Used by transient-frequency analysis and the neuron specialization views.

### Shape characterization tables

- [ ] **`shape_characterizations`** — one row per (variant, epoch, group, characterization):
  - `variant_id`, `epoch`, `group_type`, `group`, `operation_type` (the characterization name from REQ_109), `value` (scalar), and a sidecar `parameters` array column for multi-valued characterizations (e.g., Lissajous frequency-ratio + phase-offset + amplitudes).
- [ ] Multi-output characterizations (Lissajous, saddle curvature) may also emit a sibling table with explicit columns rather than a packed `parameters` array. Decide per characterization in implementation; default to explicit columns when the parameter set is fixed.

### In-memory DataFrame surface

- [ ] **DataView returns pandas DataFrame** (already the contract per REQ_101). Long-format with the dimension columns above.
- [ ] **`BoundDataView.to_wide(index, columns, values)`** wraps `pandas.pivot`. Researchers pivot for plotting.
- [ ] **Cross-variant concatenation works trivially:** `pd.concat([variant.dataview(name).data() for variant in variants])`. The `variant_id` column makes this a no-op; the long format requires no schema reconciliation.
- [ ] **Schema documented per DataView**, declaring which columns are the natural index for `to_wide()`.

### On-disk Parquet surface (internal warehouse)

- [ ] **DataView materializations write Parquet files** as their canonical persisted form.
- [ ] **Internal Parquet path:** `results/{family}/{variant}/dataviews/{view_name}.parquet` — sibling to existing `.npz` artifact directories. Layout canonicalized in implementation.
- [ ] **Coexistence with `.npz`.** Parquet does **not** replace `.npz` blanket. The two coexist:
  - Analyzers that perform a transform (PCA, Fourier, geometry, shape characterization) emit Parquet alongside or instead of `.npz`.
  - Analyzers that are extract-only — `parameter_snapshot` (raw weight matrices) and the activation-capture analyzers — continue to emit `.npz`. They have no transform output to tabulate; their job is to make raw tensors available without re-loading the model checkpoint.
  - Per-analyzer decision is documented in the analyzer's spec. Default for new analyzers: emit Parquet if there's a transform; emit `.npz` if there's no transform.
- [ ] **Internal Parquet is gitignored**, alongside `.npz` artifacts. Deterministically regeneratable.
- [ ] **Parquet writer:** `pyarrow` (already a `miscope` dep per REQ_103). Compression: snappy or zstd — pick one in implementation; zstd preferred for published bundles, snappy or none for internal warehouse (faster writes during pipeline runs).

### Cross-variant query layer (DuckDB)

- [ ] **DuckDB available as a `miscope` dep** (REQ_103 lists it).
- [ ] **`miscope.query`** module exposes a thin wrapper that opens a DuckDB connection bound to a chosen Parquet root (local path or URL):
  ```python
  con = miscope.query.open(family="modulo_addition_1layer")
  df = con.sql(\"\"\"
      SELECT variant_id, epoch, magnitude
      FROM frequency_spectrum
      WHERE site = 'mlp_out' AND frequency = 25
      ORDER BY variant_id, epoch
  \"\"\").df()
  ```
- [ ] **Local and HTTP read paths supported** with the same query surface. DuckDB issues range requests for Parquet over HTTP; the cost of a remote query is the bytes the query touches, not the file size.
- [ ] **Cross-table joins work via DuckDB's native Parquet handling** — no custom join logic in `miscope`. Example: join `frequency_spectrum` against `shape_characterizations` on `(variant_id, epoch)` to correlate frequency content with circularity.
- [ ] **DuckDB views over the Parquet root** for ergonomic naming. The user runs `SELECT * FROM frequency_spectrum`, not `SELECT * FROM 'results/.../frequency_spectrum.parquet'`.

### Publication: bundles, GitHub Releases, schema stability

- [ ] **Internal warehouse vs. published bundle is a structural distinction**, not nominal.
  - Internal warehouse: `results/.../dataviews/*.parquet`. Free to evolve, regeneratable, gitignored. Schemas can change with code; mismatch is a re-run.
  - Published bundle: a curated subset of Parquet files, frozen at a point in time, attached to a `data-*` GitHub Release. Once published, immutable.
- [ ] **Bundles are per-article, not all-of-everything.** A fieldnotes article on ring geometry publishes only the Parquet files that underlie its claims. Avoiding kitchen-sink bundles keeps file size small, schemas tight, and review focused.
- [ ] **Build script** (`scripts/build_publication_bundle.py` or `miscope.publish.build`) — selects DataView outputs by name, freezes their schemas, writes them to a versioned output directory, computes content hashes, and emits a manifest.
- [ ] **Bundle manifest** (`manifest.json` inside the bundle) declares: `bundle_version`, `mint_date`, `miscope_version`, list of included Parquet files with their schemas and content hashes, the article(s) referencing the bundle, and a description.
- [ ] **GitHub Releases as host.** Bundles attach to Releases tagged `data-*` (e.g., `data-v1.0-ring-geometry`). Distinct from package release tags (`v1.0.0` for `miscope`). Per-bundle Parquet files attached as Release assets with stable URLs.
- [ ] **GitHub Action workflow** (`.github/workflows/data-release.yml`): build bundle → create Release → attach Parquet + manifest → publish.
- [ ] **Range-request and CORS verified** (not assumed): a DuckDB query against a Release asset URL fetches only the bytes the query needs; CORS headers permit fetches from the GitHub Pages domain.
- [ ] **Schema stability per bundle version.** A column added or renamed requires a new bundle version with a new tag. Old bundles remain accessible at original tags. The build script reads declared schemas (per REQ_106 / REQ_107) and fails to publish if materialized Parquet doesn't match.

### DuckDB-WASM in fieldnotes (inline queries)

- [ ] **Fieldnotes Astro site loads DuckDB-WASM as a client-side module**, lazy-loaded on demand (the WASM bundle is several MB).
- [ ] **`<DuckDBQuery>` MDX component** accepts a Parquet URL (a Release asset) and a SQL query string. Renders a result table inline. Reader can edit the query and re-run.
- [ ] **Articles use the component to back specific claims** with inline-queryable data. Errors surface clearly to the reader; a broken query is recoverable by re-editing in-browser.

### Validation

- [ ] **End-to-end pipeline operational:** build a publication bundle from real DataView outputs → attach to a GitHub Release → query a Parquet asset via DuckDB-WASM from a deployed fieldnotes article → reader sees result table inline. This is the v1.0 acceptance bar.
- [ ] **Schema drift test:** rebuild a bundle after a benign additive column change — bundle build succeeds, version bumps, manifest updates. Rebuild after a breaking rename — bundle build fails with a clear error.
- [ ] **Cold-cache range-request test:** DuckDB query against a Release asset URL with a fresh browser cache fetches only the bytes the query needs. (Inspect Network tab; verify range requests fire.)
- [ ] **Internal cross-variant query test:** at least three canonical questions ("variants with attn FA committing after grokking", "variants with homeless_fraction > 0.2", "circularity at epoch 5000 ranked across variants") expressible as one-line SQL against the Parquet warehouse.

---

## Constraints

**Must:**
- Long-format canonical, with explicit `group_type` and `operation_type` discriminator columns. The schema decision is structural; wide-format is a consumer-side pivot.
- Coexistence of Parquet and `.npz`. Parquet for transform outputs; `.npz` for extract-only analyzers (`parameter_snapshot`, activation captures). Per-analyzer decision is documented in the analyzer's spec.
- Internal Parquet under `results/`, gitignored. Never committed to the repo.
- Published Parquet attached as GitHub Release assets, never committed to the repo. The repo stays small; data lives at release URLs.
- Bundle manifests required for every published bundle. A bundle without a manifest is not a bundle.
- Variant column always present in long-format tables. Cross-variant analysis is a primary use case.
- DuckDB-WASM integration in fieldnotes is lazy-loaded. Articles that don't query data don't pay the WASM cost.
- Internal-vs-published distinction is structural. Internal changes never silently become published; publication requires the explicit build script + Release workflow.

**Must avoid:**
- Replacing `.npz` blanket. Some analyzers genuinely have no transform output to tabulate.
- Committing Parquet files to the repo.
- Publishing every DataView in every bundle. Bundles are per-article and curated.
- Re-implementing Parquet reads in `miscope`. DuckDB / pyarrow handle this; the library composes existing tools.
- Hidden conversions inside renderers. Renderers receive long-format and pivot explicitly if they need wide.
- Loading every variant's DataFrame into memory pre-emptively. Long format scales fine but only if consumers respect filtered loads.
- Coupling DuckDB-WASM integration to a specific MDX framework beyond Astro. Cross-framework portability is not a goal.
- Requiring authentication for read access. Published bundles are public.

**Flexible:**
- Compression algorithm. Default zstd for published bundles (file size); snappy or no compression for internal warehouse (write speed).
- Whether the build script is a Python module (`miscope.publish.build`) or a standalone `scripts/build_publication_bundle.py`. Either works.
- Manifest format (JSON vs YAML). Default JSON for tooling compatibility.
- Whether `<DuckDBQuery>` is built inline in fieldnotes or extracted to a reusable component. Default inline; extract on real reuse.
- Whether multi-output shape characterizations use a packed `parameters` array column or a sibling table with explicit columns. Decide per characterization.

---

## Architecture Notes

### Why long-format with discriminator columns

The natural coordinate system from REQ_109's primitives is multi-dimensional: `(variant, epoch, site, frequency)`, `(variant, epoch, group, pc_index)`, `(variant, epoch, group, characterization)`. Long format makes this explicit; wide format would force an arbitrary hierarchy choice (is `site` a column? Or a row?).

The `group_type` and `operation_type` discriminators are what make the long format query-friendly. Without them, every consumer parses the `group` string or the table name to know what kind of object they're looking at. With them, `WHERE group_type = 'activation_site'` is the filter.

The trade-off: discriminator columns add storage redundancy (the same `group_type` value repeats for every row in a homogeneous query). Parquet's columnar compression makes this near-free.

### Why the Parquet/`.npz` split (and not blanket replacement)

The user's framing during the discussion that produced this REQ:

> "There may be some analyzers where we will always want to output `.npz` files. The `parameter_snapshot` and activation analyzers might not be doing any transform, but they are pulling weights and activations into a data format that *can* be transformed without loading model checkpoints."

`.npz` is the right format for **raw tensor caches**: dense, multi-dimensional, no natural tabular flattening. Parquet is the right format for **derived measurements**: typed, queryable, cross-variant joinable. The split honors the ETL framing — extract outputs are caches, transform outputs are tables.

Treating one as a replacement for the other would force `.npz` into a format it doesn't fit (transform outputs lose query-ability) or force Parquet into a format it doesn't fit (raw tensors become awkward when serialized as long-format rows).

### Why GitHub Releases (not the repo, not HF Hub)

Repo-as-host fails on bloat: even modest Parquet files accumulated across articles inflate clone size for every contributor. Git LFS adds friction and quota concerns.

HF Hub was the original publication framing (REQ_100). The reframe in REQ_108 moved publication to GitHub Pages + Releases for two reasons: (1) reviewer-trust is better served by inline-queryable Parquet on a stable URL than by full-platform install + raw-artifact download; (2) the moat-for-platform-evolution argument — published bundles are immutable, internal warehouse can churn freely. HF Hub re-activates if raw-artifact publication demand surfaces (REQ_100 stays deferred until then).

GitHub Releases supplies what the publication workflow needs: stable URLs per asset, immutability per tag, range-request support via Fastly CDN, permissive CORS for browser fetches from GitHub Pages.

### Why per-article bundles

A single rolling dataset (one Parquet bundle that grows over time) was rejected during discovery for two reasons:

1. **Schema stability.** A rolling dataset must absorb every schema change ever made. Bundles per-article freeze schemas at publication; the cost of a breaking change is contained to a new bundle version.
2. **Review focus.** A reviewer of a ring-geometry article wants the ring-geometry data, not every metric ever computed. Curation tightens the review surface.

Cost: redundancy across bundles when articles share data. Acceptable — Parquet compresses well, Releases storage is generous.

### DuckLake (held in reserve)

DuckLake is a DuckDB-native lakehouse format that adds catalog and registry management on top of Parquet. We are not committing to a DuckLake implementation in this REQ. The schema design (long-format with discriminators, manifest-tracked bundles, per-DataView Parquet files) is compatible with DuckLake should we adopt it later.

The risk to monitor: if `miscope` ends up building substantial registry / catalog infrastructure (REQ_107 territory), check whether DuckLake already does that work. Reinventing a registry product would be a poor use of effort.

### Composition with REQ_106 and REQ_107

- **REQ_106** establishes that DataViews are first-class peers to artifacts, with declared schemas and version-keyed cache invalidation. This REQ takes that DataView output and persists it as Parquet.
- **REQ_107** registry enumerates DataViews and their schemas. The publication build script reads from the registry to find DataView outputs by name; the bundle manifest cross-references registry entries.
- Together: REQ_106 names the contract, REQ_107 makes the contracts discoverable, REQ_110 persists and publishes the contract's outputs.

### Composition with REQ_109

This REQ depends on REQ_109 being substantially in place. The reason: tabular tables are flattenings of typed measurement results. `pca_results` is a flatten of `PCAResult` over the natural dimension columns; `frequency_spectrum` is a flatten of `FrequencySpectrum`; `shape_characterizations` is a flatten of the characterization-result types. If the upstream types churn, the table schemas churn. Stabilizing primitives first lets schemas stabilize cheaply.

---

## Notes

- **Sequencing.** REQ_109 (Measurement Primitives) lands first; REQ_110 lands once primitives are stable enough that table schemas can freeze. The user explicitly framed the asymmetry: "tackle cleaning up what we have first."
- **First published bundle candidate.** The ring-geometry-at-epoch-500 finding or the second-descent-as-destabilizing finding — claims that push against literature and most benefit from inspectable backing data.
- **DuckDB-WASM novelty.** Reader-side queryable findings, zero install, zero account, is genuinely novel for research publication. Pairs cleanly with the Astro/MDX infrastructure already in fieldnotes.
- **Citation surface.** A published bundle URL + manifest + content hash is the citable unit. Future Zenodo DOI integration is out of scope for v1.0 but contemplated; the manifest carries the metadata Zenodo would need.
- **REQ_107 stays adjacent, not merged.** A registry answers "what tables/analyzers/DataViews exist and what's their schema." This REQ answers "where's the data and how do I query it." Coupled (the registry probably reads Parquet schemas) but distinct enough that merging risks the registry getting blocked on the storage rewrite.
- **Backward-compatibility window.** Existing wide-format consumers (e.g., `get_parameter_trajectory_data_frame` in notebook prototypes) update to call `to_wide()` rather than re-implementing the pivot. The wide DataFrames in current notebooks are exploratory and remain fine; the canonical published shape is long.
