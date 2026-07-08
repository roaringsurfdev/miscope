# REQ_108: Publication Surface (Parquet + DuckDB + GitHub Releases + DuckDB-WASM)

**Status:** Superseded by REQ_110 (Lakehouse Surface)
**Priority:** High — companion to REQ_103; the data half of the v1.0 publication story.
**Branch:** TBD
**Dependencies:** REQ_106 (DataView first-class status, layering principle); REQ_101 (long-format DataFrame convention is the upstream shape that gets persisted as Parquet); REQ_103 (the package + repo + docs surfaces this REQ supplies data to). REQ_098 and REQ_097 must be substantially in place because the published Parquet bundles materialize their outputs.
**Attribution:** Engineering Claude (under user direction)

> **Superseded by REQ_110 (Lakehouse Surface).** This REQ's scope (Parquet as persisted DataView form, DuckDB as cross-variant query engine, GitHub Releases as publication host, DuckDB-WASM in fieldnotes for inline queries, internal-vs-published bundle distinction) was consolidated with REQ_101 (DataFrame Support) into a single tabular-surface REQ. The "moat for platform evolution" framing, the per-article bundle decision, the schema-stability rules, and the GitHub-Releases-vs-HF-Hub reasoning all carry forward into REQ_110 verbatim. Implementation tracks under REQ_110.

---

## Problem Statement

A finding only matters if a reviewer can trust it. For an independent researcher
publishing claims that push against existing literature — *ring geometry
appearing as early as epoch 500*, *the second descent window as a destabilizing
event rather than a consolidating one* — trust requires more than a chart. It
requires inspectable data.

The conventional path (publish raw artifacts to HuggingFace, expect reviewers
to install a platform and re-run the analysis) has two problems:

1. **Friction.** "Install MIScope, download 30 variant directories, run a
   notebook" is a high cost for a reviewer who just wants to verify a single
   claim from one article.
2. **Mismatch of shape.** Raw artifacts (per-epoch `.npz` files of weight
   matrices and activations) are not what review wants to see. Review wants
   the *derived* metrics that a finding is built on, in a queryable form.

The opportunity, surfaced during the discussion that produced this REQ:
**publish derived views as Parquet, query them with DuckDB.** Parquet is
self-describing, columnar, language-agnostic, and HTTP-range-readable. DuckDB
is in-process, single-binary, runs in the browser via WASM, and queries Parquet
over HTTP without downloading the whole file. Together they make a published
finding *inspectable* — a reviewer can run a SQL query against the data
underlying the chart, in their browser, with zero install.

This REQ defines the publication surface for derived views. It is the data
half of the v1.0 publication story; REQ_103 is the package + repo + docs half.

A second motivation, called out by the user during the discussion: the
publication-frozen Parquet snapshot creates a *moat for the platform's
evolution.* Because published artifacts are immutable and self-contained,
the platform's internal storage, schemas, and analyzer outputs can continue
to evolve without breaking citations. Internal warehouse and published
dataset are deliberately decoupled.

---

## Conditions of Satisfaction

### Parquet as the persisted form for DataView outputs

- [ ] DataView materializations write Parquet files to disk as their canonical
  persisted form. The in-memory pandas DataFrame (REQ_101) is the consumer
  surface; the Parquet file is the durable storage form.
- [ ] Internal Parquet writes go under `results/{family}/{variant}/dataviews/{view_name}.parquet`
  (or a sibling layout — final path canonicalized in implementation). Internal
  Parquet is gitignored alongside `.npz` artifacts; deterministically
  regeneratable.
- [ ] DataView Parquet schemas honor REQ_101's long-format convention:
  explicit dimension columns (`variant`, `epoch`, `site`, etc.) and value
  columns. The Parquet schema is the on-disk codification of REQ_101's
  in-memory schema.
- [ ] Parquet files written via `pyarrow` (already a `miscope` dep per
  REQ_103) with column compression (snappy or zstd — pick one in
  implementation; zstd preferred for publication file size).

### DuckDB as the cross-variant query engine

- [ ] DuckDB available as a `miscope` dep (REQ_103 lists it). Used by:
  - The library — for ergonomic cross-variant queries that REQ_101's
    `pd.concat` approach handles awkwardly at scale.
  - Researchers — direct query access against published Parquet bundles
    without needing to load every variant's DataFrame.
- [ ] A small `miscope.publish.query` (or sibling) helper exposes a
  DuckDB connection bound to a chosen Parquet root (local path or URL).
  Researchers call:
  ```python
  con = miscope.publish.query("https://github.com/.../releases/download/data-v1.0/")
  df = con.sql("SELECT variant, epoch, magnitude FROM 'frequency_spectrum' WHERE site='mlp'").df()
  ```
- [ ] DuckDB read paths support both local Parquet (working session) and
  HTTP-hosted Parquet (reviewing a published finding). The same query
  surface works for both.
- [ ] Cross-Parquet-file joins (e.g., joining `frequency_spectrum` against
  `repr_geometry_summary` on `(variant, epoch)`) work via DuckDB's native
  Parquet handling; no custom join logic in `miscope`.

### Internal warehouse vs. published bundle distinction

- [ ] **Internal warehouse:** `results/.../dataviews/*.parquet`. Written by
  analyzers / DataView materializations during normal pipeline runs. Free
  to evolve, regeneratable, gitignored. Schema can change with code; mismatch
  is a re-run.
- [ ] **Published bundle:** a curated subset of Parquet files, frozen at a
  point in time, with a stable schema, attached to a `data-*` GitHub Release.
  Once published, a bundle is immutable. Schema changes require a new
  bundle version with a new Release tag.
- [ ] **Bundle composition is per-article, not all-of-everything.** A
  fieldnotes article on ring geometry publishes only the Parquet files that
  underlie its claims. Avoiding "publish the kitchen sink" keeps bundles
  small, schemas tight, and review focused.
- [ ] **Curation step is explicit.** A `scripts/build_publication_bundle.py`
  (or sibling — name canonicalized in implementation) selects DataView
  outputs by name, freezes their schemas, writes them to a versioned
  output directory, computes content hashes, and emits a manifest.
- [ ] **Bundle manifest** (`manifest.json` or `manifest.yaml`) lives inside
  the bundle and declares: `bundle_version`, `mint_date`, `miscope_version`,
  list of included Parquet files with their schemas and content hashes,
  the article(s) that reference the bundle, and a brief description.
  Reviewers and citation systems read the manifest to know what they're
  looking at.

### GitHub Releases as the publication host

- [ ] Published bundles attach to GitHub Releases tagged with a `data-*`
  prefix (e.g., `data-v1.0-ring-geometry`, `data-v1.1-saddle-transport`).
  Distinct prefix from package release tags (`v1.0.0` for `miscope`
  itself) so the two release streams don't collide.
- [ ] Per-bundle Parquet files attached as Release assets. Stable URLs
  (`github.com/.../releases/download/data-v1.0-ring-geometry/frequency_spectrum.parquet`)
  are what fieldnotes articles reference and what DuckDB-WASM queries hit.
- [ ] GitHub Action workflow (`.github/workflows/data-release.yml`) automates:
  produce bundle via the build script → create Release with tag → attach
  Parquet files and manifest → publish.
- [ ] Range-request support verified — a DuckDB query against a Release
  asset URL fetches only the bytes it needs. (GitHub Releases serve via
  Fastly CDN, which supports range requests; this is verified, not assumed.)
- [ ] Release notes for `data-*` releases include: what article(s) the
  bundle supports, the schema-stable claim, the bundle manifest, and a
  link to the corresponding fieldnotes article.

### DuckDB-WASM in fieldnotes (inline queries)

- [ ] Fieldnotes Astro site loads DuckDB-WASM as a client-side module.
  Lazy-loaded on demand (the bundle is ~5 MB; not all articles need it).
- [ ] An MDX component — `<DuckDBQuery>` or similar — accepts a Parquet
  URL (the GitHub Releases asset) and a SQL query string. Renders a result
  table inline. Reader can edit the query and re-run.
- [ ] Articles use the component to back specific claims. Example MDX usage:
  ```mdx
  <DuckDBQuery
    parquet="https://github.com/.../releases/download/data-v1.0-ring-geometry/repr_geometry.parquet"
    query="SELECT variant, epoch, circularity FROM data WHERE epoch BETWEEN 400 AND 600 ORDER BY circularity DESC LIMIT 10"
  />
  ```
- [ ] The component handles loading state, query errors, and result-table
  rendering. Errors surface clearly to the reader; a broken query in a
  published article is recoverable by re-editing in-browser.
- [ ] CORS verified: GitHub Releases asset URLs return CORS headers
  permissive enough for browser fetches from the GitHub Pages domain.
  (Verified, not assumed.)

### Schema stability for published bundles

- [ ] **Published Parquet schemas are immutable per bundle version.** A
  column added or renamed requires a new bundle version with a new Release
  tag. Old bundles remain accessible at their original tags.
- [ ] **Schema declaration co-located with the DataView definition** (per
  REQ_106 / REQ_107). The bundle build script reads the declared schema and
  fails to publish if the materialized Parquet doesn't match.
- [ ] **Bundle manifest records schemas explicitly** so a reviewer reading
  the manifest can see what columns exist, their dtypes, and their
  semantic descriptions — without needing to open the Parquet file.
- [ ] **Citation surface:** a published bundle URL + manifest + content
  hash is the citable unit. Future Zenodo DOI integration is out of scope
  for v1.0 but contemplated; the manifest carries the metadata Zenodo
  would need.

### Validation

- [ ] End-to-end: build a publication bundle from real DataView outputs →
  attach to a GitHub Release → query a Parquet asset via DuckDB-WASM from
  a fieldnotes article in the deployed GitHub Pages site → reader sees
  result table inline. This pipeline operational is the v1.0 acceptance
  bar (and the REQ_103 validation criterion that depends on this REQ).
- [ ] Schema drift test: rebuild a bundle after a benign schema change
  (additive column) — bundle build succeeds, version bumps, manifest
  updates. Rebuild after a breaking change (column rename) — bundle build
  fails with a clear error pointing at the schema mismatch.
- [ ] Cold-cache range-request test: DuckDB query against a Release asset
  URL with a fresh browser cache fetches only the bytes the query needs,
  not the full file. (Inspect Network tab; verify range requests fire.)

---

## Constraints

**Must:**
- Internal Parquet under `results/`, gitignored. Never committed to the repo.
- Published Parquet attached as GitHub Release assets, never committed to the
  repo. The repo stays small; the data lives at release URLs.
- Bundle manifests are required for every published bundle. A bundle without
  a manifest is not a bundle, it's a loose pile of Parquet files.
- DuckDB-WASM integration in fieldnotes is lazy-loaded. Articles that don't
  query data don't pay the WASM cost.
- The internal-vs-published distinction is structural, not nominal. Internal
  changes never silently become published; publication requires the explicit
  build script + Release workflow.

**Must avoid:**
- Committing Parquet files to the repo. Bloat is the failure mode the
  Releases-as-host pattern was chosen to avoid.
- Publishing every DataView in every bundle. Bundles are per-article and
  curated.
- Re-implementing Parquet reads in `miscope` when DuckDB / pyarrow already
  do this. The library composes existing tools; it does not invent
  storage formats.
- Coupling DuckDB-WASM integration to a specific MDX component framework
  beyond what fieldnotes already uses (Astro). Cross-framework portability
  is not a goal.
- Requiring authentication for read access. Published bundles are public;
  GitHub Releases serves them without auth.

**Flexible:**
- Compression algorithm (snappy vs zstd vs uncompressed). Default zstd
  for published bundles (tighter file size); snappy or no compression for
  internal warehouse (faster writes during pipeline runs).
- DuckDB-WASM bundle format / size optimization. Default: load the
  standard build; optimize if WASM size becomes an article-load problem.
- Whether the build script is a Python module (`miscope.publish.build`) or
  a standalone `scripts/build_publication_bundle.py`. Either works.
- Manifest format (JSON vs YAML). Default JSON for tooling compatibility.
- Whether the `<DuckDBQuery>` MDX component is built inline in the
  fieldnotes site or extracted to a small reusable Astro component package.
  Default: inline component in fieldnotes; extract only if a real reuse
  case appears.

---

## Architecture Notes

### Why Parquet + DuckDB specifically

The "small-scale lakehouse" stack — Parquet files + DuckDB query engine — is
the right shape for research-scale tabular data:

- **Parquet is portable.** Self-describing schema, language-agnostic
  (Python, R, Julia, JavaScript, Rust all read it), columnar, compressible.
  A reviewer in any language ecosystem can use the published data.
- **DuckDB is in-process.** No daemon, no server, no install. Single binary
  Python wheel; single WASM module in browsers. The cost-of-entry for a
  reviewer is `pip install duckdb` (or nothing, if they query in-browser).
- **HTTP range requests work.** DuckDB issues range requests for Parquet
  reads, fetching only the columns and row groups its query touches. A 100 MB
  Parquet file is a 50 KB query, not a 100 MB download.
- **WASM in the browser is the unlock.** Reader-side queryable findings,
  zero install, zero account. The fieldnotes site stops being "trust me,
  here's a chart" and becomes "here's the data; here's a query you can
  change."

The lakehouse pattern's heavier components (Delta Lake, Iceberg, Spark) are
not needed at this scale and would be infrastructure ahead of demand.

### Why "internal vs. published" matters as much as it does

The user named the principle directly during the discussion that produced
this REQ:

> "I'm really happy with the direction this is taking, [...] this allows the
> platform to evolve while publishing a stable data artifact."

A published Parquet bundle is a **moat for platform evolution.** The internal
warehouse can churn — schemas change, DataView projections improve, analyzers
evolve — without breaking external citations. Reviewers verify against
frozen bundles; researchers iterate on the live system.

If the two were the same thing, every refactor would risk breaking a
published claim. That's the failure mode the distinction prevents.

### Why GitHub Releases, not the repo

Three considerations:

1. **Repo bloat.** Even modest Parquet files (KB to single-digit MB per
   DataView), accumulated across articles, would inflate clone size for
   every contributor over time. Releases live outside the git history.
2. **Versioning is built in.** Each Release has a stable tag; assets at
   that tag are immutable. Citations point at a tag, not a moving HEAD.
3. **Range-request support and CORS.** GitHub Releases serves through
   Fastly with permissive CORS — exactly what DuckDB-WASM needs to query
   from the GitHub Pages fieldnotes site.

Alternative paths considered: committing to `fieldnotes/public/data/`
(simple but bloats the repo over time; Git LFS adds contributor friction),
splitting fieldnotes to its own repo (was explored in the discussion thread;
rejected because writing-where-the-work-happens is a design intent and
splitting would force context-switching). Releases is the cleanest path.

### Why HF Hub is not on the v1.0 critical path

Original framing assumed HuggingFace Hub as the publication channel. The
reframe in this REQ moves derived-view publication to GitHub Pages +
Releases, which is sufficient for the reviewer-trust use case and avoids
committing to HF infrastructure when no demand has yet surfaced.

HF earns its place later if any of the following become true:
- A reviewer asks for raw `.npz` artifacts (not just derived metrics)
- Bundle sizes exceed practical Releases limits
- HF dataset-card discoverability becomes valuable as a community surface

Until then, REQ_100 (HFHubSource) stays deferred.

### Why per-article bundles, not a single rolling dataset

A single rolling dataset (one Parquet bundle that grows over time) was
rejected for two reasons:

1. **Schema stability.** A rolling dataset must absorb every schema change
   ever made; bundles per-article freeze schemas at the moment of
   publication. The cost of a breaking change is contained.
2. **Review focus.** A reviewer of a ring-geometry article wants the
   ring-geometry data, not every metric ever computed. Curation tightens
   the review surface.

The cost: redundancy across bundles when articles share data. Acceptable —
Parquet compresses well, and Releases storage is generous.

### How this composes with REQ_106 and REQ_107

- **REQ_106** establishes that DataViews are first-class peers to artifacts,
  with declared schemas and version-keyed cache invalidation. This REQ
  takes that DataView output and persists it as Parquet.
- **REQ_107** registry enumerates DataViews. The publication build script
  reads from the registry to find DataView outputs by name; the bundle
  manifest cross-references registry entries.
- **REQ_103** is the package + repo + docs surface. This REQ is the data
  surface. Together they constitute the v1.0 publication story.

---

## Notes

- The discussion thread that produced this REQ surfaced the publication-frame
  motivation that earlier REQs hadn't named. REQ_106 already committed to
  the "hybrid lakehouse pattern" architecturally; this REQ commits to the
  format (Parquet) and engine (DuckDB) and adds the publication primitive.
- The DuckDB-WASM-in-fieldnotes integration is genuinely novel for
  research publication and ties directly into the Astro/MDX infrastructure
  already in place for fieldnotes. No additional deployment surface.
- Citation strategy beyond Releases (Zenodo DOIs, persistent identifiers)
  is contemplated but out of scope for v1.0. The bundle manifest carries
  the metadata Zenodo would need; integration deferred until a published
  article actually wants a DOI.
- This REQ depends on REQ_101 being substantially in place — the long-format
  DataFrame convention is the upstream shape. If REQ_101 lands first, this
  REQ adds the persistence + publication layer on top.
- The first published bundle should probably be the one supporting the
  ring-geometry-at-epoch-500 finding or the second-descent-as-destabilizing
  finding — claims that push against literature and most benefit from
  inspectable backing data.
