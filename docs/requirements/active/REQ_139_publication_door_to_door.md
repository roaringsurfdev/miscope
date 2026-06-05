# REQ_139: First Door-to-Door Publication (the REQ_110 live acceptance bar)

**Status:** Draft (stub — checklist to execute when the first bundle ships)
**Priority:** Medium — gated on an external push (the kinomorphic.com / fieldnotes launch); not in-flight day-to-day.
**Branch:** TBD
**Dependencies:** REQ_110 (the entire lakehouse surface — 110-A…F are implemented; this REQ is the one CoS they deliberately deferred). REQ_103 (package/repo/docs surface). The fieldnotes deploy workflow (`.github/workflows/deploy-fieldnotes.yml`).
**Attribution:** Engineering Claude (under user direction)

---

## Why this exists

REQ_110 built the full publication chain and verified every link **in isolation**:
the build script freezes a bundle, the schema gate refuses a silent break,
`create_release` cuts a `data-*` Release, `<DuckDBQuery>` lazy-loads DuckDB-WASM
and range-reads remote Parquet. What it did **not** do — by design — is run the
*live assembly*: publish a real bundle, deploy an article that queries it, and watch
a cold-cache browser fetch only the bytes the query touches. That live run is the
v1.0 acceptance bar for REQ_110, and it is human-gated because it requires
publishing to the public, a Pages deploy, and (likely) the new public domain.

This REQ is the **checklist** for that first door-to-door run. Closing it closes the
last open CoS of REQ_110. It is deliberately small: the machinery exists; this is
operating it once, end to end, and recording what breaks.

This is a natural thing to **bundle with the public launch** (the kinomorphic.com
site that will host the fieldnotes), since the publication surface only becomes
real once the site is public.

---

## Prerequisite decisions (resolve before executing)

- [ ] **Public repo + org/owner name.** The build/release tooling and the
  `<DuckDBQuery>` demo currently use the placeholder `roaringsurfdev/MIScope`. Pin
  the real public repo the `data-*` Releases attach to, and update the example URLs
  (`apps/fieldnotes/bundles/README.md`, `src/pages/duckdb-demo.astro`).
- [ ] **Site URL / custom domain.** `astro.config.mjs` is set to
  `https://roaringsurfdev.github.io` + base `/miscope`. If fieldnotes moves to
  `kinomorphic.com`, update `site`/`base` and the Pages custom-domain config, and
  re-confirm asset/base-URL resolution in the built pages.
- [ ] **Pages enabled.** Repo Settings → Pages → Source → GitHub Actions (the
  fieldnotes deploy workflow already targets Pages).
- [ ] **Which finding ships first.** REQ_110 names candidates: ring-geometry-at-
  epoch-500, or second-descent-as-destabilizing — claims that push against the
  literature and most benefit from inspectable backing data.

---

## Conditions of Satisfaction (the door-to-door run)

### 1. Author the bundle

- [ ] Write the `BundleSpec` TOML at `apps/fieldnotes/bundles/{name}.toml`, selecting
  only tables that **actually materialize** in the family warehouse (audit first —
  e.g. `shape_characterizations`, `neuron_frequency_attribution`, `pca_results`,
  `variant_outcomes` exist; `frequency_spectrum` did **not** in `modulo_addition_1layer`
  as of 2026-06-05). Curate to the claim (per-article, not kitchen-sink).
- [ ] Materialize the warehouse for the variants the claim covers
  (`scripts/materialize_warehouse.py {family} --all` or per-variant), restricted to
  the verified baselines until REQ_137's variant refresh lands.

### 2. Build + gate + publish

- [ ] `scripts/build_publication_bundle.py {spec} --update-history` — build locally,
  inspect the manifest (row counts, content hashes, captured schema), commit the
  manifest history JSON (Parquet stays gitignored).
- [ ] `--release` (or `--release --draft` first) — cut the `data-*` Release via `gh`;
  confirm the Parquet + `manifest.json` attach as assets with stable URLs.
- [ ] Confirm the CI manifest gate (`data-release.yml`) passes on the spec/manifest PR.

### 3. Wire the article

- [ ] Add `<DuckDBQuery>` to the article (MDX), pointing `url=`/`tables=` at the live
  Release asset URL(s), with a starter query that backs a specific claim.
- [ ] Verify locally (`npm run dev`) the box loads the engine on first Run, renders
  the result, and recovers from a deliberately broken query (edit + re-run).

### 4. Deploy + verify live (the actual acceptance bar)

- [ ] Deploy the site (Pages, via the fieldnotes workflow) and load the published
  article.
- [ ] **Cold-cache range-request check:** fresh browser cache → run the inline query
  → in the Network tab confirm the Parquet fetch issues HTTP **range** requests and
  transfers far less than the full file.
- [ ] **CORS check:** the fetch from the Pages/custom domain to the Release-asset CDN
  succeeds (no CORS error in console). Cross-check with the Python-side probe
  `miscope.publish.verify_bundle_url(<release base url>, tables=[...])`.
- [ ] **Reader-edit check:** edit the SQL in-browser, re-run, get a new result.

### 5. Closeout

- [ ] Flip REQ_110's deferred CoS to satisfied: "End-to-end pipeline operational"
  and "Cold-cache range-request test" (and the 110-E "Range-request and CORS
  verified" item), citing this run.
- [ ] Record any breakage/friction (CORS headers, range support, base-URL issues,
  bundle size) back here and in `docs/notes/`.

---

## Constraints

**Must:**
- Real data, real Release, deployed article — this REQ is specifically the *live*
  run; isolated/unit verification is already done in REQ_110.
- Parquet never committed; manifest history committed.
- The bundle is curated to the claim (per-article).

**Must avoid:**
- Re-deriving REQ_110's machinery. If a step needs new code, that is a REQ_110
  regression, not new scope here — fix it there.
- Shipping a bundle over tables/variants known to be stale (respect the
  baselines-only rule until REQ_137).

**Flexible:**
- Which finding ships first; whether the first Release is cut as a draft then
  promoted; whether deploy targets `*.github.io` or `kinomorphic.com` first.

---

## Notes

- **Placement.** Filed in `active/` as the named REQ_110 closeout; if the public
  launch slips, move to `future/`.
- **Citation surface.** Once live, the bundle URL + manifest + content hash is the
  citable unit (REQ_110 contemplates a later Zenodo DOI; out of scope here).
- **DuckDB-WASM novelty.** Reader-side queryable findings with zero install is the
  payoff this run demonstrates publicly for the first time.
