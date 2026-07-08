# Publication data bundles (REQ_110E)

A **published bundle** is the immutable, citable counterpart to the internal
warehouse: a per-article subset of warehouse tables, frozen as flat zstd Parquet,
content-hashed, manifest-bearing, and attached to a `data-*` GitHub Release. It lets
a reviewer verify a fieldnotes claim by querying the *derived metrics* over a stable
URL — no platform install, no pipeline re-run. (Browser-side inline querying via
DuckDB-WASM is REQ_110F.)

## Layout

```
apps/fieldnotes/bundles/
  {name}.toml                  # the spec — what the bundle contains (committed)
  {name}/manifest-{ver}.json   # published manifest history (committed; the gate baseline)
  _dist/{name}/{ver}/          # built Parquet + manifest + NOTES.md (gitignored)
```

Parquet is **never committed** — it lives only in `_dist/` (local) and as Release
assets. Manifests are small JSON and *are* committed, because the schema-stability
gate compares a rebuild against the previously published manifest, and that lets the
gate run in CI without the data.

## Spec format

```toml
name = "ring-geometry"
version = "v1.0"
family = "modulo_addition_1layer"
description = "Derived metrics backing the ring-geometry-at-epoch-500 finding."
articles = ["ring-geometry"]

# Each table is a warehouse view, optionally subset/joined by a query. `name` is
# both the source view (when no query) and the published file stem ({name}.parquet).
[[tables]]
name = "shape_characterizations"
query = "SELECT * FROM shape_characterizations WHERE operation_type = 'circularity'"

[[tables]]
name = "frequency_spectrum"
```

## Build + publish (local — the data lives here)

```bash
# build only; gate against the latest committed manifest; inspect the result
uv run python scripts/build_publication_bundle.py apps/fieldnotes/bundles/ring-geometry.toml

# record the manifest into the committed history and cut the data-* Release via gh
uv run python scripts/build_publication_bundle.py apps/fieldnotes/bundles/ring-geometry.toml \
    --update-history --release
```

The build runs the **schema-stability gate**: an additive column change is allowed
but requires a bumped `version`; a breaking change (a removed or retyped column)
fails the build. A *deliberately* breaking new bundle is minted with `--no-gate` —
the explicit acknowledgment that you are re-authoring, not silently re-publishing.

After publishing, prove the bundle is range-request queryable over its Release URL:

```python
from miscope.publish import verify_bundle_url
verify_bundle_url("https://github.com/<org>/MIScope/releases/download/data-ring-geometry-v1.0",
                  tables=["shape_characterizations", "frequency_spectrum"])
```

## CI

`.github/workflows/data-release.yml` runs on PRs touching this directory: it
validates manifest integrity (filename/version agreement, no duplicate versions) and
reports the latest schema delta per bundle. It does **not** build bundles — that is
local, by construction.
