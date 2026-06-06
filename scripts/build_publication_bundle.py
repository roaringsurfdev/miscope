#!/usr/bin/env python
"""Build (and optionally publish) a publication data bundle from a spec (REQ_110E).

The warehouse data is local-only, so a bundle is built on the machine that has the
data. This script reads a declarative spec, freezes the selected warehouse tables to
flat zstd Parquet + a manifest, runs the schema-stability gate against the previously
published manifest, and — with ``--release`` — attaches the assets to a ``data-*``
GitHub Release via ``gh``.

Layout convention (co-located with the article a bundle backs)::

    apps/fieldnotes/bundles/
        {name}.toml                  # the spec (committed)
        {name}/manifest-{ver}.json   # published manifest history (committed; the gate baseline)
        _dist/{name}/{ver}/          # built Parquet + manifest (gitignored)

Examples::

    # build only, gate against the latest committed manifest, inspect the result
    uv run python scripts/build_publication_bundle.py apps/fieldnotes/bundles/ring-geometry.toml

    # build, record the manifest into the committed history, and cut the Release
    uv run python scripts/build_publication_bundle.py apps/fieldnotes/bundles/ring-geometry.toml \
        --update-history --release
"""

from __future__ import annotations

import argparse
from pathlib import Path

from miscope.publish import BundleManifest, BundleSpec, build_bundle_for_family, create_release
from miscope.publish.schema_gate import SchemaChange


def _history_dir(spec_path: Path, spec: BundleSpec) -> Path:
    """Committed manifest-history directory for a bundle (``{spec_dir}/{name}/``)."""
    return spec_path.parent / spec.name


def _version_key(version: str) -> tuple:
    """Numeric-aware sort key so ``v1.10`` orders after ``v1.9``."""
    stem = version.lstrip("vV")
    return tuple(int(p) if p.isdigit() else p for p in stem.split("."))


def _latest_baseline(spec_path: Path, spec: BundleSpec) -> BundleManifest | None:
    """The most recent committed manifest for this bundle, excluding the current version."""
    hist = _history_dir(spec_path, spec)
    if not hist.is_dir():
        return None
    candidates = [BundleManifest.read(p) for p in hist.glob("manifest-*.json")]
    prior = [m for m in candidates if m.bundle_version != spec.version]
    if not prior:
        return None
    return max(prior, key=lambda m: _version_key(m.bundle_version))


def _default_out(spec_path: Path, spec: BundleSpec) -> Path:
    return spec_path.parent / "_dist" / spec.name / spec.version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Path to the bundle spec TOML")
    parser.add_argument("--out", type=Path, help="Build directory (default: _dist/{name}/{ver})")
    parser.add_argument(
        "--no-gate", action="store_true", help="Skip the schema-stability gate (first publication)"
    )
    parser.add_argument(
        "--update-history",
        action="store_true",
        help="Copy the built manifest into the committed history dir",
    )
    parser.add_argument("--release", action="store_true", help="Publish a data-* Release via gh")
    parser.add_argument("--draft", action="store_true", help="Create the Release as a draft")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --release, print the gh command but do not run it",
    )
    args = parser.parse_args()

    spec = BundleSpec.from_toml(args.spec)
    out_dir = args.out or _default_out(args.spec, spec)
    baseline = None if args.no_gate else _latest_baseline(args.spec, spec)

    result = build_bundle_for_family(spec, out_dir, baseline=baseline)
    _report(result, baseline)

    if args.update_history:
        _record_history(args.spec, result.manifest)
    if args.release:
        argv = create_release(result, draft=args.draft, dry_run=args.dry_run)
        verb = "would run" if args.dry_run else "ran"
        print(
            f"\n[release] {verb}: {' '.join(argv[:6])} … (+{len(result.parquet_paths) + 1} assets)"
        )


def _report(result, baseline: BundleManifest | None) -> None:
    m = result.manifest
    print(f"[{m.tag}] {len(m.tables)} tables -> {result.out_dir}")
    for t in m.tables:
        prov = f"  prov={list(t.populated_provenance)}" if t.populated_provenance else ""
        print(f"    {t.name:34} {t.rows:>10,} rows  {t.content_hash[:19]}…{prov}")
    if result.gate is not None and baseline is not None:
        change = result.gate.change
        note = "" if change is SchemaChange.IDENTICAL else f" vs {baseline.bundle_version}"
        print(f"    schema gate: {change.value}{note}")
    elif baseline is None:
        print("    schema gate: skipped (no baseline)")


def _record_history(spec_path: Path, manifest: BundleManifest) -> None:
    hist = spec_path.parent / manifest.bundle_name
    hist.mkdir(parents=True, exist_ok=True)
    dest = hist / f"manifest-{manifest.bundle_version}.json"
    manifest.write(dest)
    print(f"    history updated: {dest}")


if __name__ == "__main__":
    main()
