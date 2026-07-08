#!/usr/bin/env python
"""Validate committed bundle manifest histories (REQ_110E CI gate).

Parquet never enters the repo, but bundle manifests do — and they carry the
captured schema, so they are the data-free baseline the schema-stability gate reads
in CI. This script walks every bundle history under a root, fails on structural
integrity problems (a manifest whose version disagrees with its filename, a
duplicated version), and prints the latest schema delta per bundle as information (a
*new* bundle version is permitted to change schema; the gate that refuses a silent
breaking re-publish runs at mint time in the build script).

    uv run python scripts/validate_bundle_history.py apps/fieldnotes/bundles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from miscope.publish.history import load_history, schema_delta, validate_history
from miscope.publish.schema_gate import SchemaChange


def _bundle_dirs(root: Path) -> list[Path]:
    """Directories under ``root`` that hold a manifest history (``manifest-*.json``)."""
    return sorted({p.parent for p in root.glob("*/manifest-*.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("apps/fieldnotes/bundles"),
        help="Bundles root (default: apps/fieldnotes/bundles)",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"[bundles] no bundles root at {args.root} — nothing to validate")
        return

    bundle_dirs = _bundle_dirs(args.root)
    if not bundle_dirs:
        print(f"[bundles] no manifest histories under {args.root} — nothing to validate")
        return

    problems: list[str] = []
    for bdir in bundle_dirs:
        history = load_history(bdir)
        issues = validate_history(bdir)
        problems += [f"{bdir.name}: {p}" for p in issues]
        versions = ", ".join(m.bundle_version for m in history)
        print(f"[{bdir.name}] {len(history)} version(s): {versions}")
        _report_delta(bdir)

    if problems:
        print("\nMANIFEST INTEGRITY PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nAll bundle manifest histories valid.")


def _report_delta(bundle_dir: Path) -> None:
    delta = schema_delta(bundle_dir)
    if delta is None:
        return
    if delta.change is SchemaChange.IDENTICAL:
        print("    latest delta: identical schema")
        return
    detail = "; ".join(d.describe() for d in delta.deltas if d.change is not SchemaChange.IDENTICAL)
    extra = []
    if delta.new_tables:
        extra.append(f"new tables {list(delta.new_tables)}")
    if delta.dropped_tables:
        extra.append(f"dropped tables {list(delta.dropped_tables)}")
    print(f"    latest delta: {delta.change.value} — {'; '.join(filter(None, [detail, *extra]))}")


if __name__ == "__main__":
    main()
