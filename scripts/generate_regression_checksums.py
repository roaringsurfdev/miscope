"""Generate reference checksums for regression testing.

Walks the artifacts directory for each reference variant and records the
SHA-256 hash of every ``.npz`` file (excluding non-deterministic and retired
analyzers — see ``regression_common.EXCLUDE_FROM_REGRESSION``). Output is written
to ``tests/regression/reference_checksums.json`` — the same path
``run_regression_check.py`` reads by default.

Run this on ``develop`` to establish the ground truth before a refactor; the
checker compares new outputs against these checksums.

Usage:
    uv run python scripts/generate_regression_checksums.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from regression_common import (
    EXCLUDE_FROM_REGRESSION,
    FAMILY,
    REFERENCE_CHECKSUMS_PATH,
    sha256_file,
)

from miscope.config import get_config

REFERENCE_VARIANTS = [
    # (prime, model_seed, data_seed, description)
    (113, 999, 598, "canon model"),
    (109, 485, 598, "fast clean grokker"),
    (101, 999, 598, "late grokker"),
    # (59, 485, 999, "no_second_descent (most degraded)"),
]


def checksum_variant(artifacts_dir: Path) -> list[dict]:
    records = []
    for npz_path in sorted(artifacts_dir.rglob("*.npz")):
        rel = npz_path.relative_to(artifacts_dir)
        top_dir = rel.parts[0] if rel.parts else ""
        if top_dir in EXCLUDE_FROM_REGRESSION:
            continue
        records.append(
            {
                "artifact_path": str(rel),
                "sha256": sha256_file(npz_path),
                "size_bytes": npz_path.stat().st_size,
            }
        )
        print(f"artifact checksum added: {rel}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Unified data root (default: from cfg.data_root).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REFERENCE_CHECKSUMS_PATH,
        help="Output path for checksums JSON",
    )
    args = parser.parse_args()
    data_root = args.data_root if args.data_root is not None else get_config().data_root

    output: dict = {"family": FAMILY, "variants": []}

    for prime, model_seed, data_seed, description in REFERENCE_VARIANTS:
        variant_name = f"p{prime}_seed{model_seed}_dseed{data_seed}"
        artifacts_dir = data_root / FAMILY / "variants" / variant_name / "artifacts"

        if not artifacts_dir.exists():
            print(f"  SKIP  {variant_name} — artifacts directory not found")
            continue

        records = checksum_variant(artifacts_dir)
        output["variants"].append(
            {
                "variant_id": variant_name,
                "prime": prime,
                "model_seed": model_seed,
                "data_seed": data_seed,
                "description": description,
                "artifact_count": len(records),
                "artifacts": records,
            }
        )
        print(f"  OK    {variant_name} — {len(records)} artifacts checksummed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nChecksums written to {args.output}")
    total = sum(v["artifact_count"] for v in output["variants"])
    print(f"Total: {len(output['variants'])} variants, {total} artifacts")


if __name__ == "__main__":
    main()
