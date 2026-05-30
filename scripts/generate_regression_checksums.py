"""Generate reference checksums for regression testing.

Walks the artifacts directory for each reference variant and records the
SHA-256 hash of every .npz file. Output is written to
tests/regression/reference_checksums.json.

Run this once before refactoring to establish the ground truth. The
run_regression_check.py script compares new outputs against these checksums.

Usage:
    uv run python scripts/generate_regression_checksums.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from miscope.config import get_config

REFERENCE_VARIANTS = [
    # (prime, model_seed, data_seed, description)
    (113, 999, 598, "canon model"),
    (109, 485, 598, "fast clean grokker"),
    (101, 999, 598, "late grokker"),
    # (59, 485, 999, "no_second_descent (most degraded)"),
]

FAMILY = "modulo_addition_1layer"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Analyzers excluded from regression: their output is intentionally non-deterministic.
# REQ_102 retirements — excluded so any stale on-disk artifacts from old results
# trees aren't picked up by the disk scan (these analyzers no longer exist).
DEPRECATED_ANALYZERS = {
    "effective_dimensionality",
    "centroid_dmd",
    "coarseness",
    "attention_freq",
    "attention_fourier",
    "neuron_fourier",
    "dominant_frequencies",
    "neuron_freq_norm",
}
EXCLUDED_ANALYZERS = DEPRECATED_ANALYZERS | {"landscape_flatness", "fourier_nucleation"}


def checksum_variant(artifacts_dir: Path) -> list[dict]:
    records = []
    for npz_path in sorted(artifacts_dir.rglob("*.npz")):
        rel = npz_path.relative_to(artifacts_dir)
        # Skip non-deterministic analyzers
        top_dir = rel.parts[0] if rel.parts else ""
        if top_dir in EXCLUDED_ANALYZERS:
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
        default=Path(__file__).parent.parent / "tests" / "regression" / "reference_checksums.json",
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
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nChecksums written to {args.output}")
    total = sum(v["artifact_count"] for v in output["variants"])
    print(f"Total: {len(output['variants'])} variants, {total} artifacts")


if __name__ == "__main__":
    main()
