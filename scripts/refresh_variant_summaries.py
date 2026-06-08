"""Force run of the variant_analysis_summary across all variants.

Usage:
    uv run python scripts/refresh_variant_summaries.py
"""

from __future__ import annotations

import argparse
import sys

from miscope import load_family
from miscope.analysis.variant_analysis_summary import build_variant_registry, write_variant_summary


def run(family_name: str) -> None:
    family = load_family(family_name)
    if not family.family_dir.exists():
        print(f"Family directory not found: {family.family_dir}")
        sys.exit(1)

    # Per-variant isolation: a variant that can't summarize (e.g. not yet
    # re-analyzed for the REQ_141 neuron_frequency_attribution analyzer, so its
    # conformed dimension has no source) is skipped and reported, not fatal to the
    # batch — mirrors the warehouse materializer's quarantine discipline (REQ_140).
    failed: dict[str, str] = {}
    for variant in family.variants:
        try:
            write_variant_summary(variant)
        except Exception as exc:  # noqa: BLE001 — quarantine one variant, keep the batch going
            failed[variant.name] = f"{type(exc).__name__}: {exc}"
            print(f"  skipped {variant.name}: {type(exc).__name__}: {exc}")

    build_variant_registry(family)

    if failed:
        print(
            f"\n{len(failed)} variant(s) skipped (commonly: not yet re-analyzed for the "
            f"neuron_frequency_attribution analyzer — re-run analysis on them):"
        )
        for name in sorted(failed):
            print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default="modulo_addition_1layer",
        help="Family name (subdirectory under the data root).",
    )
    args = parser.parse_args()

    run(family_name=args.family)


if __name__ == "__main__":
    main()
