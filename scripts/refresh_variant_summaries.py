"""Force run of the variant_analysis_summary across all variants.

Usage:
    uv run python scripts/refresh_variant_summaries.py
"""

from __future__ import annotations

import argparse
import sys

from miscope import load_family
from miscope.analysis.variant_analysis_summary import VariantAnalysisSummary, build_variant_registry


def run(family_name: str) -> None:
    family = load_family(family_name)
    if not family.family_dir.exists():
        print(f"Family directory not found: {family.family_dir}")
        sys.exit(1)

    for variant in family.variants:
        summary = VariantAnalysisSummary(variant)
        summary.analyze()

    build_variant_registry(family)


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
