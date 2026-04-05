"""Force run of the variant_analysis_summary across all variants.

Usage:
    uv run python scripts/refresh_variant_summaries.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from miscope import load_family
from miscope.analysis.variant_analysis_summary import VariantAnalysisSummary, build_variant_registry


def run(results_dir: Path, family_name: str) -> None:
    family_dir = results_dir / family_name
    if not family_dir.exists():
        print(f"Family directory not found: {family_dir}")
        sys.exit(1)

    family = load_family(family_name)
    for variant in family.list_variants():
        summary = VariantAnalysisSummary(variant)
        summary.analyze()

    build_variant_registry(results_dir, family_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--family",
        default="modulo_addition_1layer",
        help="Family subdirectory name",
    )
    args = parser.parse_args()

    run(
        results_dir=args.results_dir,
        family_name=args.family,
    )


if __name__ == "__main__":
    main()
