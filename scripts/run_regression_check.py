"""Regression check: re-run analysis and compare against reference checksums.

Re-runs the full analysis pipeline for each reference variant into a
secondary output directory, then compares every output .npz file against
the checksums in regression/reference_checksums.json.

Exit code 0 = all checksums matched.
Exit code 1 = one or more mismatches or missing files.

Usage:
    uv run python scripts/run_regression_check.py

Options:
    --output-dir PATH   Secondary results directory (default: results_regression/)
    --checksums PATH    Checksums file (default: regression/reference_checksums.json)
    --variants IDS      Comma-separated variant_ids to check (default: all)
    --force             Re-run analysis even if artifacts already exist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FAMILY = "modulo_addition_1layer"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(variant, force: bool) -> None:
    """Run the full analysis pipeline for a variant."""
    from miscope.analysis import AnalysisPipeline
    from miscope.analysis.analyzers import (
        AttentionFourierAnalyzer,
        AttentionFreqAnalyzer,
        AttentionPatternsAnalyzer,
        CentroidDMD,
        DominantFrequenciesAnalyzer,
        EffectiveDimensionalityAnalyzer,
        FourierFrequencyQualityAnalyzer,
        FourierNucleationAnalyzer,
        GlobalCentroidPCA,
        GradientSiteAnalyzer,
        InputTraceAnalyzer,
        InputTraceGraduationAnalyzer,
        NeuronActivationsAnalyzer,
        NeuronDynamicsAnalyzer,
        NeuronFourierAnalyzer,
        NeuronFreqClustersAnalyzer,
        NeuronGroupPCAAnalyzer,
        ParameterSnapshotAnalyzer,
        ParameterTrajectoryPCA,
        RepresentationalGeometryAnalyzer,
        TransientFrequencyAnalyzer,
    )

    pipeline = AnalysisPipeline(variant)
    pipeline.register(AttentionFreqAnalyzer())
    pipeline.register(AttentionPatternsAnalyzer())
    pipeline.register(DominantFrequenciesAnalyzer())
    pipeline.register(InputTraceAnalyzer())
    pipeline.register(NeuronActivationsAnalyzer())
    pipeline.register(NeuronFreqClustersAnalyzer())
    pipeline.register(ParameterSnapshotAnalyzer())
    pipeline.register(EffectiveDimensionalityAnalyzer())
    # LandscapeFlatnessAnalyzer excluded: stochastic by design, not regression-testable
    pipeline.register(RepresentationalGeometryAnalyzer())
    pipeline.register(AttentionFourierAnalyzer())
    pipeline.register(FourierNucleationAnalyzer())
    pipeline.register_secondary(FourierFrequencyQualityAnalyzer())
    pipeline.register_secondary(NeuronFourierAnalyzer())
    pipeline.register_cross_epoch(InputTraceGraduationAnalyzer())
    pipeline.register_cross_epoch(NeuronDynamicsAnalyzer())
    pipeline.register_cross_epoch(NeuronGroupPCAAnalyzer())
    pipeline.register_cross_epoch(ParameterTrajectoryPCA())
    pipeline.register_cross_epoch(GlobalCentroidPCA())
    pipeline.register_cross_epoch(CentroidDMD())
    pipeline.register_cross_epoch(TransientFrequencyAnalyzer())
    pipeline.register_cross_epoch(GradientSiteAnalyzer())
    pipeline.run(force=force)


# Analyzers excluded from regression: stochastic output, not byte-comparable.
EXCLUDED_ANALYZERS = {"landscape_flatness", "coarseness"}


def compare_variant(
    variant_entry: dict,
    output_artifacts_dir: Path,
) -> list[str]:
    """Compare output artifacts against reference checksums. Returns list of error messages."""
    errors = []
    ref_by_path = {a["artifact_path"]: a for a in variant_entry["artifacts"]}
    vid = variant_entry["variant_id"]

    # Check every reference artifact exists and matches
    for rel_path, ref in ref_by_path.items():
        actual_path = output_artifacts_dir / rel_path
        if not actual_path.exists():
            errors.append(f"  MISSING  {vid}/{rel_path}")
            continue
        actual_sha = sha256_file(actual_path)
        if actual_sha != ref["sha256"]:
            errors.append(f"  MISMATCH {vid}/{rel_path}")

    # Check for unexpected extra artifacts (ignore excluded analyzers)
    if output_artifacts_dir.exists():
        for actual_path in sorted(output_artifacts_dir.rglob("*.npz")):
            rel = str(actual_path.relative_to(output_artifacts_dir))
            top_dir = Path(rel).parts[0] if Path(rel).parts else ""
            if top_dir in EXCLUDED_ANALYZERS:
                continue
            if rel not in ref_by_path:
                errors.append(f"  EXTRA    {vid}/{rel}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results_regression",
        help="Secondary results directory for re-run output",
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        default=PROJECT_ROOT / "regression" / "reference_checksums.json",
        help="Reference checksums file",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant_ids to check (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run analysis even if artifacts already exist",
    )
    args = parser.parse_args()

    if not args.checksums.exists():
        print(f"ERROR: Checksums file not found: {args.checksums}")
        print("Run scripts/generate_regression_checksums.py first.")
        sys.exit(1)

    ref_data = json.loads(args.checksums.read_text())
    variants_to_check = ref_data["variants"]

    if args.variants:
        requested = set(args.variants.split(","))
        variants_to_check = [v for v in variants_to_check if v["variant_id"] in requested]
        if not variants_to_check:
            print(f"ERROR: No matching variants found for: {args.variants}")
            sys.exit(1)

    from miscope.config import get_config
    from miscope.families.registry import FamilyRegistry

    cfg = get_config()
    src_registry = FamilyRegistry(cfg.model_families_dir, cfg.results_dir)
    family = src_registry.get_family(FAMILY)

    all_errors: list[str] = []

    for entry in variants_to_check:
        vid = entry["variant_id"]
        # p, s, ds = entry["prime"], entry["model_seed"], entry["data_seed"]
        print(f"\n{'=' * 60}")
        print(f"Variant: {vid}")
        print(f"  Re-running analysis into {args.output_dir}/...")

        # Symlink checkpoints and config from original into output dir so the
        # pipeline can load checkpoints while writing artifacts to output_dir.
        original_variant_dir = PROJECT_ROOT / "results" / FAMILY / vid
        if not original_variant_dir.exists():
            print(f"  SKIP — original variant directory not found: {original_variant_dir}")
            all_errors.append(f"  SKIP    {vid} — original not found")
            continue

        output_variant_dir = args.output_dir / FAMILY / vid
        output_variant_dir.mkdir(parents=True, exist_ok=True)

        for name in ("checkpoints", "config.json", "metadata.json", "variant_summary.json"):
            src = original_variant_dir / name
            dst = output_variant_dir / name
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())

        out_registry = FamilyRegistry(cfg.model_families_dir, args.output_dir)
        variant = next((v for v in out_registry.get_variants(family) if v.name == vid), None)
        if variant is None:
            print("  ERROR — could not load variant from output dir")
            all_errors.append(f"  ERROR   {vid} — variant load failed")
            continue

        run_pipeline(variant, force=args.force)

        output_artifacts_dir = args.output_dir / FAMILY / vid / "artifacts"
        errors = compare_variant(entry, output_artifacts_dir)

        if errors:
            print(f"  FAILED — {len(errors)} issue(s):")
            for e in errors:
                print(e)
            all_errors.extend(errors)
        else:
            print(f"  PASSED — {entry['artifact_count']} artifacts matched")

    print(f"\n{'=' * 60}")
    if all_errors:
        print(f"REGRESSION FAILED — {len(all_errors)} issue(s) across all variants")
        sys.exit(1)
    else:
        print(
            f"REGRESSION PASSED — all {sum(v['artifact_count'] for v in variants_to_check)} artifacts matched"
        )


if __name__ == "__main__":
    main()
