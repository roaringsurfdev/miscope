"""Regression check: re-run analysis and compare against reference checksums.

Re-runs the analysis pipeline for each reference variant into a secondary output
directory, then compares every output ``.npz`` against the checksums in
``tests/regression/reference_checksums.json``.

The analyzer set comes from the canonical pattern — ``select_specs`` over the
family Registry, planned through ``plan_analysis`` and executed via
``pipeline.run(plan=...)`` — so it cannot fall behind the dependency graph
(REQ_134). The recompute always runs forced — the point of the check is to recompute with
the *current* code and compare. Existing artifacts are symlinked into the output
tree before recompute so cross-epoch→cross-epoch dependencies resolve via on-disk
state; recompute is non-destructive to the originals (see
``seed_artifact_overlay``).

Exit code 0 = all checksums matched.
Exit code 1 = one or more mismatches or missing files.

Usage:
    uv run python scripts/run_regression_check.py

Options:
    --output-dir PATH   Secondary results directory (default: results_regression/)
    --checksums PATH    Checksums file (default: tests/regression/reference_checksums.json)
    --variants IDS      Comma-separated variant_ids to check (default: all)
    --no-recompute      Skip the recompute; compare the canonical on-disk artifacts
                        (data_root) against the checksums directly. Fast integrity
                        check — confirms the stored artifacts still match the
                        recorded checksums (no MISSING/EXTRA/MISMATCH) without
                        running any analysis. Note: this does not exercise the
                        analyzers, so it cannot catch a regression introduced by a
                        code change — only a true recompute (the default) does that.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from regression_common import (
    EXCLUDE_FROM_REGRESSION,
    FAMILY,
    PROJECT_ROOT,
    REFERENCE_CHECKSUMS_PATH,
    plan_regression,
    seed_artifact_overlay,
    sha256_file,
)


def run_pipeline(variant) -> None:
    """Recompute the canonical regression analyzer set for a variant.

    Mirrors the dashboard / ``run_analysis.py`` pattern: enumerate Specs from the
    family Registry (minus the shared exclude set), plan, and run the plan. The
    set therefore tracks the dependency graph automatically — every declared
    upstream is present, so the run never aborts on a ``blocked_by`` for a
    missing upstream (CoS 1, 4).

    Always forced: the symlink overlay makes every artifact look fresh, so a
    non-forced run would recompute nothing and pass trivially. Forcing recompute
    is what makes the byte comparison meaningful.
    """
    from miscope.analysis import AnalysisPipeline

    plan = plan_regression(variant, force=True)
    pipeline = AnalysisPipeline(variant)
    pipeline.run(plan=plan, force=True)


def compare_variant(
    variant_entry: dict,
    output_artifacts_dir: Path,
) -> list[str]:
    """Compare output artifacts against reference checksums. Returns error messages."""
    errors = []
    ref_by_path = {a["artifact_path"]: a for a in variant_entry["artifacts"]}
    vid = variant_entry["variant_id"]

    # Check every reference artifact exists and matches.
    for rel_path, ref in ref_by_path.items():
        actual_path = output_artifacts_dir / rel_path
        if not actual_path.exists():
            errors.append(f"  MISSING  {vid}/{rel_path}")
            continue
        actual_sha = sha256_file(actual_path)
        if actual_sha != ref["sha256"]:
            errors.append(f"  MISMATCH {vid}/{rel_path}")

    # Check for unexpected extra artifacts (ignore excluded analyzers).
    if output_artifacts_dir.exists():
        for actual_path in sorted(output_artifacts_dir.rglob("*.npz")):
            rel = str(actual_path.relative_to(output_artifacts_dir))
            top_dir = Path(rel).parts[0] if Path(rel).parts else ""
            if top_dir in EXCLUDE_FROM_REGRESSION:
                continue
            if rel not in ref_by_path:
                errors.append(f"  EXTRA    {vid}/{rel}")

    return errors


def prepare_output_variant(original_variant_dir: Path, output_variant_dir: Path) -> Path:
    """Build the isolated tree for one variant; return its artifacts dir.

    Checkpoints and config are symlinked (read-only inputs). The artifacts dir is
    rebuilt fresh each run and seeded with a symlink overlay of the existing
    artifacts, so the recompute runs against on-disk upstream state (CoS 5).
    """
    output_variant_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "config.json", "metadata.json", "variant_summary.json"):
        src = original_variant_dir / name
        dst = output_variant_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    output_artifacts_dir = output_variant_dir / "artifacts"
    if output_artifacts_dir.exists() or output_artifacts_dir.is_symlink():
        shutil.rmtree(output_artifacts_dir)
    output_artifacts_dir.mkdir(parents=True)
    n = seed_artifact_overlay(original_variant_dir / "artifacts", output_artifacts_dir)
    print(f"  Seeded {n} existing artifacts as symlinks (incremental on-disk state).")
    return output_artifacts_dir


def recompute_artifacts(vid: str, cfg, output_dir: Path) -> Path | None:
    """Recompute one variant into the isolated tree; return its artifacts dir.

    Mirrors family config, seeds the symlink overlay, loads the variant from the
    output root, and runs the forced recompute. Returns None if the variant
    cannot be loaded from the output tree.
    """
    from miscope.families.discovery import discover_families

    original_variant_dir = cfg.data_root / FAMILY / "variants" / vid

    # Mirror the family-level config so discover_families finds the family
    # under the regression output root.
    for cfg_name in ("family.json", "ideal_frequency_sets.json"):
        src = cfg.data_root / FAMILY / cfg_name
        dst = output_dir / FAMILY / cfg_name
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())

    output_variant_dir = output_dir / FAMILY / "variants" / vid
    output_artifacts_dir = prepare_output_variant(original_variant_dir, output_variant_dir)

    out_family = discover_families(output_dir)[FAMILY]
    variant = next((v for v in out_family.variants if v.name == vid), None)
    if variant is None:
        print("  ERROR — could not load variant from output dir")
        return None

    run_pipeline(variant)
    return output_artifacts_dir


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
        default=REFERENCE_CHECKSUMS_PATH,
        help="Reference checksums file",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant_ids to check (default: all)",
    )
    parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="Compare existing on-disk artifacts (data_root) without recomputing.",
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

    cfg = get_config()

    all_errors: list[str] = []

    for entry in variants_to_check:
        vid = entry["variant_id"]
        print(f"\n{'=' * 60}")
        print(f"Variant: {vid}")

        original_variant_dir = cfg.data_root / FAMILY / "variants" / vid
        if not original_variant_dir.exists():
            print(f"  SKIP — original variant directory not found: {original_variant_dir}")
            all_errors.append(f"  SKIP    {vid} — original not found")
            continue

        if args.no_recompute:
            print("  Comparing existing on-disk artifacts (no recompute)...")
            compare_artifacts_dir = original_variant_dir / "artifacts"
        else:
            print(f"  Re-running analysis into {args.output_dir}/...")
            compare_artifacts_dir = recompute_artifacts(vid, cfg, args.output_dir)
            if compare_artifacts_dir is None:
                all_errors.append(f"  ERROR   {vid} — variant load failed")
                continue

        errors = compare_variant(entry, compare_artifacts_dir)

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
