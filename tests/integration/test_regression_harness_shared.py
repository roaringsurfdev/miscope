"""REQ_134: the three regression scripts must share one analyzer set / exclude
list / checksums path — not re-derive their own.

These tests guard the single-source-of-truth invariant the requirement restores:

  * CoS 2 — one shared exclude set: each script imports it from
    ``regression_common`` and none defines its own set literal.
  * CoS 4 — no missing-upstream abort: the canonical selection is closed under
    its ``ArtifactInput`` dependencies (every declared upstream is selected).
  * CoS 3 — generator and checker resolve the same checksums path.

The exclude-set checks are AST-static (parse, don't execute) so the
notebook-style ``run_analysis_regression.py`` is not run as a side effect of
testing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import regression_common

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
HARNESS_SCRIPTS = (
    "generate_regression_checksums.py",
    "run_regression_check.py",
    "run_analysis_regression.py",
)

# Names that must live only in regression_common — a script binding any of these
# at module level would be re-deriving the shared set/path.
SHARED_NAMES = {"EXCLUDE_FROM_REGRESSION", "DEPRECATED_ANALYZERS", "REFERENCE_CHECKSUMS_PATH"}


def _module_ast(script_name: str) -> ast.Module:
    return ast.parse((SCRIPTS_DIR / script_name).read_text())


def _imports_from_regression_common(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "regression_common":
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("script_name", HARNESS_SCRIPTS)
def test_script_imports_from_regression_common(script_name: str) -> None:
    """Every harness script pulls shared config from regression_common."""
    imported = _imports_from_regression_common(_module_ast(script_name))
    assert imported, f"{script_name} does not import from regression_common"


@pytest.mark.parametrize("script_name", HARNESS_SCRIPTS)
def test_script_does_not_redefine_shared_names(script_name: str) -> None:
    """No script defines its own exclude set / checksums path (CoS 2)."""
    tree = _module_ast(script_name)
    offenders = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in SHARED_NAMES:
                offenders.append(target.id)
    assert not offenders, (
        f"{script_name} re-defines shared name(s) {sorted(set(offenders))}; "
        "import them from regression_common instead"
    )


def test_generator_and_checker_share_checksums_path() -> None:
    """Generator output default and checker input default resolve to one path (CoS 3)."""
    import generate_regression_checksums as generator
    import run_regression_check as checker

    assert generator.REFERENCE_CHECKSUMS_PATH == checker.REFERENCE_CHECKSUMS_PATH
    assert checker.REFERENCE_CHECKSUMS_PATH == regression_common.REFERENCE_CHECKSUMS_PATH


def test_exclude_set_filters_known_nondeterministic() -> None:
    """The shared exclude set drops the byte-unstable declared analyzers."""
    assert {"landscape_flatness", "fourier_nucleation"} <= regression_common.EXCLUDE_FROM_REGRESSION


def test_selection_is_closed_under_dependencies() -> None:
    """Canonical selection includes every declared upstream (CoS 4).

    The original failure was an omitted upstream (activation_frequency_norm,
    neuron_grouping) aborting the run on a blocked_by. Because the set derives
    from list_for_family, every ArtifactInput dependency of a selected spec must
    itself be selected.
    """
    from miscope import load_family

    family = load_family(regression_common.FAMILY)
    specs = regression_common.select_specs(family)
    selected = {spec.name for spec in specs}
    assert selected, "selection is empty — family declared no analyzers?"

    for spec in specs:
        for dep in spec.inputs:
            dep_name = getattr(dep, "analyzer_name", None)
            if dep_name is None:  # ModelInput — no analyzer upstream
                continue
            assert dep_name in selected, (
                f"{spec.name} depends on '{dep_name}', which is not in the "
                f"regression selection — run would abort on blocked_by"
            )
