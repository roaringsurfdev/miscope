"""REQ_118: Per-epoch neuron grouping analyzer.

SecondaryAnalyzer that consumes per-epoch parameter_snapshot artifacts
and produces per-epoch GroupAssignment + GroupSummary artifacts.

At each checkpoint:

1. If the family supplied a `neuron_grouping_override` callable in the
   analysis context, call it with the parameter_snapshot artifact and
   the context. The override returns a `GroupAssignment` derived in the
   family's natural feature space (e.g., Fourier projection for
   modular addition).
2. Otherwise, run the universal path: build a per-neuron feature matrix
   by concatenating each neuron's W_in column with its W_out row, then
   call `group_neurons(features, ..., method="kmeans")`.

In either case, computes the GroupSummary on the same feature matrix
(or the matrix the override returned) and packs both the assignment
and the summary into the per-epoch artifact.

Per-epoch storage at `artifacts/neuron_grouping/epoch_{NNNNN}.npz`,
matching the per-epoch convention used by every other primary /
secondary analyzer.

Cross-epoch consumers (e.g., REQ_117 parameter_dmd) pick a single
epoch's artifact via a `reference_epoch` configuration on their side;
the grouping analyzer itself does not make that choice.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from miscope.analysis.library.grouping import (
    group_neurons,
    group_neurons_summary,
)
from miscope.core.grouping import GroupAssignment, GroupSummary

# Context keys the family may set to influence the analyzer.
_CONTEXT_OVERRIDE_KEY = "neuron_grouping_override"
_CONTEXT_N_GROUPS_KEY = "neuron_grouping_n_groups"
_CONTEXT_FEATURE_SOURCE_KEY = "neuron_grouping_feature_source"
_CONTEXT_OVERRIDE_FEATURES_KEY = "_neuron_grouping_override_features"

# Default for the universal path. Family overrides bypass this.
_DEFAULT_FEATURE_SOURCE = "weight"
_DEFAULT_N_GROUPS = 8


class NeuronGrouping:
    """Per-epoch neuron grouping. Consumes parameter_snapshot artifacts."""

    name = "neuron_grouping"
    depends_on = "parameter_snapshot"

    def analyze(
        self,
        artifact: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute per-epoch grouping for one checkpoint.

        Args:
            artifact: parameter_snapshot artifact for this epoch
                (`W_in`, `W_out`, plus the rest of the weight matrices).
            context: Family-provided analysis context. Optional keys:
                - ``neuron_grouping_override``: callable
                  ``(artifact, context) -> (GroupAssignment, features)``
                  that the family supplies to take precedence over the
                  universal path. The features it returns are the same
                  features the assignment was derived from, used for
                  the summary computation.
                - ``neuron_grouping_n_groups``: int — group count for
                  the universal path (kmeans). Defaults to 8 if absent.
                - ``neuron_grouping_feature_source``: str — currently
                  only ``"weight"`` supported in the universal path.

        Returns:
            Dict of arrays for storage in epoch_{NNNNN}.npz.
        """
        override: Callable | None = context.get(_CONTEXT_OVERRIDE_KEY)
        if override is not None:
            assignment, features = override(artifact, context)
            had_override = True
        else:
            features = _extract_features(artifact, context)
            n_groups = int(context.get(_CONTEXT_N_GROUPS_KEY, _DEFAULT_N_GROUPS))
            assignment = group_neurons(
                features,
                n_groups=n_groups,
                method="kmeans",
                feature_basis_name="weight_signature",
            )
            had_override = False

        summary = group_neurons_summary(features, assignment)

        return _pack(assignment, summary, had_override)


def _extract_features(
    artifact: dict[str, Any],
    context: dict[str, Any],
) -> np.ndarray:
    """Universal-path feature extraction: per-neuron weight signature.

    For each MLP neuron, concatenate its W_in column (length d_model)
    with its W_out row (length d_model). Result: (d_mlp, 2 * d_model)
    feature matrix.
    """
    feature_source = context.get(_CONTEXT_FEATURE_SOURCE_KEY, _DEFAULT_FEATURE_SOURCE)
    if feature_source != "weight":
        raise NotImplementedError(
            f"feature_source='{feature_source}' not yet supported in universal path "
            f"(only 'weight' available in v1)"
        )
    w_in = np.asarray(artifact["W_in"], dtype=np.float64)  # (d_model, d_mlp)
    w_out = np.asarray(artifact["W_out"], dtype=np.float64)  # (d_mlp, d_model)
    # Per-neuron feature: W_in[:, n] (d_model,) concatenated with W_out[n, :] (d_model,)
    return np.concatenate([w_in.T, w_out], axis=1)  # (d_mlp, 2*d_model)


def _pack(
    assignment: GroupAssignment,
    summary: GroupSummary,
    had_override: bool,
) -> dict[str, np.ndarray]:
    """Pack canonical types into npz-storable dict."""
    out: dict[str, np.ndarray] = {
        "assignments": assignment.assignments,
        "n_groups": np.array(assignment.n_groups, dtype=np.int64),
        "method": np.array(assignment.method, dtype="U64"),
        "feature_basis_name": np.array(assignment.feature_basis_name, dtype="U64"),
        "centroids": summary.centroids,
        "radii": summary.radii,
        "n_per_group": summary.n_per_group,
        "n_unassigned": np.array(summary.n_unassigned, dtype=np.int64),
        "fisher_min": np.array(summary.fisher_min, dtype=np.float64),
        "fisher_mean": np.array(summary.fisher_mean, dtype=np.float64),
        "dispersion": np.array(summary.dispersion, dtype=np.float64),
        "had_family_override": np.array(had_override, dtype=np.bool_),
    }
    if assignment.confidence is not None:
        out["confidence"] = assignment.confidence
    return out


def unpack_assignment(artifact: dict[str, Any]) -> GroupAssignment:
    """Reconstruct a `GroupAssignment` from a per-epoch npz artifact.

    Helper for downstream consumers (e.g., REQ_117 parameter_dmd) that
    load a `neuron_grouping` artifact at a chosen reference epoch and
    want the canonical typed view.
    """
    return GroupAssignment(
        assignments=np.asarray(artifact["assignments"], dtype=np.int64),
        n_groups=int(artifact["n_groups"]),
        method=str(artifact["method"]),
        feature_basis_name=str(artifact["feature_basis_name"]),
        confidence=(
            np.asarray(artifact["confidence"], dtype=np.float64)
            if "confidence" in artifact
            else None
        ),
    )


def unpack_summary(artifact: dict[str, Any]) -> GroupSummary:
    """Reconstruct a `GroupSummary` from a per-epoch npz artifact."""
    return GroupSummary(
        centroids=np.asarray(artifact["centroids"], dtype=np.float64),
        radii=np.asarray(artifact["radii"], dtype=np.float64),
        n_per_group=np.asarray(artifact["n_per_group"], dtype=np.int64),
        n_unassigned=int(artifact["n_unassigned"]),
        fisher_min=float(artifact["fisher_min"]),
        fisher_mean=float(artifact["fisher_mean"]),
        dispersion=float(artifact["dispersion"]),
    )
