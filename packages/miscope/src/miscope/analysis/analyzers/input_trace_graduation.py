"""Input Trace Graduation Analyzer — REQ_075.

Cross-epoch analyzer that computes graduation epochs from per-checkpoint
input_trace artifacts. A pair "graduates" at the first checkpoint where it
is correct for min_stable_window consecutive checkpoints.

This must be a CrossEpochAnalyzer (not summary_stats) because graduation
detection requires a backward look over all checkpoints: a checkpoint cannot
be declared the graduation point until future checkpoints confirm stability.
"""

from typing import Any

import numpy as np

from miscope.analysis.inputs import ArtifactInput, ResolvedInputs
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# Per-input graduation epoch (when an input becomes permanently correct). One
# row per (a, b) input; row_id = a * p + b.
SPEC = AnalyzerSpec(
    name="input_trace_graduation",
    output_scope="cross_epoch",
    inputs=(ArtifactInput("input_trace"),),
    outputs=(
        F.columnar(
            "graduation_epochs",
            "int32",
            ("variant", "row_id"),
            "Epoch at which each input becomes permanently correct (-1 if never).",
        ),
        F.columnar(
            "epochs",
            "int32",
            ("variant", "epoch"),
            "Epoch axis scanned for graduation.",
        ),
        F.columnar(
            "split",
            "bool",
            ("variant", "row_id"),
            "Train (True) vs. test (False) membership of each input.",
        ),
    ),
)


@register_analyzer(SPEC)
class InputTraceGraduationAnalyzer:
    """Computes graduation epochs across all input_trace checkpoints.

    Graduation epoch is the first checkpoint epoch at which a pair becomes
    correct and remains correct for at least min_stable_window consecutive
    checkpoints (default: 3). Pairs that never achieve this stability get -1.

    Cross-epoch artifact keys:
        graduation_epochs  int32  (p²,)           first stable-correct epoch, -1 if never
        epochs             int32  (n_checkpoints,) epoch numbers in order
        split              bool   (p²,)            True = training pair, False = test pair
    """

    name = "input_trace_graduation"
    requires = ["input_trace"]

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute graduation epochs from all input_trace per-epoch artifacts.

        Args:
            artifacts_dir: Root artifacts directory for the variant
            epochs: Sorted list of available epoch numbers
            context: Family-provided analysis context (unused)

        Returns:
            Dict with 'graduation_epochs', 'epochs', 'split'
        """
        assert inputs.deps is not None
        assert inputs.epochs is not None
        min_stable_window = 3
        sorted_epochs = sorted(inputs.epochs)

        # True per-epoch reducer: stream input_trace at the run's analyzed epochs,
        # keeping only the per-epoch correctness row (not the whole artifact).
        # `split` is constant across epochs — captured once.
        split: np.ndarray | None = None
        correct_rows: list[np.ndarray] = []
        for _epoch, artifact in inputs.deps.stream(
            "input_trace", epochs=sorted_epochs, fields=["split", "correct"]
        ):
            if split is None:
                split = artifact["split"]  # (p²,)
            correct_rows.append(artifact["correct"])

        assert split is not None
        correct_matrix = np.array(correct_rows, dtype=bool)  # (n_epochs, n_pairs)
        graduation_epochs = _compute_graduation_epochs(
            correct_matrix, sorted_epochs, min_stable_window
        )

        return {
            "graduation_epochs": graduation_epochs.astype(np.int32),
            "epochs": np.array(sorted_epochs, dtype=np.int32),
            "split": split,
        }


def _compute_graduation_epochs(
    correct_matrix: np.ndarray,
    epochs: list[int],
    window: int,
) -> np.ndarray:
    """Find first epoch where each pair is correct for `window` consecutive checkpoints.

    Uses a cumulative sum approach to check window sums in one vectorized pass.

    Args:
        correct_matrix: (n_epochs, n_pairs) bool — correctness at each checkpoint
        epochs: Epoch numbers corresponding to rows of correct_matrix
        window: Number of consecutive correct checkpoints required for graduation

    Returns:
        (n_pairs,) int32 — graduation epoch number, or -1 if never graduated
    """
    n_epochs, n_pairs = correct_matrix.shape
    epochs_arr = np.array(epochs, dtype=np.int32)

    if window > n_epochs:
        return np.full(n_pairs, -1, dtype=np.int32)

    # cumsum padded with a zero row for clean window sum computation:
    # padded[i+1] = sum of correct[0:i+1]
    # window_sums[i] = padded[i+window] - padded[i] = sum of correct[i:i+window]
    cumsum = np.cumsum(correct_matrix.astype(np.int32), axis=0)
    padded = np.vstack([np.zeros((1, n_pairs), dtype=np.int32), cumsum])

    n_valid = n_epochs - window + 1
    window_sums = padded[window : window + n_valid] - padded[:n_valid]  # (n_valid, n_pairs)

    window_stable = window_sums == window  # (n_valid, n_pairs) bool

    any_graduated = window_stable.any(axis=0)
    first_stable_idx = np.where(any_graduated, window_stable.argmax(axis=0), -1)

    return np.where(
        first_stable_idx == -1,
        np.int32(-1),
        epochs_arr[first_stable_idx],
    )
