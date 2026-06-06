"""Input Trace Analyzer — REQ_075.

Per-checkpoint predictions on all input pairs (train + test), enabling
residue-class graduation analysis and pair-level learning structure visualization.

Training pairs reach ~100% top-1 accuracy around epoch 200 across all variants —
well before train loss reaches zero. This is expected: loss tracks confidence
(log-probability), accuracy tracks argmax correctness. The test set is where
the grokking signal lives.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# Per-input trace: one row per (a, b) input (row_id = a * p + b). The basis for
# per-input-trace analysis — kept compact via narrow dtypes.
SPEC = AnalyzerSpec(
    name="input_trace",
    output_scope="per_epoch",
    # needs_cache=True because ctx.logits comes from the forward pass and
    # the pipeline only populates inputs.logits when cache is loaded.
    inputs=(ModelInput(needs_weights=False, needs_cache=True),),
    required_hooks=(),
    produces_summary=True,
    outputs=(
        F.columnar(
            "predictions",
            "int16",
            ("variant", "epoch", "row_id"),
            "Predicted class (argmax logit) for each input.",
        ),
        F.columnar(
            "correct",
            "bool",
            ("variant", "epoch", "row_id"),
            "Whether the prediction matches the target for each input.",
        ),
        F.columnar(
            "confidence",
            "float16",
            ("variant", "epoch", "row_id"),
            "Softmax probability assigned to the predicted class for each input.",
        ),
        F.columnar(
            "split",
            "bool",
            ("variant", "epoch", "row_id"),
            "Train (True) vs. test (False) membership of each input.",
        ),
    ),
)


@register_analyzer(SPEC)
class InputTraceAnalyzer:
    """Records per-pair predictions on all p² input pairs at each checkpoint.

    Uses the full probe already passed by the pipeline (no extra forward pass
    beyond the additional model(probe) call; the pipeline's cache run is on the
    same probe). Stores a `split` boolean array to distinguish train from test.

    Per-epoch artifact keys:
        predictions  int16   (p²,)   argmax of output logits for each pair
        correct      bool    (p²,)   predictions == true labels
        confidence   float16 (p²,)   max softmax probability
        split        bool    (p²,)   True = training pair, False = test pair

    Pairs are in probe order: pair k → a = k // p, b = k % p.
    """

    name = "input_trace"
    description = "Per-pair predictions on all input pairs at each checkpoint"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run predictions on all p² pairs and record train/test split."""
        assert inputs.logits is not None  # type-narrowing for pyright
        params = context["params"]
        p = int(params["prime"])
        data_seed = int(params.get("data_seed", 598))
        training_fraction = float(params.get("training_fraction", 0.3))

        # Logits shape: (batch, seq_len, vocab) for transformers,
        # (batch, vocab) for MLPs. Reduce to (batch, vocab) by taking
        # the last sequence position when applicable.
        logits = inputs.logits
        last_logits = logits[:, -1, :] if logits.ndim == 3 else logits  # (p², p)
        device = last_logits.device
        probs = last_logits.softmax(dim=-1)
        predictions = last_logits.argmax(dim=-1)

        a_all = torch.arange(p, device=device).repeat_interleave(p)
        b_all = torch.arange(p, device=device).repeat(p)
        true_labels = (a_all + b_all) % p

        correct = predictions == true_labels
        confidence = probs.max(dim=-1).values
        split = _build_split_mask(p, data_seed, training_fraction)

        return {
            "predictions": predictions.cpu().numpy().astype(np.int16),
            "correct": correct.cpu().numpy(),
            "confidence": confidence.cpu().numpy().astype(np.float16),
            "split": split,
        }

    def get_summary_keys(self) -> list[str]:
        """Declare per-epoch summary statistics accumulated across checkpoints."""
        return [
            "test_residue_class_accuracy",
            "train_residue_class_accuracy",
            "test_overall_accuracy",
            "train_overall_accuracy",
        ]

    def compute_summary(
        self, result: dict[str, np.ndarray], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute per-epoch accuracy metrics for train and test splits separately.

        Args:
            result: Per-epoch artifact dict from analyze()
            context: Family-provided analysis context

        Returns:
            Dict with per-residue and overall accuracy for both splits
        """
        p = int(context["params"]["prime"])
        correct = result["correct"]
        split = result["split"]

        a_all = np.arange(p).repeat(p)  # a = k // p  for k in 0..p²-1
        b_all = np.tile(np.arange(p), p)  # b = k % p
        residue = (a_all + b_all) % p

        test_mask = ~split
        train_mask = split

        test_residue_acc = _per_residue_accuracy(correct, residue, test_mask, p)
        train_residue_acc = _per_residue_accuracy(correct, residue, train_mask, p)

        test_overall = float(correct[test_mask].mean()) if test_mask.any() else 0.0
        train_overall = float(correct[train_mask].mean()) if train_mask.any() else 0.0

        return {
            "test_residue_class_accuracy": test_residue_acc,
            "train_residue_class_accuracy": train_residue_acc,
            "test_overall_accuracy": np.float32(test_overall),
            "train_overall_accuracy": np.float32(train_overall),
        }


def _per_residue_accuracy(
    correct: np.ndarray,
    residue: np.ndarray,
    mask: np.ndarray,
    p: int,
) -> np.ndarray:
    """Compute fraction correct per residue class, restricted to masked pairs."""
    acc = np.zeros(p, dtype=np.float32)
    for c in range(p):
        class_mask = mask & (residue == c)
        if class_mask.any():
            acc[c] = correct[class_mask].mean()
    return acc


def _build_split_mask(p: int, data_seed: int, training_fraction: float) -> np.ndarray:
    """Reconstruct the train/test split as a boolean mask over all p² pairs.

    Uses the same RNG state as generate_training_dataset to guarantee
    the reconstructed split is identical to the one used during training.

    Args:
        p: The prime modulus
        data_seed: Random seed for train/test split
        training_fraction: Fraction of pairs used for training

    Returns:
        (p²,) bool array — True = training pair, False = test pair
        Indexed in probe order: index k → a = k // p, b = k % p
    """
    torch.manual_seed(data_seed)
    indices = torch.randperm(p * p)
    cutoff = int(p * p * training_fraction)
    train_indices = indices[:cutoff]

    split = np.zeros(p * p, dtype=bool)
    split[train_indices.numpy()] = True
    return split
