"""REQ_042 / REQ_141: Neuron dynamics cross-epoch analyzer.

Produces the *genuinely cross-epoch* per-neuron frequency-dynamics metrics:
frequency switch counts and commitment epochs (plus the epoch axis and the
uncommitted-frequency threshold they are computed under).

REQ_141 (bucket-1): the per-epoch ``dominant_freq`` / ``max_frac`` attribution —
a per-epoch fact this analyzer used to back-fill by stacking an
``(n_epochs, n_freq, d_mlp)`` norm cube — now belongs to the per-epoch
``neuron_frequency_attribution`` analyzer. This analyzer *streams* that per-epoch
output (small ``(d_mlp,)`` vectors) and stacks only the ``(n_epochs, d_mlp)``
dominant/frac trajectories it needs for the switch/commitment reductions; the
``(n_epochs, n_freq, d_mlp)`` cube is gone. The switch/commitment computations are
unchanged, so their values are identical.
"""

from typing import Any

import numpy as np

from miscope.analysis.inputs import ArtifactInput, ResolvedInputs
from miscope.analysis.library.fourier_basis import get_fourier_basis
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

SPEC = AnalyzerSpec(
    name="neuron_dynamics",
    output_scope="cross_epoch",
    inputs=(ArtifactInput("neuron_frequency_attribution"),),
    outputs=(
        F.columnar(
            "epochs",
            "int64",
            ("variant", "epoch"),
            "Epoch axis labels for the per-epoch trajectories.",
        ),
        F.columnar(
            "switch_counts",
            "int32",
            ("variant", "neuron"),
            "Number of times a neuron changes its dominant frequency over training.",
        ),
        F.columnar(
            "commitment_epochs",
            "float64",
            ("variant", "neuron"),
            "Epoch at which a neuron locks into its final dominant frequency (NaN if never).",
        ),
        F.columnar(
            "threshold",
            "float64",
            ("variant",),
            "Uncommitted-frequency floor (3/n_freq) used at analysis time.",
        ),
    ),
    version=2,  # v2 (REQ_141): dominant_freq/max_frac moved to neuron_frequency_attribution
)


@register_analyzer(SPEC)
class NeuronDynamicsAnalyzer:
    """Cross-epoch analyzer for neuron frequency dynamics (switch + commitment).

    Streams the per-epoch ``neuron_frequency_attribution`` output and reduces it
    over the epoch axis. The dashboard renders neuron dynamics from the conformed
    attribution table plus these cross-epoch scalars.
    """

    name = "neuron_dynamics"
    requires = ["neuron_frequency_attribution"]

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute cross-epoch switch counts + commitment epochs by streaming."""
        assert inputs.deps is not None
        assert inputs.epochs is not None
        prime = int(context["params"]["prime"])

        # n_freq is the family basis frequency count — the same value the old path
        # read off norm_matrix.shape[1] (reconstruct yields (n_freq, d_mlp) with
        # n_freq == basis.n_frequencies), so the 3/n_freq threshold is identical.
        n_freq = get_fourier_basis(prime).n_frequencies
        threshold = 3.0 / n_freq

        # Stream the per-epoch attribution; stack only the (n_epochs, d_mlp)
        # dominant/frac trajectories — never the (n_epochs, n_freq, d_mlp) cube.
        epochs: list[int] = []
        dom_rows: list[np.ndarray] = []
        frac_rows: list[np.ndarray] = []
        for epoch, attribution in inputs.deps.stream(
            "neuron_frequency_attribution",
            epochs=sorted(inputs.epochs),
            fields=["dominant_freq", "max_frac"],
        ):
            epochs.append(int(epoch))
            dom_rows.append(attribution["dominant_freq"])
            frac_rows.append(attribution["max_frac"])

        dominant_freq = np.stack(dom_rows, axis=0)  # (n_epochs, d_mlp)
        max_frac = np.stack(frac_rows, axis=0)  # (n_epochs, d_mlp)

        switch_counts = _compute_switch_counts(dominant_freq, max_frac, threshold)
        commitment_epochs = _compute_commitment_epochs(
            dominant_freq, max_frac, np.array(epochs), threshold
        )

        return {
            "epochs": np.array(epochs),
            "switch_counts": switch_counts,
            "commitment_epochs": commitment_epochs,
            "threshold": np.array([threshold]),
        }


def _compute_switch_counts(
    dominant_freq: np.ndarray,
    max_frac: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Count how many times each neuron changes its dominant frequency.

    Only counts transitions between committed states (above threshold).
    """
    n_epochs, d_mlp = dominant_freq.shape
    switch_counts = np.zeros(d_mlp, dtype=np.int32)
    last_freq = np.full(d_mlp, -1, dtype=np.int32)

    for t in range(n_epochs):
        committed = max_frac[t] >= threshold
        changed = committed & (last_freq >= 0) & (dominant_freq[t] != last_freq)
        switch_counts[changed] += 1
        last_freq[committed] = dominant_freq[t, committed]

    return switch_counts


def _compute_commitment_epochs(
    dominant_freq: np.ndarray,
    max_frac: np.ndarray,
    epochs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Find the epoch at which each neuron commits to its final frequency.

    A neuron is "committed" when it holds the same dominant frequency
    (above threshold) from some epoch through to the end of training.
    """
    n_epochs, d_mlp = dominant_freq.shape
    commitment_epochs = np.full(d_mlp, np.nan)
    final_freq = dominant_freq[-1]

    for n in range(d_mlp):
        if max_frac[-1, n] < threshold:
            continue

        # Walk backward to find earliest stable point
        stable_from = n_epochs - 1
        for t in range(n_epochs - 2, -1, -1):
            if max_frac[t, n] >= threshold and dominant_freq[t, n] == final_freq[n]:
                stable_from = t
            else:
                break

        commitment_epochs[n] = epochs[stable_from]

    return commitment_epochs
