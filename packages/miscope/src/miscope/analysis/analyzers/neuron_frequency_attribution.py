"""REQ_141 (bucket-1): per-epoch neuron→frequency attribution analyzer.

The conformed ``(epoch, neuron) → dominant-frequency`` dimension is a **per-epoch
fact**: a pure function of one epoch's ``activation_frequency_norm``. It was
previously back-filled by ``neuron_dynamics``, a *cross-epoch* analyzer that
stacked every epoch's ``(n_freq, d_mlp)`` norm matrix into an
``(n_epochs, n_freq, d_mlp)`` cube only to argmax over the frequency axis — the
"per-epoch fact trapped in a cross-epoch analyzer" the REQ_141 litmus test names.

This analyzer emits that fact natively, one epoch at a time: read the small
``(n_freq, d_mlp)`` ``mlp_out_freq_norm`` matrix for the epoch (produced once by
``activation_frequency_norm``) and take the per-neuron argmax / max. The columnar
materializer routes ``dominant_freq`` / ``max_frac`` to
the ``neuron_frequency_attribution`` semantic table (see ``mapping_semantic``), so
the conformed dimension (:mod:`miscope.analysis.neuron_frequency`) reads identical
values — only the producer changed. ``neuron_dynamics`` keeps just its genuine
cross-epoch tail (switch counts, commitment epochs), streaming this analyzer's
per-epoch output instead of rebuilding the cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ArtifactInput, ResolvedInputs
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

SPEC = AnalyzerSpec(
    name="neuron_frequency_attribution",
    output_scope="per_epoch",
    inputs=(ArtifactInput("activation_frequency_norm"),),
    outputs=(
        F.columnar(
            "dominant_freq",
            "int64",
            ("variant", "epoch", "neuron"),
            "Dominant frequency index (0-based argmax) per neuron per epoch.",
        ),
        F.columnar(
            "max_frac",
            "float64",
            ("variant", "epoch", "neuron"),
            "Fraction of Fourier norm in the dominant frequency per neuron per epoch.",
        ),
    ),
)


@register_analyzer(SPEC)
class NeuronFrequencyAttributionAnalyzer:
    """Per-epoch neuron dominant-frequency / max-frac attribution (REQ_141)."""

    name = "neuron_frequency_attribution"
    requires = ["activation_frequency_norm"]
    depends_on = "activation_frequency_norm"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Argmax/max over the frequency axis of this epoch's norm matrix."""
        assert inputs.deps is not None
        assert inputs.epoch is not None

        projection = inputs.deps.load_epoch(
            "activation_frequency_norm", inputs.epoch, fields=["mlp_out_freq_norm"]
        )
        norm = projection["mlp_out_freq_norm"]  # (n_freq, d_mlp)

        # Identical reduction to the old neuron_dynamics inner loop (argmax/max over
        # the frequency axis), now per epoch — no (n_epochs, n_freq, d_mlp) stack.
        return {
            "dominant_freq": np.argmax(norm, axis=0),
            "max_frac": np.max(norm, axis=0),
        }
