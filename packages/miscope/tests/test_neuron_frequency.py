"""Tests for the conformed (epoch, neuron) -> dominant-frequency helper (REQ_110D).

The helper is the single source for the neuron-frequency derivations the summary
and cross-variant engines previously each recomputed. It reads the warehouse
``neuron_frequency_attribution`` table and applies the 0->1-indexed convention in
one place; these tests pin the convention and the counting primitives against a
hand-built attribution frame, with no dependency on real artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from miscope.analysis import neuron_frequency as nf


def _attribution(monkeypatch, dominant_freq_0idx: np.ndarray, frac: np.ndarray, epochs: np.ndarray):
    """Patch the helper's table readers to return a hand-built attribution frame."""
    n_epochs, d_mlp = dominant_freq_0idx.shape
    rows = []
    for ei, epoch in enumerate(epochs):
        for n in range(d_mlp):
            rows.append(
                {
                    "epoch": int(epoch),
                    "neuron": n,
                    "frequency": int(dominant_freq_0idx[ei, n]),  # 0-indexed, as stored
                    "frac_explained": float(frac[ei, n]),
                }
            )
    frame = pd.DataFrame(rows)

    monkeypatch.setattr(nf, "_read_attribution", lambda variant: frame)
    monkeypatch.setattr(nf, "_read_aux", lambda variant: {})
    return nf.load(object())


def test_frequency_is_one_indexed(monkeypatch):
    # neuron 0 stored as 0-indexed 14 -> should surface as 15.
    dom = np.full((2, 4), 14)
    frac = np.full((2, 4), 0.9)
    epochs = np.array([0, 100])
    attr = _attribution(monkeypatch, dom, frac, epochs)
    assert attr.dominant_freq.shape == (2, 4)
    assert (attr.dominant_freq == 15).all()
    assert np.allclose(attr.frac_explained, 0.9)


def test_specialized_and_committed_frequencies(monkeypatch):
    # 4 neurons: freqs (0idx) 14,14,28,28; only the first three above 0.7.
    dom = np.array([[14, 14, 28, 28]])
    frac = np.array([[0.9, 0.8, 0.75, 0.4]])
    attr = _attribution(monkeypatch, dom, frac, np.array([0]))
    # 1-indexed: 15 (x2 specialized), 29 (x1 specialized; the 0.4 neuron excluded)
    assert attr.specialized_frequencies(0, threshold=0.7) == [15, 29]
    assert attr.frequency_counts(0, threshold=0.7) == {15: 2, 29: 1}
    # population floor 0.5 of d_mlp(4)=2 -> only freq 15 (count 2) qualifies
    assert attr.committed_frequencies(0, threshold=0.7, population_floor=0.5) == [15]
    assert attr.specialized_count(0, threshold=0.7) == 3


def test_recompute_commitment_epochs(monkeypatch):
    # 2 neurons over 3 epochs. Neuron 0 stable on freq 14 from epoch 1; neuron 1
    # never specialized at final -> NaN.
    dom = np.array([[14, 28], [14, 28], [14, 28]])
    frac = np.array([[0.3, 0.2], [0.9, 0.2], [0.9, 0.2]])
    epochs = np.array([0, 100, 200])
    attr = _attribution(monkeypatch, dom, frac, epochs)
    commitment = attr.recompute_commitment_epochs(threshold=0.7)
    assert commitment[0] == 100  # specialized from epoch index 1 (epoch 100)
    assert np.isnan(commitment[1])


def test_classify_band():
    assert nf.classify_band(1, 113) == "low"
    assert nf.classify_band(60, 113) == "high"
    assert nf.classify_band(40, 113) == "mid"
