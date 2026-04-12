"""Tests for REQ_055: Attention Head Phase Relationship Analysis."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import plotly.graph_objects as go
import torch

from miscope.analysis.analyzers.attention_fourier import (
    AttentionFourierAnalyzer,
    _qk_block_norms,
    _v_band_norms,
)
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.protocols import ActivationContext
from miscope.visualization.renderers.attention_fourier import (
    render_head_alignment_trajectory,
    render_qk_freq_heatmap,
    render_v_freq_heatmap,
)

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

P = 11  # small prime for fast tests
N_HEADS = 4
D_MODEL = 32
D_HEAD = 8
N_FREQ = P // 2  # = 5


def _make_fourier_basis(p: int) -> torch.Tensor:
    """Build a normalized Fourier basis matching get_fourier_basis(p)."""
    rows = [torch.ones(p)]
    for k in range(1, p // 2 + 1):
        theta = torch.arange(p) * 2 * torch.pi * k / p
        rows.append(torch.sin(theta))
        rows.append(torch.cos(theta))
    basis = torch.stack(rows)
    return basis / basis.norm(dim=-1, keepdim=True)


def _make_model(
    p: int = P,
    n_heads: int = N_HEADS,
    d_model: int = D_MODEL,
    d_head: int = D_HEAD,
    *,
    seed: int = 42,
) -> MagicMock:
    """Build a mock HookedTransformer with random weight tensors."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    W_E = torch.randn(p + 1, d_model, generator=rng)
    W_Q = torch.randn(n_heads, d_model, d_head, generator=rng)
    W_K = torch.randn(n_heads, d_model, d_head, generator=rng)
    W_V = torch.randn(n_heads, d_model, d_head, generator=rng)

    model = MagicMock()
    model.embed.W_E = W_E
    block = MagicMock()
    block.attn.W_Q = W_Q
    block.attn.W_K = W_K
    block.attn.W_V = W_V
    model.blocks = [block]
    return model


def _make_context(p: int = P) -> dict:
    return {"fourier_basis": _make_fourier_basis(p)}


def _make_epoch_data(n_heads: int = N_HEADS, n_freq: int = N_FREQ) -> dict:
    """Build minimal per-epoch artifact data."""
    rng = np.random.default_rng(0)
    raw = rng.random((n_heads, n_freq)).astype(np.float32)
    qk = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    raw2 = rng.random((n_heads, n_freq)).astype(np.float32)
    v = (raw2 / raw2.sum(axis=1, keepdims=True)).astype(np.float32)
    return {"qk_freq_norms": qk, "v_freq_norms": v}


def _make_stacked_data(n_epochs: int = 8, n_heads: int = N_HEADS, n_freq: int = N_FREQ) -> dict:
    """Build minimal stacked (multi-epoch) artifact data."""
    rng = np.random.default_rng(1)
    raw = rng.random((n_epochs, n_heads, n_freq)).astype(np.float32)
    qk = (raw / raw.sum(axis=2, keepdims=True)).astype(np.float32)
    return {
        "epochs": np.arange(n_epochs) * 100,
        "qk_freq_norms": qk,
        "v_freq_norms": qk.copy(),
    }


# ---------------------------------------------------------------------------
# AttentionFourierAnalyzer — unit tests
# ---------------------------------------------------------------------------


class TestAttentionFourierAnalyzer:
    def test_returns_expected_keys(self):
        model = _make_model()
        context = _make_context()
        analyzer = AttentionFourierAnalyzer()
        result = analyzer.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(model, None, None),  # type: ignore
                probe=None,  # type: ignore
                analysis_params=context,
            )
        )  # type: ignore[arg-type]
        assert "qk_freq_norms" in result
        assert "v_freq_norms" in result

    def test_output_shapes(self):
        model = _make_model()
        context = _make_context()
        analyzer = AttentionFourierAnalyzer()
        result = analyzer.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(model, None, None),  # type: ignore
                probe=None,  # type: ignore
                analysis_params=context,
            )
        )  # type: ignore[arg-type]
        assert result["qk_freq_norms"].shape == (N_HEADS, N_FREQ)
        assert result["v_freq_norms"].shape == (N_HEADS, N_FREQ)

    def test_fractions_sum_to_one(self):
        model = _make_model()
        context = _make_context()
        analyzer = AttentionFourierAnalyzer()
        result = analyzer.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(model, None, None),  # type: ignore
                probe=None,  # type: ignore
                analysis_params=context,
            )
        )  # type: ignore[arg-type]
        qk_sums = result["qk_freq_norms"].sum(axis=1)
        v_sums = result["v_freq_norms"].sum(axis=1)
        np.testing.assert_allclose(qk_sums, np.ones(N_HEADS), atol=1e-5)
        np.testing.assert_allclose(v_sums, np.ones(N_HEADS), atol=1e-5)

    def test_values_are_nonnegative(self):
        model = _make_model()
        context = _make_context()
        analyzer = AttentionFourierAnalyzer()
        result = analyzer.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(model, None, None),  # type: ignore
                probe=None,  # type: ignore
                analysis_params=context,
            )
        )  # type: ignore[arg-type]
        assert (result["qk_freq_norms"] >= 0).all()
        assert (result["v_freq_norms"] >= 0).all()

    def test_output_dtype_is_float32(self):
        model = _make_model()
        context = _make_context()
        analyzer = AttentionFourierAnalyzer()
        result = analyzer.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(model, None, None),  # type: ignore
                probe=None,  # type: ignore
                analysis_params=context,
            )
        )  # type: ignore[arg-type]
        assert result["qk_freq_norms"].dtype == np.float32
        assert result["v_freq_norms"].dtype == np.float32

    def test_monofrequency_head_concentrates_on_one_freq(self):
        """A head whose Q and K are both aligned to frequency k should produce
        a QK^T dominated by that frequency."""
        p = P
        # n_freq = p // 2
        F, _ = _make_fourier_basis_and_names(p)
        k = 2  # target frequency (1-indexed)
        sin_k = F[2 * k - 1]  # (p,)
        cos_k = F[2 * k]  # (p,)

        # Construct a pure-frequency-k weight: Q = [sin_k | cos_k, zeros...]
        d_model = p
        d_head = 2
        W_E = torch.eye(p + 1, d_model)  # identity: W_E_tok = I_p
        W_Q_h = torch.zeros(d_model, d_head)
        W_K_h = torch.zeros(d_model, d_head)
        W_Q_h[:, 0] = sin_k
        W_K_h[:, 0] = sin_k
        W_Q_h[:, 1] = cos_k
        W_K_h[:, 1] = cos_k

        model = MagicMock()
        model.embed.W_E = W_E
        block = MagicMock()
        block.attn.W_Q = W_Q_h.unsqueeze(0)  # (1, d_model, d_head)
        block.attn.W_K = W_K_h.unsqueeze(0)
        block.attn.W_V = torch.zeros(1, d_model, d_head)
        model.blocks = [block]

        context = {"fourier_basis": F}
        analyzer = AttentionFourierAnalyzer()
        result = analyzer.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(model, None, None),  # type: ignore
                probe=None,  # type: ignore
                analysis_params=context,
            )
        )  # type: ignore[arg-type]
        # Dominant frequency for head 0 should be k (1-indexed → index k-1)
        dominant = int(result["qk_freq_norms"][0].argmax()) + 1
        assert dominant == k, f"Expected dominant freq {k}, got {dominant}"


def _make_fourier_basis_and_names(p: int):
    from miscope.analysis.library.fourier import get_fourier_basis

    return get_fourier_basis(p)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestBlockNorms:
    def test_returns_correct_length(self):
        QK_fourier = torch.randn(P + 1, P + 1)
        result = _qk_block_norms(QK_fourier, N_FREQ)
        assert result.shape == (N_FREQ,)

    def test_nonnegative(self):
        QK_fourier = torch.randn(P + 1, P + 1)
        result = _qk_block_norms(QK_fourier, N_FREQ)
        assert (result >= 0).all()


class TestBandNorms:
    def test_returns_correct_length(self):
        V_fourier = torch.randn(P + 1, D_HEAD)
        result = _v_band_norms(V_fourier, N_FREQ)
        assert result.shape == (N_FREQ,)

    def test_nonnegative(self):
        V_fourier = torch.randn(P + 1, D_HEAD)
        result = _v_band_norms(V_fourier, N_FREQ)
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


class TestRenderQkFreqHeatmap:
    def test_returns_figure(self):
        fig = render_qk_freq_heatmap(_make_epoch_data(), epoch=1000)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self):
        fig = render_qk_freq_heatmap(_make_epoch_data(), epoch=1000)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_heatmap_shape_matches_data(self):
        data = _make_epoch_data(n_heads=3, n_freq=5)
        fig = render_qk_freq_heatmap(data, epoch=500)
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        z = np.array(heatmap.z)
        assert z.shape == (3, 5)

    def test_custom_title_applied(self):
        fig = render_qk_freq_heatmap(_make_epoch_data(), epoch=0, title="Test Title")
        assert "Test Title" in fig.layout.title.text  # type: ignore[attr-defined]


class TestRenderVFreqHeatmap:
    def test_returns_figure(self):
        fig = render_v_freq_heatmap(_make_epoch_data(), epoch=1000)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self):
        fig = render_v_freq_heatmap(_make_epoch_data(), epoch=1000)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)


class TestRenderHeadAlignmentTrajectory:
    def test_returns_figure(self):
        fig = render_head_alignment_trajectory(_make_stacked_data())
        assert isinstance(fig, go.Figure)

    def test_has_scatter_traces(self):
        fig = render_head_alignment_trajectory(_make_stacked_data())
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) > 0

    def test_trace_count_matches_heads(self):
        # 2 rows (dominant freq + max frac), N_HEADS lines per row
        stacked = _make_stacked_data(n_heads=3)
        fig = render_head_alignment_trajectory(stacked)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 2 * 3  # 2 subplots × n_heads

    def test_custom_title_applied(self):
        fig = render_head_alignment_trajectory(_make_stacked_data(), title="Custom")
        assert "Custom" in fig.layout.title.text  # type: ignore[attr-defined]
