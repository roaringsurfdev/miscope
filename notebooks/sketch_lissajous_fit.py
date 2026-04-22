"""Lissajous / Halo fit for class-centroid PCA torus.

Hypothesis (2026-04-20): the class-centroid ring/torus in PC1/PC2/PC3 is a
Lissajous orbit of the linearized dynamics at a saddle-center-center
equilibrium (Ross, CRTBP). In modular addition the natural "time"
parameter is the *class angle* θ_i = 2π i / p — each class sits at a phase
of the orbit.

Linearized form (Ross lecture):
    x(θ) = -A_x cos(ω_p θ + φ)
    y(θ) =  κ A_x sin(ω_p θ + φ)    κ = (ω_p² + 1 + 2 c₂) / (2 ω_p)
    z(θ) =  A_z sin(ω_v θ + ψ)

Here ω_p, ω_v are integer Fourier frequencies in (Z/pZ). We expect PC1/PC2
to carry one frequency k₁ (the dominant committed frequency pair) and PC3
to carry a second frequency k₂ — the "out-of-plane" mode.

Stage 1: fit at final (grokked) epoch for p113/s999/ds598. Report κ, k₁, k₂,
residual. Cross-check (k₁, k₂) against the dominant_frequencies artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse

RESULTS = Path("results/modulo_addition_1layer")
OUT_DIR = Path("notebooks/exports")
OUT_DIR.mkdir(exist_ok=True)


def load_centroid_pca(variant: str, site: str):
    path = RESULTS / variant / "artifacts" / "global_centroid_pca" / "cross_epoch.npz"
    d = np.load(path)
    return d["epochs"].astype(int), d[f"{site}__projections"], d[f"{site}__explained_variance_ratio"]


def load_dominant_frequencies(variant: str, epoch: int):
    """Returns dict {k -> magnitude} folding (sin k, cos k) into one entry.

    Array layout from get_fourier_basis(p): index 0 = Constant,
    index 2k-1 = sin k, index 2k = cos k. Combined magnitude per k
    = sqrt(sin_coef^2 + cos_coef^2).
    """
    path = RESULTS / variant / "artifacts" / "dominant_frequencies" / f"epoch_{epoch:05d}.npz"
    coeffs = np.load(path)["coefficients"]
    n_freqs = (len(coeffs) - 1) // 2
    out = {0: float(abs(coeffs[0]))}
    for k in range(1, n_freqs + 1):
        sin_c = float(coeffs[2 * k - 1])
        cos_c = float(coeffs[2 * k])
        out[k] = float(np.hypot(sin_c, cos_c))
    return out


def fit_integer_frequency(values: np.ndarray, p: int, exclude: int | None = None):
    """Best integer frequency k in 1..p-1 for a real signal of length p.

    Returns (k, amplitude, phase, explained). Excludes k = exclude and its
    conjugate p-k to allow searching for an orthogonal mode.
    """
    n = len(values)
    assert n == p
    theta = 2 * np.pi * np.arange(p) / p
    total_var = float(np.sum((values - values.mean()) ** 2)) + 1e-12
    best = None
    for k in range(1, p // 2 + 1):
        if exclude is not None and k in {exclude, p - exclude}:
            continue
        c = np.cos(k * theta)
        s = np.sin(k * theta)
        # Orthogonal basis — directly compute Fourier coefficients
        a = 2 * (values * c).mean()
        b = 2 * (values * s).mean()
        amp = float(np.hypot(a, b))
        phase = float(np.arctan2(b, a))
        recon = a * c + b * s
        ss_res = float(np.sum((values - recon) ** 2))
        r2 = 1.0 - ss_res / total_var
        if best is None or r2 > best[3]:
            best = (k, amp, phase, r2)
    return best


def fit_lissajous_at_epoch(proj_at_epoch: np.ndarray, p: int):
    """proj_at_epoch: (p, n_pc). Fit Lissajous to first three PCs."""
    x = proj_at_epoch[:, 0]
    y = proj_at_epoch[:, 1]
    z = proj_at_epoch[:, 2]

    # Fit integer frequency for each PC independently, then cross-check
    k_x, amp_x, phase_x, r2_x = fit_integer_frequency(x, p)
    k_y, amp_y, phase_y, r2_y = fit_integer_frequency(y, p)
    k_z, amp_z, phase_z, r2_z = fit_integer_frequency(z, p, exclude=k_x)

    # Lissajous form assumes x and y carry the SAME frequency ω_p
    # Confirm k_x == k_y; if so, κ = amp_y / amp_x and phase offset → φ
    same_inplane = k_x == k_y
    kappa = amp_y / amp_x if same_inplane and amp_x > 1e-9 else None

    # In-plane phase lock check: Ross form gives x = -A_x cos(ω_p θ + φ),
    # y =  κ A_x sin(ω_p θ + φ). The phase of -cos is π, of sin is π/2.
    # So y_phase - x_phase should equal -π/2 (mod 2π) if same_inplane.
    phase_offset = None
    if same_inplane:
        phase_offset = (phase_y - phase_x + np.pi) % (2 * np.pi) - np.pi

    return dict(
        k_x=k_x, amp_x=amp_x, phase_x=phase_x, r2_x=r2_x,
        k_y=k_y, amp_y=amp_y, phase_y=phase_y, r2_y=r2_y,
        k_z=k_z, amp_z=amp_z, phase_z=phase_z, r2_z=r2_z,
        same_inplane=same_inplane, kappa=kappa, phase_offset=phase_offset,
    )


def format_fit(fit: dict) -> str:
    lines = [
        f"  PC1:  k={fit['k_x']:3d}  A={fit['amp_x']:.3f}  R²={fit['r2_x']:.4f}",
        f"  PC2:  k={fit['k_y']:3d}  A={fit['amp_y']:.3f}  R²={fit['r2_y']:.4f}",
        f"  PC3:  k={fit['k_z']:3d}  A={fit['amp_z']:.3f}  R²={fit['r2_z']:.4f}",
        f"  same in-plane freq: {fit['same_inplane']}",
    ]
    if fit["same_inplane"]:
        lines += [
            f"  κ = A_y/A_x = {fit['kappa']:.4f}",
            f"  phase offset (should be ≈ -π/2 = {-np.pi/2:.4f}): {fit['phase_offset']:.4f}",
        ]
    return "\n".join(lines)


def panel_fit_check(proj_at_epoch: np.ndarray, p: int, fit: dict, epoch: int, out_html: Path, site: str):
    theta = 2 * np.pi * np.arange(p) / p
    recon_x = fit["amp_x"] * np.cos(fit["k_x"] * theta - fit["phase_x"])
    recon_y = fit["amp_y"] * np.cos(fit["k_y"] * theta - fit["phase_y"])
    recon_z = fit["amp_z"] * np.cos(fit["k_z"] * theta - fit["phase_z"])
    x, y, z = proj_at_epoch[:, 0], proj_at_epoch[:, 1], proj_at_epoch[:, 2]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            f"PC1 vs class (k={fit['k_x']}, R²={fit['r2_x']:.3f})",
            f"PC2 vs class (k={fit['k_y']}, R²={fit['r2_y']:.3f})",
            f"PC3 vs class (k={fit['k_z']}, R²={fit['r2_z']:.3f})",
            "PC1 vs PC2 (in-plane)", "PC1 vs PC3", "PC2 vs PC3",
        ),
    )
    cls = np.arange(p)
    for row, col, data, recon in [
        (1, 1, x, recon_x), (1, 2, y, recon_y), (1, 3, z, recon_z),
    ]:
        fig.add_trace(go.Scatter(x=cls, y=data, mode="markers", name="data",
                                 marker=dict(size=4, color="steelblue")), row, col)
        fig.add_trace(go.Scatter(x=cls, y=recon, mode="lines", name="fit",
                                 line=dict(color="crimson", width=1.5)), row, col)
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers+lines", marker=dict(size=3), line=dict(width=1), showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=z, mode="markers+lines", marker=dict(size=3), line=dict(width=1), showlegend=False), 2, 2)
    fig.add_trace(go.Scatter(x=y, y=z, mode="markers+lines", marker=dict(size=3), line=dict(width=1), showlegend=False), 2, 3)
    fig.update_layout(
        height=720, width=1200, showlegend=False,
        title=f"{site} — epoch {epoch} — Lissajous fit",
    )
    fig.write_html(out_html)
    print(f"wrote {out_html}")


def amplitude_at_k(proj_ep: np.ndarray, k: int, p: int) -> float:
    """Per-frequency orbit amplitude, Frobenius over all PCs."""
    theta = 2 * np.pi * np.arange(p) / p
    c = np.cos(k * theta)
    s = np.sin(k * theta)
    a = 2 * (proj_ep * c[:, None]).mean(axis=0)
    b = 2 * (proj_ep * s[:, None]).mean(axis=0)
    return float(np.sqrt((a ** 2 + b ** 2).sum()))


def plot_per_freq_trajectories(
    epochs, proj, freqs, p, grok_ep, variant, out_html, site, export=True
):
    traces = []
    for k in freqs:
        amps = np.array([amplitude_at_k(proj[i], k, p) for i in range(len(epochs))])
        traces.append(go.Scatter(x=epochs, y=amps, mode="lines", name=f"k={k}"))
    fig = go.Figure(traces)
    if grok_ep is not None:
        fig.add_vline(x=grok_ep, line_dash="dash", line_color="gray",
                      annotation_text=f"grok={grok_ep}", annotation_position="top left")
    fig.update_layout(
        title=f"{variant} — {site} — per-freq orbit amplitude (Frobenius over PCs)",
        xaxis_title="epoch", yaxis_title="amplitude",
        height=500, width=1000,
    )
    if export:
        fig.write_html(out_html)
        print(f"wrote {out_html}")
    else:
        fig.show()


def main(variant: str, grok_ep: int | None = None, site: str = "resid_post", export=True):
    epochs, proj, evr = load_centroid_pca(variant, site)
    _, n_cls, _ = proj.shape
    p = n_cls

    print(f"\n=== {variant} / {site} ===")
    print(f"  p={p}, n_epochs={len(epochs)}, EVR first 4: {evr[:4].round(4).tolist()}")

    final_ep = int(epochs[-1])
    anchors = [0, final_ep]
    if grok_ep is not None:
        anchors = sorted(set([0, max(0, grok_ep - 2000), grok_ep, final_ep]))
    for target in anchors:
        idx = int(np.argmin(np.abs(epochs - target)))
        ep = int(epochs[idx])
        fit = fit_lissajous_at_epoch(proj[idx], p)
        print(f"\nepoch {ep}:")
        print(format_fit(fit))

    # Detailed fit at final epoch + figure
    idx = -1
    fit = fit_lissajous_at_epoch(proj[idx], p)
    ep = int(epochs[idx])
    panel_fit_check(proj[idx], p, fit, ep, OUT_DIR / f"lissajous_fit_{variant}_{site}_ep{ep}.html", site)

    # Compare to dominant_frequencies (W_E spectrum) at same epoch
    dom = load_dominant_frequencies(variant, ep)
    top_k = sorted(dom.keys(), key=lambda k: -dom[k])[:8]
    print(f"\ndominant_frequencies (embedding W_E) top-8 at epoch {ep}:")
    for k in top_k:
        print(f"  k={k:3d}  |c|={dom[k]:.3f}")
    print(
        f"  Lissajous fit picked k_inplane={fit['k_x']}, k_out={fit['k_z']} "
        f"— match top-2 embedding freqs: {sorted([top_k[0], top_k[1]]) == sorted([fit['k_x'], fit['k_z']])}"
    )

    # Per-frequency orbit amplitude trajectory for top-6 dominant freqs
    freqs_track = top_k[:6]
    plot_per_freq_trajectories(
        epochs, proj, freqs_track, p, grok_ep, variant,
        OUT_DIR / f"lissajous_per_freq_amp_{variant}_{site}.html", site,
        export=export
    )

    # Tabular first-mover view: which freq is above threshold first?
    print("\nper-freq amplitude at select epochs:")
    header = "ep    " + "  ".join(f"k={k:3d}" for k in freqs_track)
    print(header)
    step = max(1, len(epochs) // 12)
    for i in range(0, len(epochs), step):
        ep = int(epochs[i])
        row = f"{ep:5d} " + "  ".join(f"{amplitude_at_k(proj[i], k, p):6.2f}" for k in freqs_track)
        print(row)
    # And the final
    print(f"{int(epochs[-1]):5d} " + "  ".join(
        f"{amplitude_at_k(proj[-1], k, p):6.2f}" for k in freqs_track))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="modulo_addition_1layer_p113_seed999_dseed598")
    parser.add_argument("--grok", type=int, default=None)
    parser.add_argument("--site", default="resid_post",
                        choices=["resid_pre", "attn_out", "mlp_out", "resid_post"])
    args = parser.parse_args()
    main(args.variant, args.grok, args.site)
