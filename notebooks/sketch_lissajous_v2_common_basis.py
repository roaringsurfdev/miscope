"""Lissajous fit — Version 2: common-basis decomposition.

Tests the hypothesis: attention builds the ring (resid_post PC1/PC2 circular
mode), MLP builds the saddle direction (resid_post PC3 out-of-plane mode).

Approach: project each site's class centroids into the *shared* resid_post
PCA basis, then ask how much each site contributes to each resid_post PC at
the in-plane and out-of-plane integer frequencies.

Since resid_post = resid_pre + attn_contribution + mlp_contribution in
activation space, and only resid_pre/attn_out/resid_post are stored in
d_model=128 (mlp_out is stored as d_mlp=512 hidden activations), we recover
mlp_contribution by subtraction: mlp_contrib = resid_post - resid_pre - attn_out.

Output per variant: which site contributes each PC's integer-frequency
amplitude, as a fraction of the total resid_post amplitude at that (PC, k).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS = Path("results/modulo_addition_1layer")
OUT_DIR = Path("notebooks/exports")
OUT_DIR.mkdir(exist_ok=True)


def load_artifact(variant: str):
    path = RESULTS / variant / "artifacts" / "global_centroid_pca" / "cross_epoch.npz"
    d = np.load(path)
    return d


def reconstruct_centroids(d, site: str, epoch_idx: int) -> np.ndarray:
    """Reconstruct (n_classes, d_model) centroids from stored projections+basis+mean."""
    proj = d[f"{site}__projections"][epoch_idx]           # (n_cls, n_comp)
    basis = d[f"{site}__basis"]                           # (d_model, n_comp)
    mean = d[f"{site}__mean"]                             # (d_model,)
    return proj @ basis.T + mean


def project_into_basis(centroids: np.ndarray, basis: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project (n_cls, d_model) centroids into a PCA basis (d_model, n_comp). Returns (n_cls, n_comp)."""
    return (centroids - mean) @ basis


def fourier_coefs(values: np.ndarray, k: int, p: int) -> tuple[float, float]:
    """Return (a, b) such that values ≈ a·cos(kθ) + b·sin(kθ). Returns amplitude sqrt(a²+b²)."""
    theta = 2 * np.pi * np.arange(p) / p
    a = 2 * (values * np.cos(k * theta)).mean()
    b = 2 * (values * np.sin(k * theta)).mean()
    return float(a), float(b)


def amp_and_r2_at_k(values: np.ndarray, k: int, p: int) -> tuple[float, float]:
    """Amplitude at integer frequency k and R² of the single-freq fit."""
    a, b = fourier_coefs(values, k, p)
    theta = 2 * np.pi * np.arange(p) / p
    recon = a * np.cos(k * theta) + b * np.sin(k * theta)
    ss_tot = float(np.sum((values - values.mean()) ** 2)) + 1e-12
    ss_res = float(np.sum((values - recon) ** 2))
    return float(np.hypot(a, b)), 1.0 - ss_res / ss_tot


def fit_best_k(values: np.ndarray, p: int) -> tuple[int, float, float]:
    """Scan integer k in 1..p//2, return (k, amp, R²) that maximizes R²."""
    best = None
    for k in range(1, p // 2 + 1):
        amp, r2 = amp_and_r2_at_k(values, k, p)
        if best is None or r2 > best[2]:
            best = (k, amp, r2)
    return best  # (k, amp, r2)


def decompose_variant(variant: str, epoch_idx: int = -1) -> dict:
    """Project attn_out, mlp_contribution, resid_pre into resid_post basis, decompose per PC."""
    d = load_artifact(variant)
    epochs = d["epochs"].astype(int)
    ep = int(epochs[epoch_idx])
    n_cls = d["resid_post__projections"].shape[1]
    p = n_cls

    # Reconstruct d_model centroids for each site (d_model=128 for these three)
    rp_centroids = reconstruct_centroids(d, "resid_post", epoch_idx)
    resid_pre_centroids = reconstruct_centroids(d, "resid_pre", epoch_idx)
    attn_centroids = reconstruct_centroids(d, "attn_out", epoch_idx)
    mlp_contrib_centroids = rp_centroids - resid_pre_centroids - attn_centroids

    # resid_post's basis: (d_model, n_comp_rp)
    rp_basis = d["resid_post__basis"]
    rp_mean = d["resid_post__mean"]

    # Project each site into resid_post basis
    site_projections = {
        "resid_post_total": (rp_centroids - rp_mean) @ rp_basis,
        "resid_pre": project_into_basis(resid_pre_centroids, rp_basis, rp_mean),
        "attn_out": project_into_basis(attn_centroids, rp_basis, rp_mean),
        "mlp_contrib": project_into_basis(mlp_contrib_centroids, rp_basis, rp_mean),
    }

    # Additivity sanity check: attn + mlp + pre ≈ total (after mean centering)
    sum_sites = (
        site_projections["resid_pre"]
        + site_projections["attn_out"]
        + site_projections["mlp_contrib"]
    )
    add_err = float(np.max(np.abs(sum_sites - site_projections["resid_post_total"])))

    # Find resid_post's in-plane and out-of-plane frequencies (best-k per PC on total)
    total = site_projections["resid_post_total"]
    pc_freqs = {}  # PC index -> (k, amp_total, r2_total)
    for pc in range(min(3, total.shape[1])):
        pc_freqs[pc] = fit_best_k(total[:, pc], p)

    return {
        "variant": variant,
        "epoch": ep,
        "p": p,
        "add_err": add_err,
        "pc_freqs": pc_freqs,
        "site_projections": site_projections,
    }


def build_decomposition_table(result: dict) -> list[dict]:
    """Rows: one per (PC, site), with amp at the PC's chosen frequency."""
    p = result["p"]
    rows = []
    for pc_idx, (k, amp_total, r2_total) in result["pc_freqs"].items():
        for site_name, site_proj in result["site_projections"].items():
            if site_name == "resid_post_total":
                continue
            vals = site_proj[:, pc_idx]
            amp, r2 = amp_and_r2_at_k(vals, k, p)
            # Signed inner product with total (is this site adding or subtracting at this PC,k?)
            total_vals = result["site_projections"]["resid_post_total"][:, pc_idx]
            total_amp, _ = amp_and_r2_at_k(total_vals, k, p)
            # Fraction (signed) — sum of site Fourier vectors should equal total Fourier vector
            a_site, b_site = fourier_coefs(vals, k, p)
            a_tot, b_tot = fourier_coefs(total_vals, k, p)
            # Projection of site's (a,b) onto total's (a,b) / |total|
            proj_onto_total = (a_site * a_tot + b_site * b_tot) / (total_amp + 1e-12)
            frac_of_total = proj_onto_total / (total_amp + 1e-12)
            rows.append({
                "pc": pc_idx,
                "k": k,
                "site": site_name,
                "amp": amp,
                "r2": r2,
                "frac_of_total": frac_of_total,  # signed; sums to ~1 across sites
                "total_amp": total_amp,
                "total_r2": r2_total,
            })
    return rows


def print_table(result: dict, rows: list[dict]):
    print(f"\n=== {result['variant']} / epoch {result['epoch']} ===")
    print(f"  additivity check (max abs error): {result['add_err']:.3e}")
    print(f"  PC frequencies + R² on resid_post total:")
    for pc_idx, (k, amp, r2) in result["pc_freqs"].items():
        print(f"    PC{pc_idx + 1}: k={k:3d}  A={amp:.3f}  R²={r2:.4f}")
    print(f"\n  Per-site decomposition at each PC's frequency:")
    print(f"  {'PC':>3} {'k':>4} {'site':>15} {'amp':>8} {'R²':>8} {'frac':>8}")
    for row in rows:
        print(
            f"  PC{row['pc'] + 1:>1} {row['k']:>4} {row['site']:>15} "
            f"{row['amp']:>8.3f} {row['r2']:>8.3f} {row['frac_of_total']:>+8.3f}"
        )


def figure_decomposition(results: list[dict], out_html: Path):
    """For each variant (column), bar chart of per-site signed fraction at each PC's freq."""
    n = len(results)
    titles = []
    for r in results:
        titles.extend([
            f"{r['variant'].split('_')[-3]}/{r['variant'].split('_')[-1]} PC1 (k={r['pc_freqs'][0][0]})",
            f"PC2 (k={r['pc_freqs'][1][0]})",
            f"PC3 (k={r['pc_freqs'][2][0]})",
        ])
    fig = make_subplots(rows=n, cols=3, subplot_titles=titles)
    sites = ["resid_pre", "attn_out", "mlp_contrib"]
    colors = {"resid_pre": "#888", "attn_out": "#1f77b4", "mlp_contrib": "#d62728"}
    for row_idx, result in enumerate(results):
        rows = build_decomposition_table(result)
        for pc in range(3):
            pc_rows = [r for r in rows if r["pc"] == pc]
            xs = [r["site"] for r in pc_rows]
            ys = [r["frac_of_total"] for r in pc_rows]
            bar_colors = [colors[s] for s in xs]
            fig.add_trace(
                go.Bar(x=xs, y=ys, marker_color=bar_colors, showlegend=False),
                row=row_idx + 1, col=pc + 1,
            )
            fig.update_yaxes(range=[-0.3, 1.3], row=row_idx + 1, col=pc + 1)
    fig.update_layout(
        height=220 * n, width=1200,
        title="Per-site signed contribution fraction to resid_post at each PC's integer frequency",
    )
    fig.write_html(out_html)
    print(f"\nwrote {out_html}")


def main(variants: list[str]):
    results = [decompose_variant(v) for v in variants]
    for r in results:
        rows = build_decomposition_table(r)
        print_table(r, rows)
    figure_decomposition(results, OUT_DIR / "lissajous_v2_site_decomposition.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[
        "modulo_addition_1layer_p113_seed999_dseed598",
        "modulo_addition_1layer_p101_seed999_dseed999",
        "modulo_addition_1layer_p101_seed485_dseed999",
        "modulo_addition_1layer_p89_seed999_dseed999",
    ])
    args = parser.parse_args()
    main(args.variants)
