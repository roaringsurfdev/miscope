"""Per-group kink detection sketch — probe for the bold claim.

Bold claim: each frequency group traces its own tube with its own saddle
transitions; global MLP rolling PR₃ peaks are a *summary* of per-group
direction changes.

Falsifier of the collective-saddle reading:
    COLLECTIVE  → per-group PR₃ peaks cluster at global peak epochs
    STAGGERED   → per-group peaks distributed, most *not* coincident

For each frequency group in a variant:
    1. centroid(epoch, group) = mean of member neurons' W_in columns
    2. fit PCA on the group's centroid sequence → local arc basis
    3. project centroid sequence through local basis
    4. rolling PR₃ in local PC space → kink timeline for that group
    5. peak-detect the kink timeline

Intentionally small and notebook-compatible. Reuses module primitives where
available; re-implements centroid extraction inline to avoid pulling
parameter_trajectory_pca.ipynb into the import graph.

Usage:
    from notebooks.sketch_per_group_kinks import run_sketch
    run_sketch(prime=113, seed=999, data_seed=598, variant_label="canon")
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

from miscope import load_family
from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.visualization.renderers.dimensionality_dynamics import (
    _compute_rolling_trajectory_metrics,
)


def load_group_centroids(variant):
    """Return (epochs, centroids (G, N, d_model), group_freqs)."""
    loader = ArtifactLoader(str(variant.variant_dir / "artifacts"))
    gdata = loader.load_cross_epoch("neuron_group_pca")
    neuron_group_idx = gdata["neuron_group_idx"]
    group_freqs = gdata["group_freqs"]
    n_groups = len(group_freqs)

    epochs = sorted(loader.get_epochs("parameter_snapshot"))
    d_model, d_mlp = loader.load_epoch("parameter_snapshot", epochs[0])["W_in"].shape

    members = [np.where(neuron_group_idx == g)[0] for g in range(n_groups)]
    centroids = np.zeros((n_groups, len(epochs), d_model))
    for i, ep in enumerate(epochs):
        W_in = loader.load_epoch("parameter_snapshot", ep)["W_in"]  # (d_model, d_mlp)
        for g, mem in enumerate(members):
            if len(mem):
                centroids[g, i] = W_in[:, mem].mean(axis=1)

    return np.array(epochs), centroids, group_freqs


def load_transient_cohort_centroids(variant):
    """Peak-cohort centroid trajectory for every ever-qualified frequency.

    The cohort is fixed at the peak epoch (neurons committed to the frequency
    then). Their mean W_in column is tracked across all epochs. A transient
    frequency's cohort moves in weight space even as its members un-specialize
    — the direction change at abandonment is the saddle-crossing signature.

    Returns dict:
        epochs      (n_epochs,)
        centroids   (n_transient, n_epochs, d_model)   — per-cohort centroid
        freqs       (n_transient,)                     — 0-indexed frequency
        is_final    (n_transient,)  bool               — in final committed set
        peak_epoch  (n_transient,)
        peak_count  (n_transient,)
        homeless_count (n_transient,)
    """
    loader = ArtifactLoader(str(variant.variant_dir / "artifacts"))
    tf = loader.load_cross_epoch("transient_frequency")
    freqs = tf["ever_qualified_freqs"]
    if len(freqs) == 0:
        return None

    flat = tf["peak_members_flat"]
    offsets = tf["peak_members_offsets"]
    cohorts = [flat[offsets[i] : offsets[i + 1]] for i in range(len(freqs))]

    epochs = sorted(loader.get_epochs("parameter_snapshot"))
    d_model, _ = loader.load_epoch("parameter_snapshot", epochs[0])["W_in"].shape

    centroids = np.zeros((len(freqs), len(epochs), d_model))
    for i, ep in enumerate(epochs):
        W_in = loader.load_epoch("parameter_snapshot", ep)["W_in"]
        for c, cohort in enumerate(cohorts):
            if len(cohort):
                centroids[c, i] = W_in[:, cohort].mean(axis=1)

    return {
        "epochs": np.array(epochs),
        "centroids": centroids,
        "freqs": freqs,
        "is_final": tf["is_final"],
        "peak_epoch": tf["peak_epoch"],
        "peak_count": tf["peak_count"],
        "homeless_count": tf["homeless_count"],
    }


def _smooth_ma(X, k):
    """Moving-average smooth along axis=0 with reflection padding. Returns same shape."""
    if k <= 1:
        return X
    pad = k // 2
    Xp = np.concatenate([X[pad - 1 :: -1], X, X[-1 : -pad - 1 : -1]], axis=0)
    kernel = np.ones(k) / k
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        out[:, j] = np.convolve(Xp[:, j], kernel, mode="valid")[: X.shape[0]]
    return out


def per_group_rolling_pr3(centroids, window=10, smooth_k=1):
    """Rolling PR₃ on each group's centroid arc in its own local PCA basis.

    For each group, (optionally) smooths the centroid sequence, centers &
    projects it through a PCA basis fit on the smoothed sequence, then
    applies _compute_rolling_trajectory_metrics.

    Returns: (G, n_epochs) rolling PR₃ array.
    """
    G, N, _ = centroids.shape
    out = np.full((G, N), np.nan)
    for g in range(G):
        X = _smooth_ma(centroids[g], smooth_k)
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt.T
        pr3, _ = _compute_rolling_trajectory_metrics(proj, window=window)
        out[g] = pr3
    return out


def sweep_prominence(pr3_per_group, epochs, global_peaks, prominences, tolerance=500):
    """Count per-group peaks and coincidence rate across prominence values."""
    rows = []
    for prom in prominences:
        pgp = detect_per_group_peaks(pr3_per_group, epochs, prominence=prom)
        cov = classify_coincidence(pgp, global_peaks, tolerance=tolerance)
        n_total = cov["total_coincident"] + cov["total_solo"]
        pct = 100 * cov["total_coincident"] / n_total if n_total else 0.0
        # per-global-peak participation
        participation = {
            gp: sum(
                any(abs(p - gp) <= tolerance for p in row) for row in pgp
            )
            for gp in global_peaks
        }
        rows.append(
            {
                "prominence": prom,
                "peaks_per_group": [len(r) for r in pgp],
                "n_coincident": cov["total_coincident"],
                "n_solo": cov["total_solo"],
                "pct_coincident": pct,
                "participation": participation,
            }
        )
    return rows


def detect_per_group_peaks(pr3_arr, epochs, prominence=0.02):
    """Find peaks in each group's rolling PR₃ timeline.

    Returns list[list[int]] of peak epochs per group.
    """
    peaks_by_group = []
    for g in range(pr3_arr.shape[0]):
        y = pr3_arr[g]
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            peaks_by_group.append([])
            continue
        idx, _ = find_peaks(y[mask], prominence=prominence)
        valid_epochs = epochs[mask]
        peaks_by_group.append([int(valid_epochs[i]) for i in idx])
    return peaks_by_group


def load_global_mlp_peaks(variant, window=10, prominence=0.05):
    """Load mlp__projections and return (epochs, pr3, peak_epochs)."""
    loader = ArtifactLoader(str(variant.variant_dir / "artifacts"))
    pt = loader.load_cross_epoch("parameter_trajectory")
    mlp_proj = pt["mlp__projections"]  # (n_epochs, k)
    epochs = pt["epochs"]
    pr3, _ = _compute_rolling_trajectory_metrics(mlp_proj, window=window)
    mask = ~np.isnan(pr3)
    idx, _ = find_peaks(pr3[mask], prominence=prominence)
    valid_epochs = epochs[mask]
    peak_epochs = [int(valid_epochs[i]) for i in idx]
    return epochs, pr3, peak_epochs


def classify_coincidence(per_group_peaks, global_peaks, tolerance=500):
    """For each per-group peak, label: coincident (within tolerance of any global peak) or solo.

    Returns dict with counts + per-group breakdown.
    """
    rows = []
    total_coincident = 0
    total_solo = 0
    for g, peaks in enumerate(per_group_peaks):
        coincident = []
        solo = []
        for p in peaks:
            if any(abs(p - gp) <= tolerance for gp in global_peaks):
                coincident.append(p)
            else:
                solo.append(p)
        rows.append({"group": g, "coincident": coincident, "solo": solo})
        total_coincident += len(coincident)
        total_solo += len(solo)
    return {
        "per_group": rows,
        "total_coincident": total_coincident,
        "total_solo": total_solo,
        "n_global_peaks": len(global_peaks),
    }


def plot_overlay(
    epochs,
    pr3_per_group,
    group_freqs,
    global_epochs,
    global_pr3,
    global_peaks,
    per_group_peaks,
    label,
):
    """Overlay per-group rolling PR₃ + global MLP PR₃ with peak markers."""
    fig = go.Figure()

    # Global reference
    fig.add_trace(
        go.Scatter(
            x=global_epochs,
            y=global_pr3,
            mode="lines",
            name="global MLP PR₃",
            line=dict(color="black", width=2.5, dash="dot"),
        )
    )
    for gp in global_peaks:
        fig.add_vline(
            x=gp,
            line_dash="dash",
            line_color="rgba(0,0,0,0.35)",
            annotation_text=f"global {gp}",
            annotation_position="top",
        )

    # Per-group traces
    cmap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for g, freq in enumerate(group_freqs):
        color = cmap[g % len(cmap)]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=pr3_per_group[g],
                mode="lines",
                name=f"freq {int(freq) + 1} group",
                line=dict(color=color, width=1.6),
            )
        )
        for p in per_group_peaks[g]:
            fig.add_trace(
                go.Scatter(
                    x=[p],
                    y=[pr3_per_group[g][np.where(epochs == p)[0][0]]],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="triangle-up"),
                    showlegend=False,
                    hovertemplate=f"freq {int(freq) + 1} | peak at {p}<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"{label} | per-group centroid rolling PR₃ (local PCA basis) vs global MLP PR₃",
        xaxis_title="epoch",
        yaxis_title="rolling PR₃ (window=10)",
        template="plotly_white",
        height=520,
    )
    return fig


def run_sketch(
    prime=113,
    seed=999,
    data_seed=598,
    variant_label="canon",
    family_name="modulo_addition_1layer",
    window=10,
    smooth_k=5,
    per_group_prominence=0.05,
    tolerance=500,
    sweep=(0.02, 0.05, 0.08, 0.12),
):
    family = load_family(family_name)
    variant = family.get_variant(prime=prime, seed=seed, data_seed=data_seed)

    epochs, centroids, group_freqs = load_group_centroids(variant)
    pr3_per_group = per_group_rolling_pr3(centroids, window=window, smooth_k=smooth_k)
    per_group_peaks = detect_per_group_peaks(
        pr3_per_group, epochs, prominence=per_group_prominence
    )

    global_epochs, global_pr3, global_peaks = load_global_mlp_peaks(variant, window=window)
    coincidence = classify_coincidence(per_group_peaks, global_peaks, tolerance=tolerance)

    # Print report
    label = f"p{prime}/s{seed}/ds{data_seed} ({variant_label})"
    print(f"\n=== {label}  smooth_k={smooth_k} ===")
    print(f"global MLP PR₃ peaks: {global_peaks}")

    if sweep:
        print("\nprominence sweep (peaks_per_group, %coincident, per-peak participation):")
        sweep_rows = sweep_prominence(
            pr3_per_group, epochs, global_peaks, sweep, tolerance=tolerance
        )
        print(
            f"  {'prom':<6}  {'peaks/grp':<20}  {'%coinc':<8}  participation"
        )
        for row in sweep_rows:
            part_str = "  ".join(
                f"{gp}:{n}/{len(group_freqs)}" for gp, n in row["participation"].items()
            )
            print(
                f"  {row['prominence']:<6}  {str(row['peaks_per_group']):<20}  "
                f"{row['pct_coincident']:>5.0f}%    {part_str}"
            )

    print(f"\nat prominence={per_group_prominence}:")
    for row in coincidence["per_group"]:
        g = row["group"]
        freq = int(group_freqs[g]) + 1
        print(
            f"  freq {freq:>3} group  n_peaks={len(row['coincident']) + len(row['solo']):>2}  "
            f"coincident={row['coincident']}  solo={row['solo']}"
        )
    total = coincidence["total_coincident"] + coincidence["total_solo"]
    if total:
        pct = 100 * coincidence["total_coincident"] / total
        print(
            f"summary: {coincidence['total_coincident']}/{total} peaks coincident "
            f"with global ({pct:.0f}%)  — COLLECTIVE if high, STAGGERED if low"
        )

    fig = plot_overlay(
        epochs,
        pr3_per_group,
        group_freqs,
        global_epochs,
        global_pr3,
        global_peaks,
        per_group_peaks,
        label,
    )
    return {
        "epochs": epochs,
        "pr3_per_group": pr3_per_group,
        "group_freqs": group_freqs,
        "per_group_peaks": per_group_peaks,
        "global_epochs": global_epochs,
        "global_pr3": global_pr3,
        "global_peaks": global_peaks,
        "coincidence": coincidence,
        "fig": fig,
    }


def extract_cohort_transit_axes(centroids, smooth_k=5):
    """Local PC1 direction for each cohort's centroid arc, in d_model space.

    Analogous to framework's per-segment v̂, but scaled down to the cohort level:
    v̂_cohort is the principal direction the cohort's centroid moves through d_model.
    Returns (G, d_model) unit vectors.
    """
    G, _, d_model = centroids.shape
    axes = np.zeros((G, d_model))
    for g in range(G):
        X = _smooth_ma(centroids[g], smooth_k)
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        axes[g] = Vt[0]  # leading PC direction
    return axes


def cohort_sigmoidality(centroids, epochs, ep_lo, ep_hi, smooth_k=5):
    """Δ = R²_sig - R²_lin for each cohort's centroid arc projected onto its own
    PC1 from ep_lo to ep_hi.

    Mirrors the framework's fit_local_sigmoid but at the cohort scale.
    Returns list of dicts with Δ and fitted params per cohort.
    """
    from scipy.optimize import curve_fit

    i_lo = _nearest_idx(epochs, ep_lo)
    i_hi = _nearest_idx(epochs, ep_hi)
    if i_hi <= i_lo + 2:
        return [None] * centroids.shape[0]

    results = []
    for g in range(centroids.shape[0]):
        X = _smooth_ma(centroids[g], smooth_k)
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        v_hat = Vt[0]
        proj_full = Xc @ v_hat
        # Normalize to [0, 1] across the (ep_lo, ep_hi) window
        seg = proj_full[i_lo : i_hi + 1]
        t_seg = np.asarray(epochs[i_lo : i_hi + 1], dtype=np.float64)
        if np.ptp(seg) < 1e-10:
            results.append(None)
            continue
        y = (seg - seg.min()) / np.ptp(seg)

        # Linear fit
        p_lin = np.polyfit(t_seg, y, 1)
        y_lin_hat = np.polyval(p_lin, t_seg)
        ss_res_lin = ((y - y_lin_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2_lin = 1 - ss_res_lin / max(ss_tot, 1e-12)

        # Sigmoid fit
        try:
            p0 = [1.0, 0.5 * (t_seg[0] + t_seg[-1]), (t_seg[-1] - t_seg[0]) / 8, 0.0]
            popt, _ = curve_fit(_sigmoid, t_seg, y, p0=p0, maxfev=5000)
            y_sig_hat = _sigmoid(t_seg, *popt)
            ss_res_sig = ((y - y_sig_hat) ** 2).sum()
            r2_sig = 1 - ss_res_sig / max(ss_tot, 1e-12)
        except Exception:
            r2_sig = np.nan

        delta = r2_sig - r2_lin if not np.isnan(r2_sig) else np.nan
        results.append({"r2_sig": r2_sig, "r2_lin": r2_lin, "delta": delta})
    return results


def cohort_pairwise_angles(axes):
    """Pairwise angle (degrees) between cohort local PC1 directions, sign-flip invariant."""
    G = axes.shape[0]
    A = np.eye(G) * 0  # pairwise, diagonal = 0
    for i in range(G):
        for j in range(G):
            if i == j:
                A[i, j] = 0.0
                continue
            c = float(np.clip(abs(np.dot(axes[i], axes[j])), 0, 1))
            A[i, j] = np.degrees(np.arccos(c))
    return A


def _sigmoid(t, A, center, slope, baseline):
    return baseline + A / (1.0 + np.exp(-(t - center) / max(slope, 1e-6)))


def _nearest_idx(epochs_arr, target_ep):
    return int(np.argmin(np.abs(np.asarray(epochs_arr) - target_ep)))


def run_cohort_axis_probe(
    prime=101,
    seed=485,
    data_seed=999,
    variant_label="p101 rebounder",
    family_name="modulo_addition_1layer",
    smooth_k=5,
    window_lo=1500,
    window_hi=None,
):
    """Pairwise angles and per-cohort sigmoidality for final + transient cohorts.

    Prediction: in rebounders, transient cohort's local PC1 is far off-axis
    from the final cohorts' local PC1s (near 90°). In healthy models with
    no transients, all final cohorts share direction (lower angles).
    """
    family = load_family(family_name)
    variant = family.get_variant(prime=prime, seed=seed, data_seed=data_seed)
    label = f"p{prime}/s{seed}/ds{data_seed} ({variant_label})"

    t = load_transient_cohort_centroids(variant)
    if t is None:
        print(f"{label}: no ever-qualified frequencies — skipping")
        return None

    epochs = t["epochs"]
    if window_hi is None:
        window_hi = int(epochs[-1])

    axes = extract_cohort_transit_axes(t["centroids"], smooth_k=smooth_k)
    sigmoidality = cohort_sigmoidality(
        t["centroids"], epochs, window_lo, window_hi, smooth_k=smooth_k
    )
    angles = cohort_pairwise_angles(axes)

    print(f"\n=== {label}  [cohort axis probe, window={window_lo}–{window_hi}] ===")
    labels = [
        f"f{int(t['freqs'][i]) + 1}{'*' if not t['is_final'][i] else ''}"
        for i in range(len(t["freqs"]))
    ]
    print(f"cohorts (* = transient): {labels}")

    print("\nper-cohort local PC1 sigmoidality:")
    print(f"  {'cohort':<8} {'final?':<8} {'Δ':<10} {'R²_sig':<10} {'R²_lin':<10}")
    for i, row in enumerate(sigmoidality):
        if row is None:
            continue
        final_tag = "final" if t["is_final"][i] else "TRANS"
        print(
            f"  {labels[i]:<8} {final_tag:<8} {row['delta']:+.4f}    "
            f"{row['r2_sig']:.4f}    {row['r2_lin']:.4f}"
        )

    print("\npairwise angles (deg) between cohort local PC1 (sign-flip invariant):")
    header = "         " + "  ".join(f"{l:>8}" for l in labels)
    print(header)
    for i, lbl in enumerate(labels):
        row = "  ".join(f"{angles[i, j]:>8.1f}" if i != j else f"{'—':>8}" for j in range(len(labels)))
        print(f"  {lbl:<8} {row}")

    transient_idx = np.where(~t["is_final"])[0]
    final_idx = np.where(t["is_final"])[0]
    if len(transient_idx) and len(final_idx):
        tf_angles = [
            angles[ti, fi] for ti in transient_idx for fi in final_idx
        ]
        ff_angles = [
            angles[i, j] for i in final_idx for j in final_idx if i != j
        ]
        print(
            f"\nsummary: mean transient↔final angle = {np.mean(tf_angles):.1f}°   "
            f"mean final↔final angle = {np.mean(ff_angles):.1f}°"
        )

    return {
        "labels": labels,
        "axes": axes,
        "angles": angles,
        "sigmoidality": sigmoidality,
        "is_final": t["is_final"],
        "freqs": t["freqs"],
    }


def run_transient_probe(
    prime=113,
    seed=485,
    data_seed=999,
    variant_label="p113 rebounder cand",
    family_name="modulo_addition_1layer",
    window=10,
    smooth_k=5,
    prominence=0.05,
    tolerance=500,
):
    """Per-cohort kink detection using transient-frequency peak-epoch members.

    Tests the hypothesis: late phantom global PR₃ peaks in rebounders align
    with transient-frequency cohorts un-committing. If a transient cohort's
    centroid arc has a kink at the phantom peak epoch, the phantom is
    localized to transient abandonment rather than mysterious rearrangement.
    """
    family = load_family(family_name)
    variant = family.get_variant(prime=prime, seed=seed, data_seed=data_seed)
    label = f"p{prime}/s{seed}/ds{data_seed} ({variant_label})"

    t = load_transient_cohort_centroids(variant)
    if t is None:
        print(f"{label}: no ever-qualified frequencies")
        return None

    epochs = t["epochs"]
    pr3 = per_group_rolling_pr3(t["centroids"], window=window, smooth_k=smooth_k)
    peaks_per_cohort = detect_per_group_peaks(pr3, epochs, prominence=prominence)
    global_epochs, global_pr3, global_peaks = load_global_mlp_peaks(variant, window=window)

    print(f"\n=== {label}  [transient-cohort probe] ===")
    print(f"global MLP PR₃ peaks: {global_peaks}")
    print(
        f"{'freq':<5} {'final?':<7} {'peak_ep':<9} {'peak_cnt':<10} "
        f"{'homeless':<10} {'cohort_peaks':<40}  coincident_with_global"
    )
    for i, freq in enumerate(t["freqs"]):
        cohort_peaks = peaks_per_cohort[i]
        coincident = [
            p for p in cohort_peaks if any(abs(p - gp) <= tolerance for gp in global_peaks)
        ]
        print(
            f"{int(freq) + 1:<5} "
            f"{'final' if t['is_final'][i] else 'trans':<7} "
            f"{int(t['peak_epoch'][i]):<9} "
            f"{int(t['peak_count'][i]):<10} "
            f"{int(t['homeless_count'][i]):<10} "
            f"{str(cohort_peaks):<40}  {coincident}"
        )

    # Per-global-peak participation by transient vs final cohorts
    transient_idx = np.where(~t["is_final"])[0]
    final_idx = np.where(t["is_final"])[0]
    print("\nper-global-peak participation (coincident cohorts / cohort count):")
    for gp in global_peaks:
        n_trans = sum(
            any(abs(p - gp) <= tolerance for p in peaks_per_cohort[i])
            for i in transient_idx
        )
        n_final = sum(
            any(abs(p - gp) <= tolerance for p in peaks_per_cohort[i]) for i in final_idx
        )
        print(
            f"  {gp:>6}    transients: {n_trans}/{len(transient_idx)}    "
            f"finals: {n_final}/{len(final_idx)}"
        )

    return {
        "epochs": epochs,
        "pr3": pr3,
        "peaks_per_cohort": peaks_per_cohort,
        "freqs": t["freqs"],
        "is_final": t["is_final"],
        "peak_epoch": t["peak_epoch"],
        "global_peaks": global_peaks,
    }


if __name__ == "__main__":
    out = run_sketch()
    out["fig"].show()
