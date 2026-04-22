"""Saddle-transport kinematics primitives.

Pure library functions operating on existing `parameter_trajectory` and
`parameter_snapshot` artifacts. Scope: MLP weight-space geometry along a
training trajectory — wiggle detection, transit projection, arc decomposition,
valley topology, local sigmoid fits, pairwise axis geometry.

All functions are free-standing; none depend on notebook-level state.
Plotting and regime classification live outside this module — the former
because visualization is notebook-specific, the latter because the current
rule-based classifier is known inadequate (pending 2-axis rewrite).
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.library import COMPONENT_GROUPS

# Rolling PR₃ is defined in the dimensionality_dynamics renderer; importing
# the private helper keeps a single implementation. If this cross-module
# dependency becomes load-bearing elsewhere, promote to public.
from miscope.visualization.renderers.dimensionality_dynamics import (
    _compute_rolling_trajectory_metrics,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sigmoid(t, A, center, slope, baseline):
    return baseline + A / (1.0 + np.exp(-(t - center) / max(slope, 1e-6)))


def _nearest_idx(epochs_arr, target_ep):
    return int(np.argmin(np.abs(np.asarray(epochs_arr) - target_ep)))


# ---------------------------------------------------------------------------
# Parameter-space loading
# ---------------------------------------------------------------------------


def load_mlp_param_trajectory(variant):
    """Load W_in ⊕ W_out flattened per epoch — full MLP parameter space."""
    loader = ArtifactLoader(str(variant.variant_dir / "artifacts"))
    epochs = sorted(loader.get_epochs("parameter_snapshot"))
    mlp_keys = COMPONENT_GROUPS["mlp"]
    rows = []
    for e in epochs:
        snap = loader.load_epoch("parameter_snapshot", e)
        rows.append(np.concatenate([snap[k].flatten() for k in mlp_keys]))
    X = np.stack(rows).astype(np.float64)
    return np.array(epochs), X


# ---------------------------------------------------------------------------
# Transit projection (global chord)
# ---------------------------------------------------------------------------


def compute_transit_projection(X, epochs, ep_lo, ep_hi):
    """Project trajectory onto the unit chord from ep_lo to ep_hi.

    Normalized so s=0 at ep_lo, s=1 at ep_hi. Returns (s, L, v_hat, i_lo, i_hi).
    """
    i_lo = int(np.where(epochs == ep_lo)[0][0])
    i_hi = int(np.where(epochs == ep_hi)[0][0])
    delta = X[i_hi] - X[i_lo]
    L = float(np.linalg.norm(delta))
    v_hat = delta / L
    s = (X - X[i_lo]) @ v_hat / L
    return s, L, v_hat, i_lo, i_hi


def compute_velocity(s, epochs):
    """ds/dt using central differences scaled by per-epoch gap."""
    s = np.asarray(s, dtype=np.float64)
    ep = np.asarray(epochs, dtype=np.float64)
    ds_dt = np.zeros_like(s)
    ds_dt[1:-1] = (s[2:] - s[:-2]) / (ep[2:] - ep[:-2])
    ds_dt[0] = (s[1] - s[0]) / (ep[1] - ep[0])
    ds_dt[-1] = (s[-1] - s[-2]) / (ep[-1] - ep[-2])
    return ds_dt


# ---------------------------------------------------------------------------
# Arc decomposition (curvature + local-PC1 / tangent alignment)
# ---------------------------------------------------------------------------


def compute_position_backbone(proj, smooth_k=20):
    """Moving-average smoothing of a (n, k) trajectory. Returns (n, k)."""
    proj = np.asarray(proj, dtype=np.float64)
    n = proj.shape[0]
    smoothed = np.zeros_like(proj)
    for t in range(n):
        lo = max(0, t - smooth_k)
        hi = min(n, t + smooth_k + 1)
        smoothed[t] = proj[lo:hi].mean(axis=0)
    return smoothed


def _compute_arc_tangent(proj, epochs, smooth_k):
    n, k = proj.shape
    tau = np.zeros((n, k))
    for t in range(n):
        lo = max(0, t - smooth_k)
        hi = min(n - 1, t + smooth_k)
        delta = proj[hi] - proj[lo]
        norm = np.linalg.norm(delta)
        tau[t] = delta / norm if norm > 1e-12 else 0.0
    curvature = np.zeros(n)
    for t in range(1, n - 1):
        dt = epochs[t + 1] - epochs[t - 1]
        if dt > 0:
            curvature[t] = np.linalg.norm(tau[t + 1] - tau[t - 1]) / dt
    return tau, curvature


def _compute_local_pc1_alignment(proj, tau, window):
    n, k = proj.shape
    local_pc1 = np.zeros((n, k))
    alignment = np.full(n, np.nan)
    for t in range(window - 1, n):
        w = proj[t - window + 1 : t + 1]
        wc = w - w.mean(axis=0)
        _, _, vt = np.linalg.svd(wc, full_matrices=False)
        u1 = vt[0]
        local_pc1[t] = u1
        alignment[t] = abs(float(np.dot(u1, tau[t])))
    return local_pc1, alignment


def compute_arc_decomposition(proj, epochs, smooth_k=20, window=10):
    """On-arc vs off-arc decomposition along a projected trajectory.

    smooth_k — half-window for arc-tangent smoothing (>> window).
    window   — rolling PCA window (matches pipeline's rolling PR₃ window).
    Returns dict(tau, curvature, local_pc1, alignment), each length n_epochs.
    """
    proj = np.asarray(proj, dtype=np.float64)
    epochs = np.asarray(epochs, dtype=np.float64)
    tau, curvature = _compute_arc_tangent(proj, epochs, smooth_k)
    local_pc1, alignment = _compute_local_pc1_alignment(proj, tau, window)
    return dict(
        tau=tau,
        curvature=curvature,
        local_pc1=local_pc1,
        alignment=alignment,
    )


# ---------------------------------------------------------------------------
# Valley topology (peak-valley pair analysis on rolling PR₃)
# ---------------------------------------------------------------------------


def _classify_transition_kind(n_intervening):
    if n_intervening == 0:
        return "clean_valley"
    if n_intervening == 1:
        return "aborted_commit"
    return "irregular"


def analyze_valley_topology(pr3, epochs, peaks_idx, low_prom=0.02):
    """For each adjacent pair of prominent peaks, characterize the inter-peak topology.

    peaks_idx — indices into pr3/epochs (from find_peaks at higher prominence).
    low_prom  — prominence threshold for detecting sub-threshold intervening maxes.
    Returns list of dicts, one per adjacent-peak pair.
    """
    transitions = []
    for i in range(len(peaks_idx) - 1):
        lo, hi = int(peaks_idx[i]), int(peaks_idx[i + 1])
        if hi - lo < 3:
            continue
        seg = pr3[lo : hi + 1]
        interior = seg[1:-1]
        mid_max_rel, _ = find_peaks(interior, prominence=low_prom)
        mid_max_idx = mid_max_rel + 1 + lo
        valley_rel = int(np.argmin(seg))
        valley_idx = lo + valley_rel
        flank = float(min(pr3[lo], pr3[hi]))
        depth = float(flank - pr3[valley_idx])
        reach = float(pr3[valley_idx] - 1.0)
        transitions.append(
            {
                "peak_a_ep": int(epochs[lo]),
                "peak_b_ep": int(epochs[hi]),
                "peak_a_pr3": float(pr3[lo]),
                "peak_b_pr3": float(pr3[hi]),
                "valley_ep": int(epochs[valley_idx]),
                "valley_pr3": float(pr3[valley_idx]),
                "flank_pr3": flank,
                "depth": depth,
                "reach_to_one": reach,
                "intervening_eps": [int(epochs[m]) for m in mid_max_idx],
                "intervening_pr3": [float(pr3[m]) for m in mid_max_idx],
                "kind": _classify_transition_kind(len(mid_max_idx)),
            }
        )
    return transitions


# ---------------------------------------------------------------------------
# Valley-bounded segments + local sigmoid fits
# ---------------------------------------------------------------------------


def build_valley_segments(pr3, epochs, fd_end, commit_ep, topology, prom=0.05):
    """Valley-bounded segments, one per prominent PR₃ peak in [fd_end, commit_ep].

    Peaks re-derived with find_peaks(prominence=prom) for parity with §14.
    topology — output of analyze_valley_topology for the same window.
    Returns list of dicts: segment_idx, ep_lo, ep_hi, peak_ep.
    """
    in_win = (epochs >= fd_end) & (epochs <= commit_ep)
    pr3_win = pr3[in_win]
    eps_win = epochs[in_win]
    peaks_idx, _ = find_peaks(pr3_win, prominence=prom)
    peak_eps = [int(eps_win[i]) for i in peaks_idx]
    if not peak_eps:
        return []
    valley_eps = [t["valley_ep"] for t in topology]
    if len(valley_eps) != len(peak_eps) - 1:
        raise ValueError(
            f"topology has {len(valley_eps)} valleys but {len(peak_eps)} peaks "
            f"require {len(peak_eps) - 1}. Pass the output of "
            f"analyze_valley_topology for the same (pr3, epochs, peaks_idx)."
        )
    segments = []
    for i, pk in enumerate(peak_eps):
        ep_lo = fd_end if i == 0 else valley_eps[i - 1]
        ep_hi = commit_ep if i == len(peak_eps) - 1 else valley_eps[i]
        segments.append(
            dict(segment_idx=i, ep_lo=int(ep_lo), ep_hi=int(ep_hi), peak_ep=pk)
        )
    return segments


def _fit_sigmoid_and_linear(t_norm, s):
    try:
        popt, _ = curve_fit(
            _sigmoid,
            t_norm,
            s,
            p0=[1.0, 0.5, 0.15, 0.0],
            bounds=([0.1, -0.5, 0.01, -0.5], [2.0, 1.5, 1.0, 0.5]),
            maxfev=5000,
        )
        s_sig = _sigmoid(t_norm, *popt)
        ss_res_sig = float(np.sum((s - s_sig) ** 2))
    except Exception:
        popt, s_sig, ss_res_sig = None, None, float("inf")
    coef = np.polyfit(t_norm, s, 1)
    s_lin = np.polyval(coef, t_norm)
    ss_res_lin = float(np.sum((s - s_lin) ** 2))
    return popt, s_sig, ss_res_sig, s_lin, ss_res_lin


def fit_local_sigmoid(X, epochs, ep_lo, ep_hi):
    """Local transit projection + sigmoid-vs-linear fit within [ep_lo, ep_hi].

    Transit vector computed in full MLP parameter space. Returns None for
    segments too short (<5 samples) or with vanishing chord length.
    """
    i_lo = _nearest_idx(epochs, ep_lo)
    i_hi = _nearest_idx(epochs, ep_hi)
    if i_hi - i_lo < 5:
        return None
    delta = X[i_hi] - X[i_lo]
    L = float(np.linalg.norm(delta))
    if L < 1e-12:
        return None
    v_hat = delta / L
    eps_seg = np.asarray(epochs[i_lo : i_hi + 1], dtype=np.float64)
    seg = X[i_lo : i_hi + 1]
    s = (seg - X[i_lo]) @ v_hat / L
    t_norm = (eps_seg - eps_seg[0]) / (eps_seg[-1] - eps_seg[0])
    popt, s_sig, ss_res_sig, s_lin, ss_res_lin = _fit_sigmoid_and_linear(t_norm, s)
    ss_tot = float(np.sum((s - s.mean()) ** 2)) + 1e-12
    r2_sig = 1.0 - ss_res_sig / ss_tot
    r2_lin = 1.0 - ss_res_lin / ss_tot
    return dict(
        eps=eps_seg,
        s=s,
        s_sig=s_sig,
        s_lin=s_lin,
        L=L,
        v_hat=v_hat,
        sigmoid_params=popt,
        r2_sig=r2_sig,
        r2_lin=r2_lin,
        sigmoidality=r2_sig - r2_lin,
    )


# ---------------------------------------------------------------------------
# Per-segment axes + pairwise geometry
# ---------------------------------------------------------------------------


def extract_local_axes(entries, epochs, X):
    """Unit transit vector, chord length, and chord velocity per segment entry.

    entries — list of dicts each with keys ep_lo, ep_hi, segment_idx, peak_ep,
              and any per-segment fit metadata. Skipped if chord is degenerate.
    """
    axes = []
    for entry in entries:
        i_lo = _nearest_idx(epochs, entry["ep_lo"])
        i_hi = _nearest_idx(epochs, entry["ep_hi"])
        delta = X[i_hi] - X[i_lo]
        L = float(np.linalg.norm(delta))
        if L < 1e-12:
            continue
        v_hat = delta / L
        dur = float(entry["ep_hi"] - entry["ep_lo"])
        velocity = L / dur if dur > 0 else 0.0
        axis = dict(entry)
        axis.update(v_hat=v_hat, L=L, duration=dur, velocity=velocity)
        axes.append(axis)
    return axes


def pairwise_angle_matrix(axes):
    """Pairwise angles (degrees) between unit transit vectors, sign-flip invariant.

    Uses |cos θ| so anti-parallel axes (same geometric direction) register as 0°.
    Returns (n, n) symmetric matrix.
    """
    n = len(axes)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            c = float(np.clip(np.dot(axes[i]["v_hat"], axes[j]["v_hat"]), -1.0, 1.0))
            M[i, j] = float(np.degrees(np.arccos(abs(c))))
    return M


# ---------------------------------------------------------------------------
# Commitment anchor + convenience pipeline
# ---------------------------------------------------------------------------


def commitment_epoch(timing):
    """Commitment anchor — grok if present, else sd_end."""
    return timing["grok"] if timing.get("grok") is not None else timing["sd_end"]


def _load_variant_timing(variant):
    with open(variant.variant_dir / "variant_summary.json") as f:
        summary = json.load(f)
    grok_raw = summary.get("test_loss_threshold_first_epoch")
    grok = grok_raw if grok_raw not in (None, -1) else None
    timing = {
        "fd_end": summary["first_descent_window"]["end_epoch"],
        "sd_onset": summary.get("second_descent_onset_epoch"),
        "sd_end": summary.get("second_descent_window", {}).get("end_epoch"),
        "grok": grok,
    }
    return summary, timing


def _detect_wiggle(pt_eps, pr3, pc3, fd_end, search_hi):
    in_win = (pt_eps >= fd_end) & (pt_eps <= search_hi)
    we_pr3 = int(pt_eps[int(np.argmax(np.where(in_win, pr3, -np.inf)))])
    we_pc3 = int(pt_eps[int(np.argmax(np.where(in_win, np.abs(pc3), -np.inf)))])
    return we_pr3, we_pc3


def run_saddle_pipeline(variant, terminal=None, window=10):
    """Full saddle-transport pipeline for a variant.

    For non-grokkers (test_loss_threshold_first_epoch ∈ {-1, None}), the wiggle
    search window falls back to [fd_end, sd_onset] — the structural analog of
    [fd_end, grok], since 'commitment' for these models lives inside / after
    second descent rather than at a grok point.

    terminal defaults to the final checkpoint; pass an explicit epoch to study
    alternative landing points (e.g. 'near-landing' for rebounders).
    """
    p = variant.params
    label = f"p{p['prime']}/s{p['seed']}/ds{p['data_seed']}"
    summary, timing = _load_variant_timing(variant)

    epochs, X = load_mlp_param_trajectory(variant)

    loader = ArtifactLoader(str(variant.variant_dir / "artifacts"))
    pt = loader.load_cross_epoch("parameter_trajectory")
    pt_eps = pt["epochs"]
    mlp_proj = pt["mlp__projections"]
    pr3, _ = _compute_rolling_trajectory_metrics(mlp_proj, window=window)
    pc3 = mlp_proj[:, 2]

    search_hi = timing["grok"] if timing["grok"] is not None else timing["sd_onset"]
    we_pr3, we_pc3 = _detect_wiggle(pt_eps, pr3, pc3, timing["fd_end"], search_hi)

    if terminal is None:
        terminal = int(epochs[-1])
    s, L, v_hat, i_w, i_T = compute_transit_projection(X, epochs, we_pr3, terminal)
    v = compute_velocity(s, epochs)
    saddle = int(epochs[int(np.argmax(v))])

    return dict(
        label=label,
        variant=variant,
        summary=summary,
        timing=timing,
        epochs=epochs,
        X=X,
        pt_eps=pt_eps,
        mlp_proj=mlp_proj,
        pr3=pr3,
        pc3=pc3,
        wiggle_pr3=we_pr3,
        wiggle_pc3=we_pc3,
        terminal=terminal,
        s=s,
        v=v,
        L=L,
        v_hat=v_hat,
        saddle=saddle,
        max_dsdt=float(v.max()),
        min_dsdt=float(v.min()),
        s_min=float(s.min()),
        s_min_ep=int(epochs[int(np.argmin(s))]),
        s_max=float(s.max()),
        s_max_ep=int(epochs[int(np.argmax(s))]),
    )
