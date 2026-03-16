# %% imports
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope import load_family

# %% Configuration
# Each group shares a model_seed — epoch 0 weights should be (near-)identical.
# Reference seed is the first listed; comparisons are against it.
#
# Note: epoch 0 weights are NOT bit-for-bit identical across data seeds despite
# sharing model_seed. Max per-element diff is ~0.002 vs parameter norms of ~8-18
# (~1-2%). Origin unclear — possibly initialization randomness tied to data loading
# order, or first-batch inclusion in epoch-0 checkpoint. Either way, the epoch-0
# L2 distance serves as a natural baseline for "pre-training divergence."

FAMILY_NAME = "modulo_addition_1layer"
WEIGHT_KEYS = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"]

# Variant groups: (prime, model_seed, [data_seeds], reference_dseed)
GROUPS = [
    (113, 999, [598, 42, 999], 598),   # dseed=999 struggles, dseed=42 partial
    (113, 485, [598, 42, 999], 598),   # dseed=999 struggles
    (101, 485, [598, 42, 999], 598),   # dseed=999 struggles
    (101, 999, [598, 42, 999], 598),   # all three grokk — control group
    (109, 485, [598, 42, 999], 598),   # all three grokk — p109 stable case
]

COLORS = {42: "tomato", 598: "steelblue", 999: "orange"}

family = load_family(FAMILY_NAME)


# %% Helper: load all parameter_snapshot epochs for a variant
def load_weight_trajectory(variant) -> tuple[list[int], dict[str, np.ndarray]]:
    """Load all parameter_snapshot epochs.

    Returns:
        epochs: sorted list of checkpoint epochs
        weights: dict mapping matrix name -> (n_epochs, *shape) stacked array
    """
    epochs = variant.artifacts.get_epochs("parameter_snapshot")
    stacked: dict[str, list[np.ndarray]] = {k: [] for k in WEIGHT_KEYS}

    for epoch in epochs:
        snap = variant.artifacts.load_epoch("parameter_snapshot", epoch)
        for k in WEIGHT_KEYS:
            stacked[k].append(snap[k])

    return epochs, {k: np.stack(v) for k, v in stacked.items()}


def compute_pairwise_divergence(
    epochs: list[int],
    weights_ref: dict[str, np.ndarray],
    weights_cmp: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute L2 divergence between two weight trajectories at each epoch.

    Returns:
        per_matrix: dict mapping matrix name -> (n_epochs,) L2 distances
        aggregate: (n_epochs,) total L2 across all matrices
        normalized: (n_epochs,) aggregate / reference norm (relative divergence)
    """
    per_matrix: dict[str, np.ndarray] = {}
    aggregate = np.zeros(len(epochs))
    ref_norms = np.zeros(len(epochs))

    for k in WEIGHT_KEYS:
        diff = weights_cmp[k] - weights_ref[k]  # (n_epochs, *shape)
        flat_diff = diff.reshape(len(epochs), -1)
        l2 = np.linalg.norm(flat_diff, axis=1)
        per_matrix[k] = l2
        aggregate += l2 ** 2

        flat_ref = weights_ref[k].reshape(len(epochs), -1)
        ref_norms += np.linalg.norm(flat_ref, axis=1) ** 2

    aggregate = np.sqrt(aggregate)
    normalized = aggregate / (np.sqrt(ref_norms) + 1e-12)

    return {"per_matrix": per_matrix, "aggregate": aggregate, "normalized": normalized}


# %% Load all trajectories
print("Loading weight trajectories...")
trajectories: dict[tuple, tuple[list[int], dict[str, np.ndarray]]] = {}

for prime, mseed, dseeds, ref_ds in GROUPS:
    for ds in dseeds:
        v = family.get_variant(prime=prime, seed=mseed, data_seed=ds)
        epochs, weights = load_weight_trajectory(v)
        trajectories[(prime, mseed, ds)] = (epochs, weights)
        print(f"  p{prime}/seed{mseed}/dseed{ds}: {len(epochs)} epochs")

print("Done.")


# %% Compute pairwise divergence for each group
# divergences key: (prime, mseed, ref_ds, cmp_ds)
divergences: dict[tuple, dict] = {}

def _compute_group_divergences(
    prime: int,
    mseed: int,
    ref_ds: int,
    cmp_dseeds: list[int],
) -> None:
    """Compute and store divergences for one reference seed against all comparisons."""
    ref_epochs, ref_weights = trajectories[(prime, mseed, ref_ds)]
    for cmp_ds in cmp_dseeds:
        if cmp_ds == ref_ds:
            continue
        cmp_epochs, cmp_weights = trajectories[(prime, mseed, cmp_ds)]
        shared = sorted(set(ref_epochs) & set(cmp_epochs))
        ref_idx = [ref_epochs.index(e) for e in shared]
        cmp_idx = [cmp_epochs.index(e) for e in shared]
        ref_w = {k: ref_weights[k][ref_idx] for k in WEIGHT_KEYS}
        cmp_w = {k: cmp_weights[k][cmp_idx] for k in WEIGHT_KEYS}
        div = compute_pairwise_divergence(shared, ref_w, cmp_w)
        div["epochs"] = shared
        divergences[(prime, mseed, ref_ds, cmp_ds)] = div

# Standard: dseed=598 as reference
for prime, mseed, dseeds, ref_ds in GROUPS:
    _compute_group_divergences(prime, mseed, ref_ds, dseeds)

# Additional: dseed=42 as reference (to see dseed42 vs dseed999 directly)
for prime, mseed, dseeds, _ in GROUPS:
    if 42 in dseeds and 999 in dseeds:
        _compute_group_divergences(prime, mseed, 42, [999])


# %% Plot 1: Aggregate divergence trajectories — all groups
# Rows = model groups, one line per comparison pair.
# Reveals when (and whether) weight-space paths fork and whether they reconverge.

fig_agg = make_subplots(
    rows=3, cols=2,
    subplot_titles=[f"p{p}/seed{ms} (ref=dseed{ref})" for p, ms, _, ref in GROUPS],
    shared_xaxes=False,
)

for group_idx, (prime, mseed, dseeds, ref_ds) in enumerate(GROUPS):
    row, col = divmod(group_idx, 2)
    for cmp_ds in dseeds:
        if cmp_ds == ref_ds:
            continue
        div = divergences[(prime, mseed, ref_ds, cmp_ds)]
        fig_agg.add_trace(
            go.Scatter(
                x=div["epochs"],
                y=div["aggregate"].tolist(),
                mode="lines",
                name=f"dseed{cmp_ds} vs dseed{ref_ds}",
                line=dict(color=COLORS[cmp_ds]),
                showlegend=(group_idx == 0),
            ),
            row=row + 1, col=col + 1,
        )

fig_agg.update_layout(
    title="Weight-space L2 divergence from reference (dseed=598) — all groups",
    height=800,
)
fig_agg.update_xaxes(title_text="Epoch")
fig_agg.update_yaxes(title_text="L2 distance")
fig_agg.show()


# %% Plot 2: Normalized (relative) divergence
# L2 distance / reference parameter norm — accounts for weight magnitude growth.
# Tells us whether divergence is large relative to where the weights are.

fig_norm = make_subplots(
    rows=3, cols=2,
    subplot_titles=[f"p{p}/seed{ms} (ref=dseed{ref})" for p, ms, _, ref in GROUPS],
    shared_xaxes=False,
)

for group_idx, (prime, mseed, dseeds, ref_ds) in enumerate(GROUPS):
    row, col = divmod(group_idx, 2)
    for cmp_ds in dseeds:
        if cmp_ds == ref_ds:
            continue
        div = divergences[(prime, mseed, ref_ds, cmp_ds)]
        fig_norm.add_trace(
            go.Scatter(
                x=div["epochs"],
                y=(div["normalized"] * 100).tolist(),
                mode="lines",
                name=f"dseed{cmp_ds} vs dseed{ref_ds}",
                line=dict(color=COLORS[cmp_ds]),
                showlegend=(group_idx == 0),
            ),
            row=row + 1, col=col + 1,
        )

fig_norm.update_layout(
    title="Normalized weight divergence (% of reference norm) — all groups",
    height=800,
)
fig_norm.update_xaxes(title_text="Epoch")
fig_norm.update_yaxes(title_text="Divergence (%)")
fig_norm.show()


# %% Plot 3: Per-matrix divergence breakdown — p113/seed999 (most interesting case)
# Which matrices drive the fork? Do they all diverge together or does one lead?
# Focus on the group with mixed outcomes (one grokks, one struggles, one partial).

FOCUS_GROUP = (113, 999, [598, 42, 999], 598)
prime, mseed, dseeds, ref_ds = FOCUS_GROUP

fig_matrix = make_subplots(
    rows=3, cols=3,
    subplot_titles=WEIGHT_KEYS,
    shared_xaxes=True,
)

for midx, matrix_name in enumerate(WEIGHT_KEYS):
    mrow, mcol = divmod(midx, 3)
    for cmp_ds in dseeds:
        if cmp_ds == ref_ds:
            continue
        div = divergences[(prime, mseed, ref_ds, cmp_ds)]
        fig_matrix.add_trace(
            go.Scatter(
                x=div["epochs"],
                y=div["per_matrix"][matrix_name].tolist(),
                mode="lines",
                name=f"dseed{cmp_ds}",
                line=dict(color=COLORS[cmp_ds]),
                showlegend=(midx == 0),
            ),
            row=mrow + 1, col=mcol + 1,
        )

fig_matrix.update_layout(
    title=f"Per-matrix L2 divergence — p{prime}/seed{mseed} (ref=dseed{ref_ds})",
    height=700,
)
fig_matrix.update_xaxes(title_text="Epoch")
fig_matrix.show()


# %% Plot 4: Per-matrix breakdown — p101/seed999 (control: all grokk)
# Sanity check — do weights still diverge when all seeds succeed?
# If yes: divergence is inherent to different training data, not tied to failure.
# If no: divergence is specific to cases where outcomes differ.

CONTROL_GROUP = (101, 999, [598, 42, 999], 598)
prime_c, mseed_c, dseeds_c, ref_ds_c = CONTROL_GROUP

fig_control = make_subplots(
    rows=3, cols=3,
    subplot_titles=WEIGHT_KEYS,
    shared_xaxes=True,
)

for midx, matrix_name in enumerate(WEIGHT_KEYS):
    mrow, mcol = divmod(midx, 3)
    for cmp_ds in dseeds_c:
        if cmp_ds == ref_ds_c:
            continue
        div = divergences[(prime_c, mseed_c, ref_ds_c, cmp_ds)]
        fig_control.add_trace(
            go.Scatter(
                x=div["epochs"],
                y=div["per_matrix"][matrix_name].tolist(),
                mode="lines",
                name=f"dseed{cmp_ds}",
                line=dict(color=COLORS[cmp_ds]),
                showlegend=(midx == 0),
            ),
            row=mrow + 1, col=mcol + 1,
        )

fig_control.update_layout(
    title=f"Per-matrix L2 divergence — p{prime_c}/seed{mseed_c} (ref=dseed{ref_ds_c}) [control: all grokk]",
    height=700,
)
fig_control.update_xaxes(title_text="Epoch")
fig_control.show()


# %% Plot 5: Early window zoom — epochs 0-2000
# Focuses on the critical first-descent window identified in gradient analysis.
# Does divergence grow during first descent (0-400) and plateau thereafter?
# Or does it continue growing through the memorization plateau?

EARLY_CUTOFF = 2000

fig_early = make_subplots(
    rows=3, cols=2,
    subplot_titles=[f"p{p}/seed{ms} (ref=dseed{ref})" for p, ms, _, ref in GROUPS],
    shared_xaxes=False,
)

for group_idx, (prime, mseed, dseeds, ref_ds) in enumerate(GROUPS):
    row, col = divmod(group_idx, 2)
    for cmp_ds in dseeds:
        if cmp_ds == ref_ds:
            continue
        div = divergences[(prime, mseed, ref_ds, cmp_ds)]
        epochs_arr = np.array(div["epochs"])
        mask = epochs_arr <= EARLY_CUTOFF
        fig_early.add_trace(
            go.Scatter(
                x=epochs_arr[mask].tolist(),
                y=div["aggregate"][mask].tolist(),
                mode="lines+markers",
                name=f"dseed{cmp_ds} vs dseed{ref_ds}",
                line=dict(color=COLORS[cmp_ds]),
                marker=dict(size=5),
                showlegend=(group_idx == 0),
            ),
            row=row + 1, col=col + 1,
        )

fig_early.update_layout(
    title=f"Weight-space divergence — early window (epochs 0–{EARLY_CUTOFF})",
    height=800,
)
fig_early.update_xaxes(title_text="Epoch")
fig_early.update_yaxes(title_text="L2 distance")
fig_early.show()


# %% Plot 6: Per-matrix divergence — p109/seed485 (all three grokk, p109 stability case)
# Does a prime that is robust to data seed show the same divergence pattern,
# or does stability reflect lower divergence in specific matrices?

P109_GROUP = (109, 485, [598, 42, 999], 598)
prime_p, mseed_p, dseeds_p, ref_ds_p = P109_GROUP

fig_p109 = make_subplots(
    rows=3, cols=3,
    subplot_titles=WEIGHT_KEYS,
    shared_xaxes=True,
)

for midx, matrix_name in enumerate(WEIGHT_KEYS):
    mrow, mcol = divmod(midx, 3)
    for cmp_ds in dseeds_p:
        if cmp_ds == ref_ds_p:
            continue
        div = divergences[(prime_p, mseed_p, ref_ds_p, cmp_ds)]
        fig_p109.add_trace(
            go.Scatter(
                x=div["epochs"],
                y=div["per_matrix"][matrix_name].tolist(),
                mode="lines",
                name=f"dseed{cmp_ds}",
                line=dict(color=COLORS[cmp_ds]),
                showlegend=(midx == 0),
            ),
            row=mrow + 1, col=mcol + 1,
        )

fig_p109.update_layout(
    title=f"Per-matrix L2 divergence — p{prime_p}/seed{mseed_p} (ref=dseed{ref_ds_p}) [all grokk — p109 stable]",
    height=700,
)
fig_p109.update_xaxes(title_text="Epoch")
fig_p109.show()


# %% Plot 7: dseed42 vs dseed999 — direct pairwise comparison across groups
# Uses dseed=42 as reference. Shows whether dseed42 and dseed999 diverge from
# each other or converge, independent of dseed=598.
# Key question: are these two "non-canonical" seeds also on different paths,
# or do they converge to a shared non-canonical trajectory?

COLOR_999_VS_42 = "purple"

fig_pairwise = make_subplots(
    rows=3, cols=2,
    subplot_titles=[f"p{p}/seed{ms}" for p, ms, _, _ in GROUPS],
    shared_xaxes=False,
)

for group_idx, (prime, mseed, dseeds, _) in enumerate(GROUPS):
    if 42 not in dseeds or 999 not in dseeds:
        continue
    row, col = divmod(group_idx, 2)
    div_42_vs_999 = divergences[(prime, mseed, 42, 999)]
    div_42_vs_598 = divergences[(prime, mseed, 598, 42)]

    fig_pairwise.add_trace(
        go.Scatter(
            x=div_42_vs_999["epochs"],
            y=div_42_vs_999["normalized"] * 100,
            mode="lines",
            name="dseed42 vs dseed999",
            line=dict(color=COLOR_999_VS_42),
            showlegend=(group_idx == 0),
        ),
        row=row + 1, col=col + 1,
    )
    fig_pairwise.add_trace(
        go.Scatter(
            x=div_42_vs_598["epochs"],
            y=div_42_vs_598["normalized"] * 100,
            mode="lines",
            name="dseed42 vs dseed598",
            line=dict(color=COLORS[42], dash="dot"),
            showlegend=(group_idx == 0),
        ),
        row=row + 1, col=col + 1,
    )

fig_pairwise.update_layout(
    title="Normalized divergence — dseed42 vs dseed999 direct comparison (purple) vs dseed42 vs dseed598 (red dot)",
    height=800,
)
fig_pairwise.update_xaxes(title_text="Epoch")
fig_pairwise.update_yaxes(title_text="Divergence (%)")
fig_pairwise.show()
