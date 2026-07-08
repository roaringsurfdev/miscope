"""First-pass: which neurons got displaced in the p109 late-training event (~27k)?

p109_seed485_dseed598 grokked early, sat stable ~5k–26k, then spiked hard at
~27k and re-settled by ~30k with frequencies intact. The within-group W_in
spread plot implicates a small number of neurons "exploding". This script
ranks neurons by how far their W_in column moves during the event, relative to
their normal plateau drift, and reports the frequency-group membership of the
outliers.

Access goes through the miscope API (variant.artifacts), not file paths.
"""

from __future__ import annotations

import numpy as np

from miscope.families.discovery import load_family_from_dir

# --- Event windows (epochs) -------------------------------------------------
PLATEAU = (20000, 26000)   # quiet reference stretch before the event
EVENT = (26500, 28500)     # the spike + immediate aftermath
POST = 29999               # final settled checkpoint
GROUP_EPOCH = 26000        # assign neurons to freq groups from the stable solution


def load_w_in_trajectory(variant):
    """Return (epochs, W) with W shape (n_epochs, n_neurons, d_model)."""
    loader = variant.artifacts
    epochs = np.array(sorted(loader.get_epochs("parameter_snapshot")))
    cols = []
    for e in epochs:
        w_in = loader.load_epoch("parameter_snapshot", int(e))["W_in"]  # (d_model, n_neurons)
        cols.append(w_in.T.astype(np.float64))                          # (n_neurons, d_model)
    return epochs, np.stack(cols)


def freq_assignments(variant, epoch):
    g = variant.artifacts.load_epoch("neuron_grouping", epoch)
    return g["assignments"]  # (n_neurons,) freq index per neuron, -1 = unassigned


def _window_mask(epochs, lo, hi):
    return (epochs >= lo) & (epochs <= hi)


def main():
    fam = load_family_from_dir("data/modulo_addition_1layer", "data")
    variant = fam.get_variant(prime=109, seed=485, data_seed=598)

    epochs, W = load_w_in_trajectory(variant)   # (E, N, D)
    assign = freq_assignments(variant, GROUP_EPOCH)
    n_neurons = W.shape[1]

    plateau = _window_mask(epochs, *PLATEAU)
    event = _window_mask(epochs, *EVENT)
    post_idx = int(np.argmin(np.abs(epochs - POST)))

    # Per-neuron reference position = mean W_in column over the plateau.
    ref = W[plateau].mean(axis=0)                       # (N, D)

    # Normal plateau drift: std of per-step displacement during the quiet window.
    plateau_steps = np.linalg.norm(np.diff(W[plateau], axis=0), axis=2)  # (E_p-1, N)
    normal_step = plateau_steps.mean(axis=0) + 1e-9                      # (N,)

    # Peak excursion during the event, relative to plateau reference.
    event_disp = np.linalg.norm(W[event] - ref[None], axis=2)           # (E_e, N)
    peak_disp = event_disp.max(axis=0)                                  # (N,)

    # Net permanent relocation (plateau -> final).
    net_disp = np.linalg.norm(W[post_idx] - ref, axis=1)               # (N,)

    # "Explosion" score: peak excursion in units of the neuron's normal drift.
    score = peak_disp / normal_step

    order = np.argsort(score)[::-1]
    col_norm = np.linalg.norm(ref, axis=1)

    print(f"variant: {variant.name}   neurons: {n_neurons}   epochs: "
          f"{epochs[0]}..{epochs[-1]} (n={len(epochs)})")
    print(f"plateau {PLATEAU}  event {EVENT}  post {POST}\n")
    print(f"median normal per-step drift: {np.median(normal_step):.4f}")
    print(f"median peak event excursion:  {np.median(peak_disp):.4f}")
    print(f"median explosion score:       {np.median(score):.1f}x\n")

    print(f"{'rank':>4} {'neuron':>6} {'freq':>5} {'score(x)':>9} "
          f"{'peak':>7} {'net':>7} {'|w|':>6}")
    for rank, j in enumerate(order[:25], 1):
        print(f"{rank:>4} {j:>6} {assign[j]:>5} {score[j]:>9.1f} "
              f"{peak_disp[j]:>7.3f} {net_disp[j]:>7.3f} {col_norm[j]:>6.2f}")

    # How concentrated is the event? neurons whose excursion is a strong outlier.
    thresh = np.median(score) + 5 * (np.percentile(score, 75) - np.median(score))
    outliers = np.where(score > thresh)[0]
    print(f"\noutliers (score > {thresh:.0f}x, ~5 IQR above median): "
          f"{len(outliers)} of {n_neurons} neurons")

    # Freq-group breakdown of the outliers vs. the population.
    print("\nfreq-group membership (outliers / population):")
    for f in sorted(set(assign.tolist())):
        pop = int((assign == f).sum())
        out = int((assign[outliers] == f).sum()) if len(outliers) else 0
        if pop:
            print(f"  freq {f:>3}: {out:>3} / {pop:>3}")

    # Net-relocation outliers (permanent movers) — may differ from peak movers.
    net_score = net_disp / normal_step
    net_order = np.argsort(net_score)[::-1]
    print("\ntop net-relocation neurons (permanent movers):")
    print(f"{'neuron':>6} {'freq':>5} {'net(x)':>8} {'peak(x)':>8}")
    for j in net_order[:10]:
        print(f"{j:>6} {assign[j]:>5} {net_score[j]:>8.1f} {score[j]:>8.1f}")

    frequency_switch_report(variant)


def frequency_switch_report(variant):
    """Did neurons change frequency group across the event? (the switch guess)

    Caveat: neuron_grouping is a clustering; weakly-committed boundary neurons
    jitter between groups. Gate switches on confidence to separate genuine
    functional reassignment from clustering noise.
    """
    g_pre = variant.artifacts.load_epoch("neuron_grouping", GROUP_EPOCH)
    g_post = variant.artifacts.load_epoch("neuron_grouping", POST)
    pre, post = g_pre["assignments"], g_post["assignments"]
    c_pre, c_post = g_pre["confidence"], g_post["confidence"]

    sw = pre != post
    print(f"\n--- frequency-switch report ({GROUP_EPOCH} -> {POST}) ---")
    print("group sizes pre :", _sizes(pre))
    print("group sizes post:", _sizes(post))
    print(f"neurons switching group: {int(sw.sum())} / {len(pre)}")
    print(f"mean confidence  stayers {c_pre[~sw].mean():.2f}  "
          f"switchers {c_pre[sw].mean():.2f}")
    confident = sw & (c_pre > 0.6) & (c_post > 0.6)
    print(f"high-confidence switches (conf>0.6 both ends): {int(confident.sum())}")
    flows = {}
    for j in np.where(sw)[0]:
        flows[(int(pre[j]), int(post[j]))] = flows.get((int(pre[j]), int(post[j])), 0) + 1
    print("dominant flows:", dict(sorted(flows.items(), key=lambda x: -x[1])[:5]))


def _sizes(a):
    u, c = np.unique(a, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


if __name__ == "__main__":
    main()
