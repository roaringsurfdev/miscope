"""Does p109's ~27k 'odd neuron pattern' recur in p101/s999/ds999 or p103/s485/ds598?
p109 §7 signature: an exploder neuron's (a,b) activation map COLLAPSES to a coarse low-freq
block envelope while its activation POWER DETONATES (n327 std 0.16->4, ~25x std / ~800x power).

Computed from the neuron_activations artifact (per-neuron pxp maps) — no forward passes.
Per neuron: std over the (a,b) grid per epoch = activation power. Detonation = max(std in event
window)/median(std on plateau). For the top detonator, the 2D-FFT dominant spatial frequency at
its peak (low index = coarse blocks = the p109 collapse; high = fine = normal grid).
"""
import os
from pathlib import Path
import numpy as np

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "modulo_addition_1layer").exists())
os.chdir(root)
from miscope.families.discovery import load_family_from_dir
fam = load_family_from_dir("data/modulo_addition_1layer", "data")

CASES = [
    ("p109 (CALIB escaper)", dict(prime=109, seed=485, data_seed=598), (18000, 26000), [(27000, 29000)]),
    ("p101/s999/ds999 (escaper)", dict(prime=101, seed=999, data_seed=999), (20000, 28000), [(28500, 34999)]),
    ("p103/s485/ds598 (non-escaper)", dict(prime=103, seed=485, data_seed=598), (16000, 22000), [(22000, 39999)]),
]


def power_traj(v, lo, hi):
    eps = np.array(v.artifacts.get_epochs("neuron_activations"))
    eps = eps[(eps >= lo) & (eps <= hi)]
    eps = eps[::2] if len(eps) > 250 else eps           # stride for I/O, keeps multi-100ep features
    stds = []
    for e in eps:
        a = v.artifacts.load_epoch("neuron_activations", int(e))["activations"].astype(np.float32)  # (N,p,p)
        stds.append(a.std(axis=(1, 2)))
    return eps, np.array(stds)                           # (E,N)


def dom_freq(v, e, j):
    a = v.artifacts.load_epoch("neuron_activations", int(e))["activations"][j].astype(np.float64)
    F = np.abs(np.fft.fft2(a - a.mean())); F[0, 0] = 0
    p = a.shape[0]; fa, fb = np.unravel_index(F.argmax(), F.shape)
    return (min(fa, p - fa), min(fb, p - fb))            # folded low-freq index


for tag, sel, plat, events in CASES:
    v = fam.get_variant(**sel)
    lo = min(plat[0], events[0][0]); hi = max(plat[1], events[-1][1])
    eps, S = power_traj(v, lo, hi)
    base = np.median(S[(eps >= plat[0]) & (eps <= plat[1])], axis=0) + 1e-9   # per-neuron plateau std
    em = np.zeros(len(eps), bool)
    for elo, ehi in events: em |= (eps >= elo) & (eps <= ehi)
    peakratio = (S[em] / base).max(0)                    # per-neuron max detonation in event window
    order = np.argsort(peakratio)[::-1][:6]
    print(f"\n=== {tag} ===  plateau {plat}  event {events}  ({len(eps)} epochs sampled)")
    print(f"  {'neuron':>6} {'detonation x':>13} {'plateau_std':>12} {'peak_std':>10} {'peak_ep':>8} {'dom2Dfreq':>11}")
    for j in order:
        pe = int(eps[em][np.argmax(S[em][:, j])])
        f = dom_freq(v, pe, j)
        print(f"  {j:>6} {peakratio[j]:>13.1f} {base[j]:>12.4f} {S[:, j].max():>10.3f} {pe:>8} {str(f):>11}")
    print(f"  -> max detonation {peakratio.max():.1f}x | "
          f"{'DETONATION present' if peakratio.max() > 8 else 'no detonation (≈stable power)'}")
