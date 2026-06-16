"""Localize the p103/s485 late test-loss roughening: per-stream weight velocity +
radial/tangential decomposition across ALL weight matrices (not just W_in), compared
in three windows — plateau, the 24-25k attention-DMD excursion, the 30-35k roughening tail.

Answers: is the roughening weight-side at all, and which stream? W_in radial was flat at
the tail (0.002), so if W_out/attention are also flat the roughening is activation/numeric,
not a weight reorganization. Free: no forward passes. Streams epochs to keep memory low.
"""
import os
from pathlib import Path
import numpy as np

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "modulo_addition_1layer").exists())
os.chdir(root)
from miscope.families.discovery import load_family_from_dir

fam = load_family_from_dir("data/modulo_addition_1layer", "data")
variant = fam.get_variant(prime=103, seed=485, data_seed=598)
ps = np.array(variant.artifacts.get_epochs("parameter_snapshot"))


def units(npz, name):
    """Return (U, D) array of per-unit weight vectors for a stream."""
    M = npz[name].astype(np.float64)
    if name == "W_in":   return M.T                      # (512,128) neuron input rows
    if name == "W_out":  return M                        # (512,128) neuron output rows
    if name == "W_U":    return M.T                      # (103,128) per-token unembed
    if name == "W_E":    return M                        # (104,128) per-token embed
    if name in ("W_Q", "W_K", "W_V"):  return M.reshape(M.shape[0], -1)   # (4, 4096) per head
    if name == "W_O":    return M.reshape(M.shape[0], -1)                  # (4, 4096) per head
    return M.reshape(M.shape[0], -1)


STREAMS = ["W_E", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"]
vel = {s: [] for s in STREAMS}        # relative Frobenius velocity / step
maxrad = {s: [] for s in STREAMS}     # max |radial| over units / step
maxrad_unit = {s: [] for s in STREAMS}

prev = None
for e in ps:
    npz = variant.artifacts.load_epoch("parameter_snapshot", int(e))
    cur = {s: units(npz, s) for s in STREAMS}
    if prev is not None:
        for s in STREAMS:
            u0, u1 = prev[s], cur[s]
            dM = u1 - u0
            vel[s].append(np.linalg.norm(dM) / (np.linalg.norm(u0) + 1e-12))
            n0 = np.linalg.norm(u0, axis=1) + 1e-12
            rad = (dM * (u0 / n0[:, None])).sum(1)       # signed per-unit radial
            j = int(np.abs(rad).argmax())
            maxrad[s].append(abs(rad[j])); maxrad_unit[s].append(j)
    prev = cur

mid = ps[:-1]
for s in STREAMS:
    vel[s] = np.array(vel[s]); maxrad[s] = np.array(maxrad[s]); maxrad_unit[s] = np.array(maxrad_unit[s])

WIN = {"plateau(16-22k)": (16000, 22000), "attn-excursion(23.5-26k)": (23500, 26000),
       "rough-tail(30-34.7k)": (30000, 34700)}


def wstat(arr, lo, hi):
    m = (mid >= lo) & (mid <= hi)
    return arr[m].max() if m.any() else np.nan


print("=== p103/s485 roughening localization — RELATIVE WEIGHT VELOCITY (max/step in window) ===")
print(f"{'stream':>7} | " + " | ".join(f"{w:>22}" for w in WIN))
for s in STREAMS:
    print(f"{s:>7} | " + " | ".join(f"{wstat(vel[s], *r):>22.5f}" for r in WIN.values()))

print("\n=== MAX |radial| over units (max/step in window) — norm-changing motion ===")
print(f"{'stream':>7} | " + " | ".join(f"{w:>22}" for w in WIN))
for s in STREAMS:
    print(f"{s:>7} | " + " | ".join(f"{wstat(maxrad[s], *r):>22.5f}" for r in WIN.values()))

# Ratios: how elevated is each window vs plateau, per stream (velocity)
print("\n=== velocity ELEVATION vs plateau (window_max / plateau_max) ===")
print(f"{'stream':>7} | {'attn-excursion':>16} | {'rough-tail':>16}")
for s in STREAMS:
    base = wstat(vel[s], 16000, 22000)
    print(f"{s:>7} | {wstat(vel[s], 23500, 26000)/base:>16.2f} | {wstat(vel[s], 30000, 34700)/base:>16.2f}")

# Which stream dominates the tail roughening + its top unit
print("\n=== TAIL (30-34.7k) — most-active stream by max|radial| and its unit ===")
tail = (mid >= 30000) & (mid <= 34700)
for s in STREAMS:
    a = maxrad[s][tail]
    if a.size:
        k = int(a.argmax()); ep = int(mid[tail][k]); u = int(maxrad_unit[s][tail][k])
        print(f"  {s:>7}  max|radial| {a.max():>8.5f} @ ep{ep}  unit {u}")
