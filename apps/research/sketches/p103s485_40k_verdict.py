"""p103/s485 extended to 40k — apply the locked out-of-woods decision rule:
 (A) W_Q velocity by 1k bin through 40k: does it ring DOWN to the ~0.001 plateau?
 (B) fresh small-norm rising-radial W_in cohort in the p101 serial-2nd-event window (~35.7k) and 37-40k?
 (C) any W_in radial escape post-35k (max|radial| vs floor / p109 0.62 / p101 0.97)?
 (D) the ~35k velocity spike the user saw — which stream, and quiet after?
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
print(f"param_snapshot {len(ps)} epochs {ps[0]}..{ps[-1]}  (onset 7699, predicted-escape 31.7k, serial-2nd ~35.7k)\n")

STREAMS = ["W_E", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"]


def units(npz, name):
    M = npz[name].astype(np.float64)
    if name == "W_in": return M.T
    if name in ("W_Q", "W_K", "W_V", "W_O"): return M.reshape(M.shape[0], -1)
    return M

prev = None
rad = []; nrm = []; vel = {s: [] for s in STREAMS}
for e in ps:
    npz = variant.artifacts.load_epoch("parameter_snapshot", int(e))
    cur = {s: units(npz, s) for s in STREAMS}
    if prev is not None:
        for s in STREAMS:
            vel[s].append(np.linalg.norm(cur[s] - prev[s]) / (np.linalg.norm(prev[s]) + 1e-12))
        w0 = prev["W_in"]; n0 = np.linalg.norm(w0, axis=1) + 1e-12
        rad.append(((cur["W_in"] - w0) * (w0 / n0[:, None])).sum(1)); nrm.append(np.linalg.norm(w0, axis=1))
    prev = cur
mid = ps[:-1]; rad = np.abs(np.array(rad)); nrm = np.array(nrm)
for s in STREAMS: vel[s] = np.array(vel[s])
plat = (mid >= 16000) & (mid <= 22000)

print("=== (C) W_in radial escape post-35k? ===")
post = mid >= 35000
mr = rad[post].max(); k = int(rad[post].max(1).argmax())
print(f"  max|radial| 35-40k = {mr:.4f} @ ep{int(mid[post][k])}  (floor ~0.001 | p109 0.62 | p101 0.97/1.18)"
      f"  -> {'ESCAPE' if mr>0.1 else 'no escape (≈floor)'}\n")

print("=== (B) fresh small-norm rising-radial cohort (the serial-2nd-event test) ===")
def budding(lo, hi):
    ref = int(np.argmin(np.abs(mid - lo)))
    small = (nrm[ref] >= 0.2) & (nrm[ref] <= 0.5)
    e = (mid >= lo) & (mid <= (lo+hi)/2); l = (mid > (lo+hi)/2) & (mid <= hi)
    er = rad[e][:, small].mean(0); lr = rad[l][:, small].mean(0); ratio = lr/(er+1e-9)
    idx = np.where(small)[0]
    return list(idx[(ratio > 2.0) & (lr > 0.003)]), float(ratio.max())
for lo, hi in [(34000, 37000), (37000, 39999)]:
    bud, mx = budding(lo, hi)
    print(f"  window {lo}-{hi}: budding cohort {bud}  (max ratio {mx:.2f})  "
          f"-> {'FRESH COHORT (not out)' if bud else 'none (gate holding)'}")

print("\n=== (A) W_Q velocity ring-down to plateau? (median/step by 1k bin) ===")
print(f"  plateau(16-22k) reference = {np.median(vel['W_Q'][plat]):.5f}")
for lo in range(30000, int(mid[-1]), 1000):
    m = (mid >= lo) & (mid < lo+1000)
    if m.any(): print(f"  [{lo:>5}-{lo+1000:>5}]  W_Q {np.median(vel['W_Q'][m]):.5f}")

print("\n=== (D) the ~35k velocity spike — per-stream max in 34.5-35.5k vs plateau ===")
spike = (mid >= 34500) & (mid <= 35500)
for s in STREAMS:
    print(f"  {s:>6}  spike-max {vel[s][spike].max():.5f}  ({vel[s][spike].max()/ (np.median(vel[s][plat])+1e-12):.1f}x plateau)")
