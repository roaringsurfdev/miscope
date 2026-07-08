"""Is p103/s485 out of the woods, or sitting at a p109-style pre-collapse ceiling?
Three tail diagnostics:
 (1) the CORRECTED p109 collapse precursor — small-norm (0.2-0.5) W_in neurons with RISING |radial|
     (budding exploders). p109 showed it 18-26.9k pre-event (0.0006->0.0026, 3-4x). Present at p103/s485 tail?
 (2) test-loss spike ENVELOPE trend across 25-35k — still growing at the 34.7k cutoff (building) or flattening?
 (3) W_Q velocity trend at the very tail — stationary (gauge) or accelerating (building)?
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

# --- W_in radial + norm, streamed ---
prev = None; rad = []; nrm = []; wq_v = []
for e in ps:
    npz = variant.artifacts.load_epoch("parameter_snapshot", int(e))
    win = npz["W_in"].T.astype(np.float64)             # (512,128)
    wq = npz["W_Q"].astype(np.float64).reshape(4, -1)
    if prev is not None:
        w0 = prev["win"]; n0 = np.linalg.norm(w0, axis=1) + 1e-12
        dM = win - w0
        rad.append((dM * (w0 / n0[:, None])).sum(1))    # (512,) signed radial
        nrm.append(np.linalg.norm(w0, axis=1))
        wq_v.append(np.linalg.norm(wq - prev["wq"]) / (np.linalg.norm(prev["wq"]) + 1e-12))
    prev = {"win": win, "wq": wq}
rad = np.abs(np.array(rad)); nrm = np.array(nrm); wq_v = np.array(wq_v); mid = ps[:-1]

# (1) budding small-norm rising-radial cohort at the tail
def cohort_in(lo, hi, ref_ep=None):
    m = (mid >= lo) & (mid <= hi)
    ref = int(np.argmin(np.abs(mid - (ref_ep or lo))))
    small = (nrm[ref] >= 0.2) & (nrm[ref] <= 0.5)        # small-norm at window start
    early = (mid >= lo) & (mid <= (lo + hi) / 2)
    late = (mid > (lo + hi) / 2) & (mid <= hi)
    er = rad[early][:, small].mean(0); lr = rad[late][:, small].mean(0)
    ratio = lr / (er + 1e-9)
    idx = np.where(small)[0]
    budding = idx[(ratio > 2.0) & (lr > 0.003)]          # rising AND reaching meaningful level
    return small.sum(), budding, ratio

print("=== (1) CORRECTED COLLAPSE PRECURSOR — small-norm rising-radial W_in cohort ===")
for tag, lo, hi in [("p103/s485 TAIL 28-34.7k", 28000, 34700), ("p103/s485 mid 22-28k", 22000, 28000)]:
    nsmall, bud, ratio = cohort_in(lo, hi)
    print(f"  {tag}: {nsmall} small-norm neurons; BUDDING (radial ratio>2 & late>0.003): {list(bud)} "
          f"(max ratio {ratio.max():.2f})")
print("  p109 reference: budding cohort PRESENT 18-26.9k pre-collapse (small-norm |radial| rose 3-4x)")
print("  -> empty budding list = the p109 collapse precursor is ABSENT; non-empty = NOT out of the woods")

# (2) test-loss spike envelope trend
tl = np.asarray(variant.test_losses)
print("\n=== (2) test-loss spike ENVELOPE by 1k bin (max in bin) — growing or flattening at cutoff? ===")
last = len(tl)
for lo in range(25000, last, 1000):
    hi = min(lo + 1000, last)
    seg = tl[lo:hi]
    print(f"  [{lo:>5}-{hi:>5}]  max {seg.max():.3e}  median {np.median(seg):.3e}")

# (3) W_Q velocity trend at the very tail
print("\n=== (3) W_Q velocity trend (median/step by 1k bin) — stationary or accelerating? ===")
for lo in range(25000, int(mid[-1]), 1000):
    m = (mid >= lo) & (mid < lo + 1000)
    if m.any():
        print(f"  [{lo:>5}-{lo+1000:>5}]  W_Q vel median {np.median(wq_v[m]):.5f}  max {wq_v[m].max():.5f}")
print("  plateau(16-22k) W_Q vel median for reference:",
      f"{np.median(wq_v[(mid>=16000)&(mid<=22000)]):.5f}")
