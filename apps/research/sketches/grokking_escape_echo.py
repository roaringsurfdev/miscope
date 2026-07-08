"""Test the 'curve vocabulary / echo' reading of the radial-escape plot (user, 2026-06-16):
grokking and the late regime-onset are the SAME dynamical motif (a radial reorganization transient),
fired twice. Predictions: (1) UNIVERSALITY — every variant shows a localized late echo at ~onset+24k
above its local plateau, muted in flat / explosive in escapers (a continuum of echo strength, not a
binary). (2) SELF-SIMILARITY — the late hump's SHAPE matches that variant's own grokking hump.
Param-side (W_in radial), no forward passes.
"""
import os
from pathlib import Path
import numpy as np

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "modulo_addition_1layer").exists())
os.chdir(root)
from miscope.families.discovery import load_family_from_dir
fam = load_family_from_dir("data/modulo_addition_1layer", "data")

def parse(n):
    import re; m = re.match(r"p(\d+)_seed(\d+)_dseed(\d+)", n); return dict(prime=int(m[1]), seed=int(m[2]), data_seed=int(m[3]))
def grok_onset(v):
    tl = np.asarray(v.test_losses); pk = float(tl.max()); pe = int(tl.argmax())
    h = np.where((np.arange(len(tl)) >= pe) & ((pk - tl) / pk >= 0.8))[0]; return int(h[0]) if len(h) else None
def max_radial(v):
    ps = np.array(v.artifacts.get_epochs("parameter_snapshot"))
    W = np.stack([v.artifacts.load_epoch("parameter_snapshot", int(e))["W_in"].T.astype(np.float64) for e in ps])
    w0 = W[:-1]; n0 = np.linalg.norm(w0, axis=2) + 1e-12
    rad = np.abs(((W[1:] - w0) * (w0 / n0[:, :, None])).sum(2)).max(1)
    k = 3; sm = np.convolve(rad, np.ones(k) / k, mode="same")          # light smooth
    return ps[:-1], sm

DECIDABLE = {"p109_seed485_dseed598": "ESC", "p101_seed999_dseed999": "ESC", "p107_seed999_dseed42": "ESC",
             "p103_seed485_dseed598": "flat", "p103_seed999_dseed598": "flat", "p109_seed485_dseed999": "flat",
             "p59_seed999_dseed598": "flat", "p109_seed485_dseed42": "flat", "p109_seed999_dseed42": "flat"}

def hump(mid, sm, lo, hi):
    m = (mid >= lo) & (mid <= hi)
    if m.sum() < 4: return np.nan, np.nan, None
    seg = sm[m]; e = mid[m]; i = int(seg.argmax()); return float(seg.max()), float(e[i]), (e, seg)

def resample(curve, N=40):
    e, s = curve; x = np.linspace(e[0], e[-1], N); y = np.interp(x, e, s)
    y = (y - y.min()) / (np.ptp(y) + 1e-12); return y

print(f"{'variant':>24}{'out':>5}{'grok_pk':>9}{'echo_pk':>9}{'plateau':>9}{'echo/plat':>10}{'shape_corr':>11}")
res = []
for name, out in DECIDABLE.items():
    v = fam.get_variant(**parse(name)); on = grok_onset(v); mid, sm = max_radial(v)
    gpk, gep, gcurve = hump(mid, sm, max(0, on - 2000), on + 8000)
    plat = np.median(sm[(mid >= on + 12000) & (mid <= on + 18000)]) + 1e-9
    epk, eep, ecurve = hump(mid, sm, on + 18000, mid[-1])
    corr = np.nan
    if gcurve and ecurve:
        gr, er = resample(gcurve), resample(ecurve); corr = float(np.corrcoef(gr, er)[0, 1])
    res.append((name, out, gpk, epk, plat, epk / plat, corr))
    print(f"{name:>24}{out:>5}{gpk:>9.3f}{epk:>9.3f}{plat:>9.4f}{epk/plat:>10.1f}{corr:>11.2f}")

esc = [r for r in res if r[1] == "ESC"]; flat = [r for r in res if r[1] == "flat"]
print(f"\nECHO STRENGTH (late peak / local plateau):  ESC {sorted(round(r[5],1) for r in esc)}  "
      f"flat {sorted(round(r[5],1) for r in flat)}")
print(f"  universality: all variants echo>plateau? min echo/plat = {min(r[5] for r in res):.1f} "
      f"(>1 = a localized late echo even in flat)")
print(f"  separation: max flat echo/plat {max(r[5] for r in flat):.1f}  vs  min ESC {min(r[5] for r in esc):.1f}")
print(f"SHAPE self-similarity (grok hump vs late hump):  ESC {sorted(round(r[6],2) for r in esc)}  "
      f"flat {sorted(round(r[6],2) for r in flat)}")
