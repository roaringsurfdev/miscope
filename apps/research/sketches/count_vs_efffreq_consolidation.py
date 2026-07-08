"""#2 consolidation: committed-COUNT vs EFFECTIVE-freq as escape predictors across the dense set.
For each variant: grok onset/class, learned-freq COUNT (variant_summary), eff-freq (power-weighted PR of
W_E@W_in per-neuron dominant frequency @~20k), max|radial| post-lock (escape proxy), run length, post-grok
runway, and an outcome class that makes the CENSORING confound explicit (escape only decidable when
runway > ~24k = the post-grok clock). Then: does COUNT separate the runway-CLEARED escapers from non-escapers,
and does eff-freq do worse?
"""
import os, json
from pathlib import Path
import numpy as np

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "modulo_addition_1layer").exists())
os.chdir(root)
from miscope.families.discovery import load_family_from_dir
fam = load_family_from_dir("data/modulo_addition_1layer", "data")
VROOT = Path("data/modulo_addition_1layer/variants")
CLOCK = 24000

def parse(name):
    import re
    m = re.match(r"p(\d+)_seed(\d+)_dseed(\d+)", name)
    return dict(prime=int(m[1]), seed=int(m[2]), data_seed=int(m[3]))

def eff_freq(v, target=20000):
    eps = np.array(v.artifacts.get_epochs("parameter_snapshot"))
    e = int(eps[np.argmin(np.abs(eps - target))])
    d = v.artifacts.load_epoch("parameter_snapshot", e)
    M = (d["W_E"].astype(np.float64) @ d["W_in"].astype(np.float64))   # (tokens, n_mlp)
    p = M.shape[0]
    F = np.fft.rfft(M - M.mean(0), axis=0); A = np.abs(F); A[0] = 0     # (freqs, n_mlp)
    dom = A.argmax(0); power = A.max(0)                                  # per-neuron dom freq + amplitude
    hist = np.zeros(A.shape[0])
    np.add.at(hist, dom, power)                                         # POWER-weighted histogram over freqs
    hist = hist[1:]                                                     # drop DC bin
    return float(hist.sum() ** 2 / (np.square(hist).sum() + 1e-12))

def max_radial_escapezone(v, onset):
    # ESCAPE zone only: onset+18k..last (past grok churn, brackets the ~24k clock). Excludes the
    # post-grok settling radial that a naive window mistakes for escape.
    eps = np.array(v.artifacts.get_epochs("parameter_snapshot"))
    lo = (onset or 0) + 18000
    use = eps[eps >= lo]
    if len(use) < 3: return np.nan, 0
    prev = None; mx = 0.0; mxep = None
    for e in use:
        w = v.artifacts.load_epoch("parameter_snapshot", int(e))["W_in"].T.astype(np.float64)
        if prev is not None:
            n0 = np.linalg.norm(prev, axis=1) + 1e-12
            r = np.abs(((w - prev) * (prev / n0[:, None])).sum(1)).max()
            if r > mx: mx, mxep = r, int(e)
        prev = w
    return mx, mxep

rows = []
for d in sorted(VROOT.iterdir()):
    sm = d / "variant_summary.json"
    if not sm.exists() or not (d / "artifacts/parameter_snapshot").exists(): continue
    s = json.load(open(sm)); onset = s.get("second_descent_onset_epoch")
    if onset is None: continue
    v = fam.get_variant(**parse(d.name))
    eps = np.array(v.artifacts.get_epochs("parameter_snapshot")); last = int(eps[-1])
    runway = last - onset
    mr, _ = max_radial_escapezone(v, onset)
    cls = "early" if onset < 9000 else "normal" if onset <= 12000 else "late"
    # escape decidable only if the run reaches the ~24k clock; else censored regardless of mr
    if runway < CLOCK: outcome = "flat-censored"
    elif mr > 0.1: outcome = "ESCAPED"
    else: outcome = "flat-CLEARED"
    rows.append(dict(name=d.name, onset=onset, cls=cls, count=s.get("learned_frequency_count"),
                     eff=eff_freq(v), maxrad=mr, last=last, runway=runway, outcome=outcome))

rows.sort(key=lambda r: (r["outcome"] != "ESCAPED", r["outcome"], -r["runway"]))
print(f"{'variant':>28} {'cls':>6} {'onset':>6} {'cnt':>3} {'eff':>5} {'max|rad|':>8} {'last':>6} {'runway':>7} {'outcome':>14}")
for r in rows:
    print(f"{r['name']:>28} {r['cls']:>6} {r['onset']:>6} {r['count']!s:>3} {r['eff']:>5.2f} "
          f"{r['maxrad']:>8.3f} {r['last']:>6} {r['runway']:>7} {r['outcome']:>14}")

# --- separation on the runway-CLEARED subset (escape is decidable) ---
cleared = [r for r in rows if r["outcome"] in ("ESCAPED", "flat-CLEARED")]
esc = [r for r in cleared if r["outcome"] == "ESCAPED"]; flat = [r for r in cleared if r["outcome"] == "flat-CLEARED"]
print(f"\n=== SEPARATION on runway-cleared subset (n={len(cleared)}: {len(esc)} escaped, {len(flat)} flat) ===")
for metric in ("count", "eff"):
    ev = sorted(r[metric] for r in esc); fv = sorted(r[metric] for r in flat)
    sep = (max(ev) < min(fv)) if (ev and fv) else None
    print(f"  {metric:>5}: escaped={ev}  flat={fv}  -> clean threshold separation: {sep}")
