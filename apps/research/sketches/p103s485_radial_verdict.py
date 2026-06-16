"""Param-side radial-escape verdict for p103/s485/ds598 (the 3rd pre-registered
fragility-test escaper). Free: no forward passes, reads parameter_snapshot only.

Method matches p109/p113 weights-move notebooks (Kosson radial/tangential decomp of
W_in row steps). Prints the four extend/don't-extend signals + escape magnitude vs
the calibrated floor (~0.001) and the two confirmed escapers (p109 0.62, p101 0.97/1.18).
"""
import os
from pathlib import Path
import numpy as np

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "modulo_addition_1layer").exists())
os.chdir(root)
from miscope.families.discovery import load_family_from_dir

fam = load_family_from_dir("data/modulo_addition_1layer", "data")
variant = fam.get_variant(prime=103, seed=485, data_seed=598)


def grok_onset(v):
    tl = np.asarray(v.test_losses); peak = float(tl.max()); pe = int(tl.argmax())
    hit = np.where((np.arange(len(tl)) >= pe) & ((peak - tl) / peak >= 0.8))[0]
    return (int(hit[0]) if len(hit) else None), peak, pe


onset, peak, peak_ep = grok_onset(variant)
predicted = onset + 24000

ps = np.array(variant.artifacts.get_epochs("parameter_snapshot"))
W = np.stack([variant.artifacts.load_epoch("parameter_snapshot", int(e))["W_in"].T.astype(np.float64) for e in ps])
norm = np.linalg.norm(W, axis=2)                       # (E, N)
w0, w1 = W[:-1], W[1:]
n0 = np.linalg.norm(w0, axis=2) + 1e-12
ang = np.degrees(np.arccos(np.clip((w0 * w1).sum(2) / (n0 * (np.linalg.norm(w1, axis=2) + 1e-12)), -1, 1)))
dw = w1 - w0
radial = (dw * (w0 / n0[:, :, None])).sum(2)           # (E-1, N) signed norm change
tang = np.sqrt(np.clip((dw ** 2).sum(2) - radial ** 2, 0, None))
mid = ps[:-1]
N = W.shape[1]

print(f"=== p103/s485/ds598 radial verdict ===")
print(f"onset {onset} (peak {peak:.1f}@{peak_ep}); PRE-REGISTERED escape = onset+24k = {predicted}")
print(f"parameter_snapshot available: {len(ps)} epochs, {ps[0]}..{ps[-1]}  "
      f"({'COVERS' if ps[-1] >= predicted else 'SHORT OF'} predicted {predicted})\n")

# Plateau calibration (post-lock, pre any event window): 15k..min(25k, predicted-2k)
plat = (mid >= 15000) & (mid <= min(25000, predicted - 2000))
tf = tang / (np.sqrt(radial ** 2 + tang ** 2) + 1e-12)
print(f"PLATEAU (15k-{min(25000, predicted-2000)}) equilibrium:")
print(f"  median |radial|/step = {np.median(np.abs(radial[plat])):.4f}  (calib floor ~0.0009; clean p113/p103s999)")
print(f"  median angular       = {np.median(ang[plat]):.3f} deg/step   median tang.frac = {np.median(tf[plat]):.2f}")
print(f"  median ||W_in[j]||    = {np.median(norm[:-1][plat]):.3f}\n")

# --- escape magnitude (post-lock) ---
post = mid >= 15000
flat_idx = np.abs(radial[post]).argmax()
ep_idx, neuron = np.unravel_index(flat_idx, radial[post].shape)
maxrad = np.abs(radial[post]).max()
maxrad_ep = int(mid[post][ep_idx])
print(f"MAX |radial| post-lock = {maxrad:.3f}  @ epoch {maxrad_ep}, neuron n{neuron}")
print(f"  vs floor ~0.001 | p109 escape 0.620 | p101 escapes 0.97/1.18")
print(f"  verdict: {'ESCAPE (>> floor)' if maxrad > 0.1 else 'NO escape (≈floor)' if maxrad < 0.02 else 'ELEVATED (intermediate)'}\n")

# --- exploder cohort (top-11 by post-lock max |radial|) ---
peakrad = np.abs(radial[post]).max(0)
cohort = np.argsort(peakrad)[::-1][:11]
i_plat = int(np.argmin(np.abs(mid - 22000)))   # plateau norm reference
print("EXPLODER COHORT (top-11 by post-lock max|radial|):")
print(f"  {'n':>4} {'max|rad|':>9} {'peak_ep':>8} {'norm@22k':>9} {'norm_peak':>10} {'norm@tail':>10} {'dormant?':>9}")
peak_eps = []
for j in cohort:
    jp = int(mid[post][np.abs(radial[post][:, j]).argmax()])
    peak_eps.append(jp)
    npre, npk, ntl = norm[i_plat, j], norm[:, j].max(), norm[-1, j]
    print(f"  {j:>4} {peakrad[j]:>9.3f} {jp:>8} {npre:>9.3f} {npk:>10.3f} {ntl:>10.3f} "
          f"{'YES' if npre < 0.1 else 'no':>9}")
peak_eps = np.array(peak_eps)
print(f"\n  synchronized? peak-radial epochs span {peak_eps.max()-peak_eps.min()} epochs "
      f"[{peak_eps.min()}..{peak_eps.max()}]  ({'SYNC' if peak_eps.max()-peak_eps.min()<=500 else 'spread/serial'})")

# --- output-subspace convergence (W_out rows of cohort, current tail) ---
Wout = variant.artifacts.load_epoch("parameter_snapshot", int(ps[-1]))["W_out"].astype(np.float64)  # (N,128)
v = Wout[cohort]; v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
cos = np.abs(v @ v.T); offdiag = cos[np.triu_indices(len(cohort), 1)]
# random baseline
rng = np.random.default_rng(0)
rnd = rng.standard_normal((len(cohort), Wout.shape[1])); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
rcos = np.abs(rnd @ rnd.T)[np.triu_indices(len(cohort), 1)]
print(f"  W_out convergence: cohort mean|cos| {offdiag.mean():.3f} vs random {rcos.mean():.3f} "
      f"({'CONVERGENT out-subspace' if offdiag.mean() > rcos.mean()*1.4 else 'diffuse/independent'})")

# --- TAIL: still climbing? (extend signal #1) ---
maxrad_t = np.abs(radial).max(1)   # per-step max over neurons
k = min(8, len(mid))
recent, prior = maxrad_t[-k:], maxrad_t[-2*k:-k]
print(f"\nTAIL TREND (extend signal): max|radial| last {k} steps mean {recent.mean():.3f} "
      f"vs prior {k} {prior.mean():.3f}  ->  "
      f"{'STILL CLIMBING -> EXTEND' if recent.mean() > prior.mean()*1.15 else 'peaked/declining (ring-down)' if recent.mean() < prior.mean()*0.7 else 'flat/plateau'}")
print(f"  cohort norm@tail vs peak: {np.mean([norm[-1,j]/max(norm[:,j].max(),1e-9) for j in cohort]):.2f} "
      f"(1.0 = mid-detonation/not settled; <0.8 = relaxing)")
print(f"  last available epoch {ps[-1]} vs predicted escape {predicted}: "
      f"{'tail past prediction' if ps[-1] > predicted else f'{predicted-ps[-1]} epochs SHORT of prediction'}")
