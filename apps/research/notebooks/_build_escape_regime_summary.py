"""Generator for escape_regime_summary.ipynb — the consolidated map of the late-reorganization
('escape') regime. Run: `uv run python apps/research/notebooks/_build_escape_regime_summary.py`.
Outputs left unsaved (per user notebook pref); execute interactively to render figures.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# The late-reorganization ("escape") regime — consolidated map

A grokked, fully-solved toy transformer can **spontaneously destabilize ~24k steps after grokking**
(loss → ~3, ~11 neurons detonate ~800× in activation) and then **self-heal**, function round-tripping
to ~1e-7. This notebook consolidates what we know across the dense `modulo_addition_1layer` set.

**The two-layer picture this notebook builds:**
- **Surface / trigger layer** (indexed by *eff-freq*, power concentration): upstream attention-pattern (QK)
  drift, spokes/wedges, a budding small-norm cohort. Present in escapers *and* in the absorbed non-escaper.
- **The MLP gate** (set by *committed-frequency count*): ≤3 → the trigger propagates and detonates; ≥4 →
  permanently absorbed.
- **Downstream / functional layer** (escape-specific): W_in radial escape, MLP-out DMD excursion, activation
  detonation. Only in true escapers.

Method note: escape is a *late, post-plateau* event — measure radial in an **escape-zone window** (onset+18k,
bracketing the ~24k clock), never a naive post-grok window (that mistakes grok-churn for escape).""")

code("""import os, json, re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "modulo_addition_1layer").exists())
os.chdir(root)
from miscope.families.discovery import load_family_from_dir
fam = load_family_from_dir("data/modulo_addition_1layer", "data")
VROOT = Path("data/modulo_addition_1layer/variants")
CLOCK = 24000

def parse(name):
    m = re.match(r"p(\\d+)_seed(\\d+)_dseed(\\d+)", name)
    return dict(prime=int(m[1]), seed=int(m[2]), data_seed=int(m[3]))

def grok_onset(v):
    tl = np.asarray(v.test_losses); peak = float(tl.max()); pe = int(tl.argmax())
    hit = np.where((np.arange(len(tl)) >= pe) & ((peak - tl) / peak >= 0.8))[0]
    return int(hit[0]) if len(hit) else None

def radial_decomp(v):
    # Kosson radial/tangential decomposition of W_in row steps (matches the weights-move notebooks).
    ps = np.array(v.artifacts.get_epochs("parameter_snapshot"))
    W = np.stack([v.artifacts.load_epoch("parameter_snapshot", int(e))["W_in"].T.astype(np.float64) for e in ps])
    w0, w1 = W[:-1], W[1:]; n0 = np.linalg.norm(w0, axis=2) + 1e-12
    dw = w1 - w0
    radial = (dw * (w0 / n0[:, :, None])).sum(2)
    return ps[:-1], np.abs(radial)

def eff_freq(v, target=20000):
    # power-weighted PR of W_E@W_in per-neuron dominant frequency: effective # of frequencies used.
    eps = np.array(v.artifacts.get_epochs("parameter_snapshot")); e = int(eps[np.argmin(np.abs(eps - target))])
    d = v.artifacts.load_epoch("parameter_snapshot", e)
    A = np.abs(np.fft.rfft((d["W_E"].astype(np.float64) @ d["W_in"].astype(np.float64)), axis=0)); A[0] = 0
    h = np.zeros(A.shape[0]); np.add.at(h, A.argmax(0), A.max(0)); h = h[1:]
    return float(h.sum() ** 2 / (np.square(h).sum() + 1e-12))

# decidable outcomes (grokkers with post-grok runway past the ~24k clock, escape-zone radial measured)
DECIDABLE = {"p109_seed485_dseed598": "escaped", "p101_seed999_dseed999": "escaped",
             "p107_seed999_dseed42": "escaped",                       # count-4 escaper (the counterexample)
             "p103_seed485_dseed598": "flat", "p103_seed999_dseed598": "flat",
             "p109_seed485_dseed999": "flat",                         # count-4 flat; smooth-rise morphology
             "p59_seed999_dseed598": "flat", "p109_seed485_dseed42": "flat",
             "p109_seed999_dseed42": "flat"}                          # three count-3 flats (falsified the gate)
CLR = {"escaped": "#d62728", "flat": "#2ca02c", "censored": "#cccccc"}""")

md("""## §1 — The regime at the behavioral level

The two escapers spike post-grok and **round-trip** (function returns to ~1e-7); the two non-escapers stay
solved throughout.""")
code("""fig = go.Figure()
for name, out in DECIDABLE.items():
    tl = np.asarray(fam.get_variant(**parse(name)).test_losses)
    fig.add_trace(go.Scatter(x=np.arange(len(tl)), y=tl, mode="lines",
                             name=f"{name} ({out})", line=dict(color=CLR[out], width=1)))
fig.update_layout(title="§1 — escapers spike + round-trip post-grok; non-escapers stay solved",
                  xaxis_title="epoch", yaxis_title="test loss", yaxis_type="log", height=440,
                  legend=dict(x=0.01, y=0.01))
fig.show()""")

md("""## §2 — The weight-level mechanism: radial escape

Per neuron, decompose each W_in row step into **radial** (norm-changing) and tangential (rotation). On the
plateau the median |radial| sits at a tiny equilibrium **floor (~0.001)** — late motion is gauge rotation.
The escapers break that floor by ~600–1000× in the escape zone; the non-escapers never leave it.""")
code("""fig = go.Figure(); summary = []
for name, out in DECIDABLE.items():
    v = fam.get_variant(**parse(name)); onset = grok_onset(v)
    mid, rad = radial_decomp(v)
    fig.add_trace(go.Scatter(x=mid, y=rad.max(1), mode="lines", name=f"{name} ({out})", line=dict(color=CLR[out])))
    zone = mid >= onset + 18000; plat = (mid >= 15000) & (mid <= 22000)
    summary.append((name, out, onset, float(np.median(rad[plat])), float(rad[zone].max()) if zone.any() else np.nan))
fig.add_hline(y=0.1, line=dict(dash="dot", color="gray"), annotation_text="escape threshold")
fig.update_layout(title="§2 — radial escape (max |W_in row norm-change| / step)", xaxis_title="epoch",
                  yaxis_title="max |radial| / step", yaxis_type="log", height=440, legend=dict(x=0.01, y=0.99))
fig.show()
print(f"{'variant':>26}{'outcome':>9}{'onset':>7}{'plateau|rad|':>14}{'escape-zone max':>16}")
for n, o, on, pl, mr in summary: print(f"{n:>26}{o:>9}{on:>7}{pl:>14.4f}{mr:>16.3f}")""")

md("""## §3 — The predictor question is OPEN: no static scalar gates escape (n=9)

**A retraction.** On the first 4 decidable variants, committed **count** looked like a clean gate (3→escape,
4→flat) and eff-freq did not. With 9 decidable variants that 2×2 collapsed: a **count-4 variant escaped hard**
(p107/999/42, max|radial| 0.75) and **three count-3 early grokkers stayed flat**. Below, escapers (red) and
non-escapers (green) span the *same* count (3–4) and eff-freq (2.3–3.2) ranges — no separation on either axis.
Onset, margin, prime and seed fail too. **We can predict the regime exists and round-trips; we cannot yet
predict *which model* escapes from static 20k-weight structure** — the predictor is likely *dynamic* (the
budding small-norm cohort → dormant detonation), not a static scalar. Grey = censored (insufficient runway).""")
code("""pts = []
for d in sorted(VROOT.iterdir()):
    sm = d / "variant_summary.json"
    if not (sm.exists() and (d / "artifacts/parameter_snapshot").exists()): continue
    s = json.load(open(sm)); onset = s.get("second_descent_onset_epoch")
    if onset is None: continue
    v = fam.get_variant(**parse(d.name))
    pts.append((d.name, s.get("learned_frequency_count"), eff_freq(v), DECIDABLE.get(d.name, "censored")))
rng = np.random.default_rng(0)
fig = go.Figure()
for out in ["censored", "flat", "escaped"]:
    g = [p for p in pts if p[3] == out]
    fig.add_trace(go.Scatter(x=[p[1] + rng.uniform(-0.13, 0.13) for p in g], y=[p[2] for p in g],
                  mode="markers", name=out, text=[p[0] for p in g],
                  marker=dict(color=CLR[out], size=12 if out != "censored" else 7,
                              line=dict(width=1, color="#333"))))
fig.add_vline(x=3.5, line=dict(dash="dot", color="#bbb"),
              annotation_text="count=3.5 (falsified gate: count-4 escapes, count-3 stay flat)")
fig.update_layout(title="§3 — neither count (x) nor eff-freq (y) separates escapers from non-escapers (n=9)",
                  xaxis_title="learned-frequency COUNT", yaxis_title="eff-freq (power-weighted PR)", height=480)
fig.show()
print("decidable:", sorted([(p[0], p[1], round(p[2], 2), p[3]) for p in pts if p[3] != "censored"]))""")

md("""## §4 — The absorbed case (p103/s485): trigger present, amplification gated

p103/s485 (count 4) shows the *surface* trigger — the attention pattern (W_Q/W_K) reorganizes 5–6× plateau
from ~24k — while the **MLP (W_in/W_out) stays flat**. The perturbation is absorbed, never propagating into
the functional core. This is the variant that draws the line between the two layers.""")
code("""v = fam.get_variant(prime=103, seed=485, data_seed=598)
ps = np.array(v.artifacts.get_epochs("parameter_snapshot"))
streams = ["W_Q", "W_K", "W_in", "W_out"]; vel = {s: [] for s in streams}; prev = None
for e in ps:
    d = v.artifacts.load_epoch("parameter_snapshot", int(e))
    cur = {s: (d[s].astype(np.float64).reshape(d[s].shape[0], -1) if d[s].ndim > 2 else d[s].astype(np.float64)) for s in streams}
    if prev is not None:
        for s in streams: vel[s].append(np.linalg.norm(cur[s] - prev[s]) / (np.linalg.norm(prev[s]) + 1e-12))
    prev = cur
mid = ps[:-1]
fig = go.Figure()
for s, c in [("W_Q", "#1f77b4"), ("W_K", "#17becf"), ("W_in", "#d62728"), ("W_out", "#ff7f0e")]:
    fig.add_trace(go.Scatter(x=mid, y=vel[s], mode="lines", name=s, line=dict(color=c)))
fig.update_layout(title="§4 — p103/s485 absorbed: attention QK reorganizes (5–6×), MLP stays flat",
                  xaxis_title="epoch", yaxis_title="relative weight velocity / step", height=440,
                  legend=dict(x=0.01, y=0.99))
fig.show()""")

md("""## §5 — An escape-specific instrument: activation-power detonation

The p109 §7 signature — a neuron's activation **power detonating** — recurs in the other escaper
(p101, in a *dormant* flavor: dead neurons → std 5–8) and is **absent** in the absorbed non-escaper.
A *true detonator* needs **both** a high power *ratio* (vs its plateau) **and** high absolute power: escapers
have several; p103's highest-ratio neuron is a dead neuron that stays near-zero (high ratio, no power). This is
a downstream/functional instrument that, unlike the surface signatures, fires only on real escapes.""")
code("""def detonation_stats(name, plat, event):
    v = fam.get_variant(**parse(name)); eps = np.array(v.artifacts.get_epochs("neuron_activations"))
    def std_at(es):
        return np.array([v.artifacts.load_epoch("neuron_activations", int(e))["activations"].std(axis=(1, 2)) for e in es])
    base = np.median(std_at(eps[(eps >= plat[0]) & (eps <= plat[1])][::4]), axis=0) + 1e-9
    peak = std_at(eps[(eps >= event[0]) & (eps <= event[1])][::3]).max(axis=0)
    ratio = peak / base
    n_true = int(((ratio > 8) & (peak > 2)).sum())          # both high ratio AND high absolute power
    return float(ratio.max()), float(peak[ratio.argmax()]), n_true

cases = [("p109_seed485_dseed598", (18000, 26000), (27000, 29000), "escaped"),
         ("p101_seed999_dseed999", (20000, 28000), (28500, 34999), "escaped"),
         ("p103_seed485_dseed598", (16000, 22000), (22000, 39999), "absorbed")]
rows = []
print(f"{'variant':>26}{'outcome':>9}{'max ratio':>11}{'pwr@max-ratio':>14}{'#true detonators':>17}")
for nm, pl, ev, out in cases:
    mr, pw, nd = detonation_stats(nm, pl, ev); rows.append((nm, out, nd))
    print(f"{nm:>26}{out:>9}{mr:>11.1f}{pw:>14.2f}{nd:>17}")
fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[2] for r in rows], text=[r[2] for r in rows],
                       textposition="outside",
                       marker_color=[CLR["escaped"] if r[1] == "escaped" else CLR["flat"] for r in rows]))
fig.update_layout(title="§5 — true detonators (power ratio>8 AND peak std>2): escapers many, absorbed case zero",
                  yaxis_title="# neurons that detonate", height=420)
fig.show()""")

md("""## §6 — Severity: the margin hypothesis (n=3, weak)

*Among the escapers*, does the margin (count − eff-freq) modulate *how hard* it blows? The 3 escapers:
p101/s999/ds999 (margin ~0.0) → 1.18; p107/s999/ds42 (margin ~0.9) → 0.75; p109/s485/ds598 (margin 0.66) → 0.62.
The lowest-margin one blows hardest, but p107 (high margin) exceeds p109 — so the trend is weak and noisy at n=3,
and confounded (dormant vs structured exploders; different primes). Open, not load-bearing.

Note this is a *severity* question *conditional on escaping* — it is separate from the (now-falsified) question of
*which* models escape, which §3 shows no static scalar answers.""")

md("""## §7 — What's earned, falsified, and open

**Earned (load-bearing):**
- The regime is real and reversible (weights/activations explode ~800×; function round-trips to ~1e-7).
- Prospective prediction of the *regime itself* (a late post-grok escape that round-trips) confirmed on multiple
  variants — the *phenomenon* was called before the extension data existed.
- **Two-layer mechanism:** an upstream attention/embedding trigger vs downstream MLP amplification; escape-specific
  instruments (W_in radial / MLP-out DMD / activation detonation) fire only on real escapes, while surface
  signatures (spokes/wedges/attention drift/budding cohort) appear in non-escapers too (the p103/s485 case, §4–5).
- The event is **silent in every aggregate diagnostic** tested — slingshot (‖W_U‖ flat), attention-entropy
  collapse, softmax collapse (fp64 loss), Wortsman logit-divergence (log Z pinned), median-norm climb. Only
  per-neuron / per-circuit instruments see it (global→local migration thesis, well-supported *here*).

**Falsified / retracted:**
- **Committed-count as the escape gate** (and few-eff-freq sufficiency). The clean n=4 2×2 was a small-sample
  coincidence: at n=9 a count-4 variant escapes hard and three count-3 early grokkers stay flat (§3). No static
  20k-weight scalar — count, eff-freq, margin, onset, prime, seed — separates escapers from non-escapers.
- The neuron-level *pinpoint* precursor (rotating-stall) — actual exploders are dormant capacity.

**Not yet / open:**
- **Which model escapes is unpredicted from static structure.** Leading hypothesis: the predictor is *dynamic* —
  the budding small-norm rising-radial cohort that goes on to detonate dormant capacity. Next test: does that
  cohort form in the escapers and not the non-escapers, across the decidable set?
- A third morphology exists (p109/s485/ds999): a *smooth* post-35k test-loss rise with attention anomalies but no
  radial escape — gauge/attention drift slowly eroding the readout, distinct from the discrete event.
- Severity (margin) is n=3 and weak (§6).
- Safety relevance is a hypothesis to earn (toy model): the bet is that at scale a localized reorganization like
  this goes silent in aggregate metrics, and standing local instruments are how you'd see it coming.""")

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
out = "apps/research/notebooks/escape_regime_summary.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
