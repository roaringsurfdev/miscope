"""Generator for attention_divergence.ipynb — tests the user's hypothesis (2026-06-16):
attention's signature is 'divergence from the pack' — the attention (QK pattern) weight group
DIFFERENTIATES from the embedding+MLP groups at regime boundaries, plausibly compensating for
drift between embeddings and MLP. Operationalized as a *relative* (differential) velocity index.

Divergence index D(epoch) = vel(attn QK) / median(vel(W_E), vel(W_in), vel(W_out)).
Param-side, no forward passes. Built via nbformat; outputs unsaved (user pref).
"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Attention as 'divergence from the pack'

Hypothesis (2026-06-16): the load-bearing attention signature for the escape regime is not that attention
*moves*, but that it **differentiates from the other weight groups** — specifically the embedding + MLP
'pack' it sits between — possibly as the adaptive element compensating for drift between them.

We make this a *relative* quantity: **divergence index** D = vel(attn QK) / median(vel(W_E), vel(W_in),
vel(W_out)). D≈1 → attention moves with the pack; D≫1 → attention pulls away. We ask:
1. In the **matched count-4 pair** (p107/999/42 escapes vs p103/485/598 absorbs), does attention diverge
   from the pack only in the escaper? (count held fixed — closest thing to a causal test.)
2. Does D **spike at regime boundaries** (grokking and the late escape)?
3. Does escape-zone D **separate** escapers from non-escapers across the decidable set — i.e. is it a
   candidate regime-boundary signature where the static scalars (count, eff-freq) failed?""")

code("""import os, re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "modulo_addition_1layer").exists())
os.chdir(root)
from miscope.families.discovery import load_family_from_dir
fam = load_family_from_dir("data/modulo_addition_1layer", "data")

def parse(n):
    m = re.match(r"p(\\d+)_seed(\\d+)_dseed(\\d+)", n); return dict(prime=int(m[1]), seed=int(m[2]), data_seed=int(m[3]))
def grok_onset(v):
    tl = np.asarray(v.test_losses); pk = float(tl.max()); pe = int(tl.argmax())
    h = np.where((np.arange(len(tl)) >= pe) & ((pk - tl) / pk >= 0.8))[0]; return int(h[0]) if len(h) else None

GROUPS = ["W_E", "W_Q", "W_K", "W_in", "W_out", "W_V", "W_O"]
def velocities(v):
    # relative Frobenius velocity per weight group, one streaming pass.
    ps = np.array(v.artifacts.get_epochs("parameter_snapshot")); prev = None; vel = {g: [] for g in GROUPS}
    for e in ps:
        d = v.artifacts.load_epoch("parameter_snapshot", int(e))
        cur = {g: d[g].astype(np.float64).reshape(-1) for g in GROUPS}
        if prev is not None:
            for g in GROUPS: vel[g].append(np.linalg.norm(cur[g] - prev[g]) / (np.linalg.norm(prev[g]) + 1e-12))
        prev = cur
    return ps[:-1], {g: np.array(vel[g]) for g in vel}
def divergence(vel):
    attn = 0.5 * (vel["W_Q"] + vel["W_K"])                       # the QK pattern circuit
    pack = np.median(np.stack([vel["W_E"], vel["W_in"], vel["W_out"]]), axis=0) + 1e-12
    return attn / pack

DECIDABLE = {"p109_seed485_dseed598": "ESC", "p101_seed999_dseed999": "ESC", "p107_seed999_dseed42": "ESC",
             "p103_seed485_dseed598": "flat", "p103_seed999_dseed598": "flat", "p109_seed485_dseed999": "flat",
             "p59_seed999_dseed598": "flat", "p109_seed485_dseed42": "flat", "p109_seed999_dseed42": "flat"}
CLR = {"ESC": "#d62728", "flat": "#2ca02c"}""")

md("""## §1 — The matched count-4 pair: does attention leave the pack only in the escaper?

p107/999/42 (count 4, ESCAPED) vs p103/485/598 (count 4, ABSORBED). Count is held fixed, so a difference in
attention's divergence from the embedding+MLP pack is closer to causal. Per-group velocity; watch whether
W_Q/W_K (attention pattern) separate from W_E/W_in/W_out in the escape zone of the escaper.""")
code("""from plotly.subplots import make_subplots
pair = [("p107_seed999_dseed42", "ESCAPED"), ("p103_seed485_dseed598", "ABSORBED")]
fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{n} ({o})" for n, o in pair], shared_yaxes=True)
COL = {"W_Q": "#1f77b4", "W_K": "#17becf", "W_E": "#9467bd", "W_in": "#d62728", "W_out": "#ff7f0e"}
for j, (name, _) in enumerate(pair, 1):
    v = fam.get_variant(**parse(name)); mid, vel = velocities(v)
    for g in ["W_Q", "W_K", "W_E", "W_in", "W_out"]:
        fig.add_trace(go.Scatter(x=mid, y=vel[g], mode="lines", name=g, line=dict(color=COL[g]),
                                 showlegend=(j == 1)), row=1, col=j)
fig.update_yaxes(type="log"); fig.update_layout(height=430,
    title="§1 — matched count-4 pair: attention (blue/cyan) pulls away from the pack in the escaper only")
fig.show()""")

md("""## §2 — Divergence index D over training: does it spike at the escape?

D = vel(attn QK) / median(embedding, MLP). Overlaid for all 9 decidable variants, marked at each variant's
escape clock (onset+24k). Escapers in red, non-escapers green.""")
code("""fig = go.Figure(); zone = []
for name, out in DECIDABLE.items():
    v = fam.get_variant(**parse(name)); on = grok_onset(v); mid, vel = velocities(v)
    D = divergence(vel); k = 5; Ds = np.convolve(D, np.ones(k) / k, mode="same")
    fig.add_trace(go.Scatter(x=mid, y=Ds, mode="lines", name=f"{name} ({out})",
                             line=dict(color=CLR[out], width=1.3 if out == "ESC" else 1)))
    z = (mid >= on + 18000); zone.append((name, out, float(D[z].max()) if z.any() else np.nan,
                                          float(np.median(D[(mid >= on + 12000) & (mid <= on + 18000)]))))
fig.add_hline(y=1.0, line=dict(dash="dot", color="gray"), annotation_text="D=1 (moves with pack)")
fig.update_layout(title="§2 — attention divergence index D over training (red=escaper, green=flat)",
                  xaxis_title="epoch", yaxis_title="D = attn-QK vel / pack median", yaxis_type="log", height=450)
fig.show()""")

md("""## §3 — Is escape-zone divergence a candidate separator? (with multiple-comparisons humility)

The static scalars (count, eff-freq) failed at n=9. If escape-zone D separates here, it is a **candidate
regime-boundary signature** to validate on more data — NOT a declared gate (n=9 with a 3/6 split spuriously
separates easily; that is exactly how the count gate burned). The honest bar: does it separate *and* is it
mechanistically grounded (the matched-pair test in §1)?""")
code("""print(f"{'variant':>24}{'out':>5}{'escZone Dmax':>13}{'preEsc Dmed':>13}{'ratio':>8}")
esc, flat = [], []
for name, out, dmax, dmed in zone:
    r = dmax / (dmed + 1e-9); (esc if out == "ESC" else flat).append(r)
    print(f"{name:>24}{out:>5}{dmax:>13.1f}{dmed:>13.2f}{r:>8.1f}")
print(f"\\nescape-zone D-rise:  ESC {sorted(round(x,1) for x in esc)}  flat {sorted(round(x,1) for x in flat)}")
print(f"separation gap: max flat {max(flat):.1f}  vs  min ESC {min(esc):.1f}  -> "
      f"{'separates' if min(esc) > max(flat) else 'overlaps'}")""")

md("""## §4 — Compensation / lead-lag: embedding → attention → MLP

If attention compensates for embedding↔MLP drift, its velocity peak should sit *between* embedding and MLP
in time at the escape (the propagation order seen in p109). Per-group velocity-peak epoch in the escape
window, escapers only.""")
code("""print(f"{'variant':>24}  velocity-peak epoch in escape window  (E -> attn -> MLP?)")
for name, out in DECIDABLE.items():
    if out != "ESC": continue
    v = fam.get_variant(**parse(name)); on = grok_onset(v); mid, vel = velocities(v)
    w = (mid >= on + 18000)
    peaks = {g: int(mid[w][np.argmax(vel[g][w])]) for g in ["W_E", "W_Q", "W_K", "W_in", "W_out"]}
    order = " < ".join(f"{g}:{peaks[g]}" for g in sorted(peaks, key=peaks.get))
    print(f"{name:>24}  {order}")""")

md("""## §5 — Read & next

- §1 is the load-bearing panel: in the matched count-4 pair, attention pulling away from the embedding+MLP
  pack should appear in the escaper and not the absorber. If so, 'divergence from the pack' is the
  count-independent signature.
- §3's separator, **if** it holds, is a candidate standing **regime-boundary detector** — the analyzer the
  pipeline needs (since §3 of the summary notebook showed it can't be a static scalar). Validate on the
  censored variants as they extend, and on the in-flight runs, before treating it as a gate.
- This notebook + `escape_regime_summary.ipynb` are the two that feed the pipeline push-down inventory.""")

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
out = "apps/research/notebooks/attention_divergence.ipynb"
nbf.write(nb, out); print("wrote", out, "with", len(cells), "cells")
