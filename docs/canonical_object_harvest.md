# Canonical Object Harvest — Mining the Literature for STABLE Objects

**Status:** Build-queue companion to [data_model_master.md](data_model_master.md) (updated
2026-06-29). Build queue items 1–4 (all four Layer 4 circuits) are **BUILT** (REQ_152/154/156);
item 5 (irrep basis) and the `dominant_frequency` column (REQ_157) remain. Validate against the
master doc's Part V scan before building any row.

## Method — the algebra-vs-behavior test

We harvest only objects whose **definition is a theorem about the architecture** — a closed-form
function of given primitives (weights, activations, the group the family is built on). Those are
`STABLE` regardless of which paper names them: the math is the authority, not the citation.

We **set aside** objects whose definition is an **empirical-behavioral claim** — a pattern the
model was *observed* to implement. Those are `EVOLVING`: the citation is the authority, and
citation-authority is exactly the kind that reifies a frame we might later retract.

> The discriminator in one line: **harvest the algebra, flag the behavior.** `W_O W_V` is an
> identity; an "induction head" is an observation. Both appear in canonical papers; only the first
> is safe to mint.

Arch-gating (needs ≥2 layers / longer context) is recorded but is **not** a reason to set aside —
an arch-gated STABLE object is a real future object, just uninstantiated in a 1-layer model.

**Verdict legend:** `STABLE` = closed-form / architecture-intrinsic · `EVOLVING` = behavioral,
set aside for now · gate `arch` / `family` / none · landing = which master-doc layer.

---

## Source 1 — Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*

The richest vein: nearly every primitive is defined as a matrix identity.

### Harvest (STABLE — closed form)

| Object | Closed form | Gate | Lands | vs current schema |
| --- | --- | --- | --- | --- |
| Residual stream | the additive channel; `ActivationSite[resid_post]` | none | L2 | ✅ have it (naming win already noted) |
| Attention head (additive operator) | per-head decomposition of attn | none | L2 | ✅ have it (`AttentionHead`, near-empty) |
| **QK circuit** (residual-space) | `W_Q^T W_K` per head ([d_model,d_model], rank ≤ d_head) | none | L4 | ◑ `QKCircuit` PLANNED |
| **OV circuit** (residual-space) | `W_O W_V` per head | none | L4 | ◑ `OVCircuit` PLANNED |
| **Full QK circuit** | `W_E^T W_Q^T W_K W_E` ([vocab,vocab] bilinear: token i's pull toward token j) | none | L4 | 🆕 **new operand** |
| **Full OV circuit** | `W_U W_O W_V W_E` ([vocab,vocab]: source token → output-logit contribution) | none | L4 | 🆕 **new operand — family-relevant** |
| **Direct path** | `W_U W_E` ([vocab,vocab] 0-layer bigram term in the logit decomposition) | none | L4 | 🆕 **new operand** |
| Positional QK/OV variants | `W_pos^T W_Q^T W_K W_pos`, etc. | none | L4 | 🆕 same recipe, W_pos operand |

### Decompositions on those operands (measurements, **not** objects → L8 / attribute columns)

| Measurement | On operand | Lands | Status |
| --- | --- | --- | --- |
| OV eigen-copying score | `W_U W_O W_V W_E` eigenvalues; Σ(positive)/Σ\|λ\| = copying tendency | column on FullOVCircuit | ✅ BUILT (`circuit_spectra`) |
| QK eigenvalues | full QK circuit (square) | tensor ref + scalar tail | ✅ BUILT (`circuit_spectra`) |
| Effective rank / SVD spectrum | any circuit operand | column (`circuit_spectra`) | ✅ BUILT |
| Dominant frequency / power | any circuit operand | column | ⏳ pending REQ_157 — *not* the clean `weight_basis_projection` repoint first assumed (Fourier site ≠ circuit site) |

### Arch-gated STABLE (record, don't build — 1-layer models don't instantiate)

| Object | Closed form | Gate |
| --- | --- | --- |
| Q-composition / K-composition / V-composition | earlier head's `W_O W_V` writes a subspace a later head's `W_Q`/`W_K`/`W_V` reads | `arch` (≥2 layers) |
| Virtual attention heads | products of OV circuits across layers `W_O^{(2)}W_V^{(2)}W_O^{(1)}W_V^{(1)}` | `arch` |

### Set aside (behavioral)
Skip-trigrams; "copying head" / "primarily-positional head" *as labels* (note: the copying *score*
above is algebraic and IS harvested — the boundary runs between the eigenvalue measurement and the
"this is a copying head" claim).

---

## Source 2 — Olsson et al. 2022, *In-context Learning and Induction Heads*

**Result: almost nothing to harvest.** This paper's contributions are behavioral.

| Construct | Verdict | Why |
| --- | --- | --- |
| Induction head | EVOLVING — set aside | functional definition ("attend to the token after the previous occurrence of the current token"); `arch`-gated |
| Previous-token head | EVOLVING — set aside | behavioral |
| Induction *mechanism* | already covered | structurally it's a **K-composition instance** (prev-token head K-composing into a second head) — the *algebra* is Source 1's composition object; only the *induction label* is behavioral |

Clean finding: Source 2 is a behavior catalogue over Source 1's algebra. Nothing new under the
algebra test. Revisit only when models go multi-layer **and** we choose to adopt behavioral objects.

---

## Source 3 — Nanda et al. 2023, *Progress Measures for Grokking* (the family-relevant paper)

The STABLE primitive here is **representation-theoretic**, and it dignifies the turn-1 intuition
("we're dipping into representation theory"):

| Object | Closed form | Gate | Lands | Verdict |
| --- | --- | --- | --- | --- |
| **Irrep basis of the group** | for Z/p, the DFT basis `{1, cos(2πk·/p), sin(2πk·/p)}` = the irreducible representations of the cyclic group | `family` | family instrument (underlies L5) | **STABLE** — a theorem of group rep theory, *provided by the family* |
| Fourier projection of weights/activations | coordinate change into the irrep basis | `family` | instrument (have it) | STABLE instrument |
| **FrequencyMode** (the model's *selected* modes) | which k the trained weights actually use | `family` | L5 | **EVOLVING** — an *empirical selection within* the STABLE basis |
| Trig-identity algorithm | `cos(w(a+b)) = cos wa cos wb − sin wa sin wb` as the learned mechanism | `family` | — | EVOLVING — set aside (the *algorithm* is a behavioral claim; the *basis* it's written in is STABLE) |
| Restricted / excluded loss | progress metrics defined via Fourier components | `family` | metric/instrument | STABLE instrument, not an object |

**The sharp distinction this source forces:** the **irrep basis is STABLE and family-provided**;
**which frequencies the model chooses is empirical (EVOLVING)**. The master doc marks
`FrequencyMode` EVOLVING — correct — but it never names the STABLE thing *underneath* it. The basis
is the family's canonical instrument; the selected modes are the finding. That's the representation
theory the data model actually touches: the family supplies the irreps; the research measures which
the model populates.

---

## Source 4 — Elhage et al. 2022, *Toy Models of Superposition*

**Result: almost entirely EVOLVING — set aside.**

| Construct | Verdict | Why |
| --- | --- | --- |
| Feature | EVOLVING — set aside | the field's central *contested* object; method-dependent |
| Superposition | EVOLVING — set aside | a phenomenon/regime, not a closed-form object |
| Feature direction | EVOLVING — set aside | empirical |
| Gram / interference matrix `W^T W` | STABLE *measurement* | closed form, but a decomposition on a weight matrix (→ CHEAP attribute), **not** an object; only becomes interesting once "feature" is the index, which is the EVOLVING part |

The superposition program is the cleanest illustration of the algebra-vs-behavior cut: its math
(`W^T W`) is a measurement you can take freely; its *objects* (features) are the part to defer.

---

## Source 5 — SAE / Dictionary-Learning line (Bricken et al. 2023; Cunningham et al. 2023)

**Result: set aside in full.** SAE features are defined by a *learned dictionary* — maximally
method-dependent (different SAE, different features). This is the EVOLVING extreme: the object is
constituted by the instrument that finds it. No closed-form harvest. Reconsider only if/when the
field reaches consensus on a canonical decomposition (it has not).

---

## Synthesis — the build queue (1-layer modular addition, buildable now)

Ordered by value × cheapness. All are STABLE, none arch-gated, all are mostly `CHEAP` instrument
repoints onto a composed operand.

> **Status (2026-06-29).** Items 1–4 are **BUILT** — all four circuit operands are materialized by
> the universal `circuit_spectra` analyzer (REQ_152 → REQ_154) and conformed into one site-keyed
> Layer 4 object table (REQ_156). The spectral columns (`copying_score`, `effective_rank`,
> `operator_norm`) and the matrix/eigenvalue tensor refs are populated on the baselines. The one
> column listed below that did **not** land is per-circuit `dominant_frequency` — it needs a
> first-class per-head home (not the Fourier-site `weight_basis_projection`) and is now its own
> queue item, **REQ_157**. Item 5 (irrep basis) is the remaining buildable-now harvest.

1. ✅ **BUILT (REQ_152)** — **Full OV circuit** `W_U W_O W_V W_E` — per `(variant, epoch, head)`,
   [vocab,vocab]. Directly exposes the additive structure of modulo addition (source token →
   output-logit map). L4 operand; matrix as tensor ref; columns = effective rank, operator norm,
   **copying score**. (`dominant_frequency` → REQ_157.) *Highest value: known object,
   family-relevant — built first.*
2. ✅ **BUILT (REQ_154)** — **Full QK circuit** `W_E^T W_Q^T W_K W_E` — per `(variant, epoch, head)`,
   [vocab,vocab]. Which token-pairs the head wants to bind. Same instrument repoint (a sibling
   `circuit_spectra` site). copying_score is degenerate here (OV-meaningful only).
3. ✅ **BUILT (REQ_154)** — **Direct path** `W_U W_E` — per `(variant, epoch)`, [vocab,vocab]. The
   0-layer logit term; baseline against which head contributions are read. Head-less site.
4. ✅ **BUILT (REQ_154)** — **Residual-space QK / OV circuits** (`W_Q^T W_K`, `W_O W_V`) — the master
   doc's `QKCircuit`/`OVCircuit` rows (sites `qk` / `ov`), now `populated · ACTIVE`. Built as
   `circuit_spectra` sites rather than by repointing `weight_spectra`.
5. **Irrep basis (family object)** — *remaining* — name the family's DFT/irrep basis as the STABLE
   instrument underlying `FrequencyMode`. Mostly a *naming + provenance* move (the transform
   exists); it makes explicit that the basis is given and the selected modes are the finding.

### Deferred (record only)
- Composition terms (Q/K/V), virtual attention heads — STABLE, `arch`-gated (≥2 layers).
- All of Sources 2, 4, 5 and the trig-algorithm / feature / induction-label objects — EVOLVING,
  set aside per the algebra-vs-behavior decision.

### One ontology note carried from the master-doc conversation
Every harvested object above is a **derived operand** (a matrix you then point instruments at), not
a **derived measurement** (SVD/Fourier/eig output). That is exactly why they earn object rows in
L4: deriving an *operand* can yield an object; deriving a *measurement* yields a decomposition
(L8 / tensor ref), never an object. The eigen-copying score is the test case — it sits on the
object as an attribute, it is not itself one.
