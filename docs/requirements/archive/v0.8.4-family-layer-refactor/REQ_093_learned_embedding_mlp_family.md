# REQ_093: Learned-Embedding MLP Family

**Status:** Draft
**Priority:** Medium
**Branch:** TBD
**Attribution:** Engineering Claude

---

## Problem Statement

The 2-layer one-hot MLP (REQ_088) establishes a clean null result: no geometry, no
topological reorganization, no frequency competition. The model solves modular addition
via statistical diversity (lottery ticket + majority voting) without building any
load-bearing internal structure.

The 1-layer transformer builds rich geometry — rings, saddles, frequency group
separation — under pressure from shared learned representations and compositional
bottlenecks. The critical question is: *which* architectural ingredient is responsible?

Two candidates:
1. **Learned embeddings** — encoding a structured domain into fixed-width space creates
   geometric pressure even without attention.
2. **The attention mechanism** — the router/gate that selects which part of the
   representation to read for a given input creates selection pressure that makes
   geometry load-bearing rather than optional.

These are currently confounded in the transformer. The learned-embedding MLP separates
them: it adds learned embeddings without adding attention. If geometry appears, the
embedding is sufficient. If it doesn't, the router is necessary.

This is rung 2 of a three-rung ladder:

| Architecture | Learned Embedding | Attention | Geometry? |
|---|---|---|---|
| One-hot MLP | No | No | No (confirmed, REQ_088) |
| **Learned-embedding MLP** | **Yes** | **No** | **?** |
| 1-layer Transformer | Yes | Yes | Yes (confirmed) |

A secondary question: do *deeper* MLP-only networks (3+ layers) develop geometry via
layer-to-layer coordination, even without a learned embedding? Layer outputs must
serve as inputs to the next layer, creating a representational bottleneck analogous
(weakly) to the transformer's residual stream.

---

## Architecture

**Learned-Embedding MLP:**
- Embedding layer: `Embedding(p, d_embed)` for each input (a, b) — two separate
  embeddings summed or concatenated, producing a dense input representation
- Hidden layer: `Linear(d_embed or 2*d_embed, d_hidden)` → ReLU
- Output layer: `Linear(d_hidden, p)` → logits
- No positional embedding, no attention, no residual stream
- Cross-entropy loss, AdamW optimizer

**Design decision needed:** sum vs. concatenate the two embeddings.
- Sum: forces the embedding to encode a + b in a single d_embed space — closest analog
  to how the transformer handles the two inputs additively
- Concatenate: gives the hidden layer more information to work with, but may allow the
  network to avoid learning a compositional embedding

Recommendation: sum first (it's the cleaner test of the hypothesis); concatenate as a
variant if results are ambiguous.

**Deeper MLP variant (stretch):**
- One-hot input, 3+ hidden layers (e.g., d_hidden → d_hidden → d_hidden → p)
- Tests whether layer-to-layer coordination alone creates geometric pressure
- Lower priority than the learned-embedding variant

---

## Conditions of Satisfaction

### Architecture
- [ ] `LearnedEmbeddingMLP` model class implemented (sum variant)
- [ ] `LearnedEmbeddingMLPFamily` implementing `ModelFamily` protocol
- [ ] Same checkpoint format and directory structure as existing families
- [ ] `family.json` registered under `model_families/modulo_addition_learned_emb_mlp/`

### Grokking
- [ ] Model reliably grokks on at least one (p, seed, data_seed) combination
- [ ] Grokking hyperparameters documented and justified (see Notes)
- [ ] At least one trained variant with artifacts available for analysis

### Analysis
- [ ] All applicable analyzers from the 2L MLP family run cleanly
- [ ] Geometry timeseries (`repr_geometry`) available for comparison with 1L transformer
  and 2L one-hot MLP
- [ ] Neuron frequency distribution and trajectory available
- [ ] Parameter trajectory PCA available

### Comparison
- [ ] Side-by-side qualitative comparison documented (geometry.timeseries, neuron
  specialization, parameter PCA) against 2L one-hot MLP and 1L transformer
- [ ] Finding logged: does geometry appear, and if so, which geometric signatures?

---

## Constraints

- Architecture must remain comparable to the 2L one-hot MLP — same hidden width (512),
  same prime (p=113), same optimizer family — so that differences are attributable to
  the embedding, not capacity or training differences
- Do not add attention or residual stream — that would collapse this into a transformer
- Grokking is a precondition for geometry analysis; if the model does not grokk,
  geometry results are uninterpretable

---

## Notes

**Grokking challenge.** The one-hot MLP grokked readily with He et al. settings
(lr=1e-4, wd=2.0, frac=0.75). The learned-embedding MLP introduces a new optimization
surface — the embedding parameters may behave differently under weight decay. The
approach: start with the same He et al. settings, then hypothesis-test deviations
based on observed training curves (not random search). Candidate hypotheses:
- Embedding parameters may need different weight decay than hidden layer parameters
  (parameter-group-specific weight decay)
- d_embed should create real compression pressure without being geometrically impossible

**d_embed sizing from the viability certificate.** The canon transformer (p=113,
frequencies {9,33,38,55}) has W_E participation ratio 23.45 with a theoretical
minimum of 8 dimensions (2|F| for 4 frequencies). The transformer operates with
a compression margin of 15.4 — it has significant slack above the geometric floor.

For the learned-embedding MLP, the goal is to reduce that slack deliberately:
- d_embed=8: at the theoretical floor — may be too fragile to learn reliably
- **d_embed=16**: below the transformer's observed PR (23.45), above the floor (8)
  — creates real compression pressure, the interesting regime. Start here.
- d_embed=32: above the transformer's observed PR — more slack than the transformer
  actually uses, less interesting as a first experiment

If geometry forms at d_embed=16 (under tighter compression than the transformer
faces), that's a strong result. If it forms at 32 but not 16, the margin matters
and that is itself a finding.

**Deeper MLP stretch goal.** This is a separate architectural question and should be
treated as a variant experiment, not a primary deliverable. If the learned-embedding
MLP result is clear (geometry appears or doesn't), the deeper MLP question may become
less urgent.

**Related work.** Anthropic has studied attention-only transformers. No known published
study of learned-embedding MLP-only networks on modular arithmetic tasks as of 2026-04.
This may be a genuinely novel comparison.
