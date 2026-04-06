# REQ_088: 2L MLP Family — Architecture and Training

**Status:** Active
**Priority:** High
**Branch:** feature/req-088-2l-mlp-family
**Attribution:** Engineering Claude

---

## Problem Statement

With the ActivationBundle abstraction in place (REQ_087), the platform can support non-transformer
architectures. The 2L MLP is the first target: same modular addition task, same data, architecture
stripped to two fully-connected layers with no attention heads, no positional embedding, and no
residual stream.

The scientific motivation is a minimal perturbation experiment. If the phase tiling, spoke
structure, and saddle geometry observed in the transformer persist in the MLP, they are
properties of gradient descent finding compact representations — not transformer-specific
mechanisms. If they disappear and the MLP uses a simpler 2D ring, the extra structure is
localized to transformer components (attention or residual stream composition).

This requirement covers the architecture, training loop, and family registration.
Analysis views for MLP-specific geometry are out of scope here.

---

## Conditions of Satisfaction

### Architecture
- [ ] `TwoLayerMLP` PyTorch module defined: input → Linear(vocab_size, d_hidden) → ReLU →
  Linear(d_hidden, vocab_size) → output logits
- [ ] Input is a flat one-hot encoding of (a, b) concatenated: size 2p (no equals token,
  no positional embedding)
- [ ] Architecture hyperparameters (d_hidden, vocab_size derived from prime p) stored in
  `config.json` alongside checkpoints, matching the existing variant config convention
- [ ] Model is seedable for reproducible initialization

### Training
- [ ] Training loop produces checkpoints at the same schedule as the transformer family
  (configurable, defaulting to the existing adaptive schedule logic)
- [ ] Checkpoints saved as `.safetensors` files in `checkpoints/checkpoint_epoch_{NNNNN}.safetensors`
  under the variant directory — same path convention as the transformer family
- [ ] Train/test split uses the same `data_seed` mechanism as the transformer family

### Family Registration
- [ ] `TwoLayerMLPFamily` class implementing `ModelFamily` protocol registered in the family
  registry under name `"modulo_addition_2layer_mlp"`
- [ ] `prepare_analysis_context()` provides the same `fourier_basis` key as the transformer
  family — enabling all Fourier-based analyzers to run without modification
- [ ] Family `analyzers` list excludes transformer-only analyzers (`attention_freq`,
  `attention_fourier`, `attention_patterns`) and includes all architecture-agnostic ones
- [ ] `run_forward_pass(model, probe) -> ActivationBundle` implemented, returning an
  `MLPActivationBundle` that hooks into the MLP's intermediate activations

### ActivationBundle for MLP
- [ ] `MLPActivationBundle` implements the `ActivationBundle` protocol:
  - `mlp_post(layer, position)` — returns hidden layer activations (layer 0 only; layer 1
    is the output)
  - `weight(name)` — returns named weight matrices. Supported: `W_in` (first layer weights),
    `W_out` (second layer weights). Raises `KeyError` for transformer-specific names.
  - `attention_pattern(layer)` — raises `NotImplementedError`
  - `residual_stream(layer, position, location)` — raises `NotImplementedError`
  - `logits(position)` — returns output logits (position argument ignored; MLP has no sequence)

### Validation
- [ ] At least one variant (p=113, seed=999, data_seed=598) trains to grokking
- [ ] Checkpoints load correctly via the analysis pipeline
- [ ] `neuron_activations`, `neuron_freq_norm`, `parameter_snapshot`, and `neuron_group_pca`
  analyzers run successfully against MLP checkpoints

---

## Constraints

**Must:**
- Reuse the existing checkpoint save/load infrastructure (`safetensors`)
- Reuse the existing `fourier_basis` computation — the task and prime structure are the same
- The family must be usable from the same dashboard/pipeline entry points as the transformer family

**Must not:**
- Require changes to any existing analyzer to support MLP inputs — the ActivationBundle
  abstraction (REQ_087) is the only adapter layer needed
- Introduce a dependency on TransformerLens for MLP model construction or forward passes

**Flexible:**
- Whether the MLP uses one-hot input or embedding lookup (one-hot is simpler and more
  interpretable for this experiment; embedding lookup would require a separate embedding analysis)
- Exact d_hidden default value (literature uses 512 for this task; match or parameterize)
- Whether training uses Adam or AdamW (match transformer family default)

---

## Architecture Notes

**Input encoding choice:** One-hot concatenation of (a, b) — two vectors of size p, concatenated
to 2p — is the cleanest choice for interpretability. It avoids the embedding lookup layer that
the transformer uses, removing one variable from the comparison. The Fourier analysis still
applies to the hidden layer activations; the input is just a different encoding.

**Hook strategy for `MLPActivationBundle`:** Since there's no TL cache, use PyTorch forward
hooks on the ReLU output to capture hidden activations during the forward pass. The bundle
stores these tensors and exposes them via the protocol methods.

**`parameter_snapshot` compatibility:** The analyzer currently extracts W_E, W_pos, W_Q, etc.
by name. For the MLP, only W_in and W_out are available. The analyzer should gracefully handle
missing weight names (the bundle raises `KeyError`; the analyzer skips or stores NaN for those
keys). Alternatively, provide a separate MLP-specific snapshot analyzer — but prefer the
graceful degradation path to avoid forking.

**Results directory:** `results/modulo_addition_2layer_mlp/` — separate from the transformer
family's directory. The family `name` property drives this.

---

## Notes

- The primary scientific deliverable from this requirement is a trained variant of p=113/s=999/ds=598
  that can be analyzed with the full Fourier/neuron/group PCA toolkit. The geometry comparison
  (ring vs. saddle) will be done in a notebook first, not a dashboard page.
- d_hidden=512 is a common choice in the grokking literature for this task. If the transformer
  uses d_mlp=256 (check config), consider matching it to control for neuron count — or use 512
  and treat it as a separate variable.
- The one-hot input means there is no W_E to analyze for Fourier structure. This is expected —
  the comparison point is the hidden layer (W_in), not the embedding.
- Training the MLP to grokking may require different hyperparameters than the transformer.
  Weight decay is especially important for grokking in MLPs. Use literature values as starting
  point; adjust as needed and document final values in the variant config.
