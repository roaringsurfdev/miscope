# REQ_021c: Modulo Addition 1-Layer Family Implementation

**Status:** Active
**Priority:** High
**Parent:** [REQ_021](REQ_021_model_families.md)
**Dependencies:** REQ_021a (Core Abstractions), REQ_021b (Analysis Library)
**Last Updated:** 2026-02-03

## Problem Statement

With core abstractions and analysis library in place, we need a concrete implementation to validate the design. The Modulo Addition 1-Layer family serves as:
1. The first complete family implementation
2. Validation that the abstractions work in practice
3. Reference implementation for future families

## Scope

This sub-requirement covers:
1. Creating `model_families/modulo_addition_1layer/family.json`
2. Implementing `ModuloAddition1LayerFamily` class
3. Wiring existing analyzers to the family
4. Validating variant discovery and state management
5. Ensuring existing trained models are discoverable

## Proposed Design

### family.json

```json
{
  "name": "modulo_addition_1layer",
  "display_name": "Modulo Addition (1 Layer)",
  "description": "Single-layer transformer trained on modular arithmetic",

  "architecture": {
    "n_layers": 1,
    "n_heads": 4,
    "d_model": 128,
    "d_mlp": 512,
    "act_fn": "relu"
  },

  "domain_parameters": {
    "prime": {
      "type": "int",
      "description": "Modulus for the addition task",
      "default": 113
    },
    "seed": {
      "type": "int",
      "description": "Random seed for model initialization",
      "default": 999
    }
  },

  "analyzers": [
    "dominant_frequencies",
    "neuron_activations",
    "neuron_frequency_clusters"
  ],

  "visualizations": [
    "dominant_frequencies_bar",
    "neuron_activation_heatmap",
    "neuron_activation_grid",
    "frequency_clusters_heatmap"
  ],

  "analysis_dataset": {
    "type": "modulo_addition_grid",
    "description": "Full (a, b) input grid for modular arithmetic"
  },

  "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}"
}
```

### Family Implementation

```python
from pathlib import Path
from analysis.core import ModelFamily, ParameterSpec
import torch
from transformer_lens import HookedTransformer

class ModuloAddition1LayerFamily:
    """Implementation of ModelFamily for 1-layer modular addition transformer."""

    def __init__(self, config_path: Path):
        self._config = self._load_config(config_path)

    @property
    def name(self) -> str:
        return self._config["name"]

    @property
    def display_name(self) -> str:
        return self._config["display_name"]

    @property
    def description(self) -> str:
        return self._config["description"]

    @property
    def architecture(self) -> dict:
        return self._config["architecture"]

    @property
    def domain_parameters(self) -> dict[str, ParameterSpec]:
        return self._config["domain_parameters"]

    @property
    def analyzers(self) -> list[str]:
        return self._config["analyzers"]

    @property
    def visualizations(self) -> list[str]:
        return self._config["visualizations"]

    @property
    def variant_pattern(self) -> str:
        return self._config["variant_pattern"]

    def create_model(self, params: dict) -> HookedTransformer:
        """Create a HookedTransformer for modular addition."""
        p = params["prime"]

        cfg = HookedTransformerConfig(
            n_layers=self.architecture["n_layers"],
            n_heads=self.architecture["n_heads"],
            d_model=self.architecture["d_model"],
            d_mlp=self.architecture["d_mlp"],
            act_fn=self.architecture["act_fn"],
            d_vocab=p,
            d_vocab_out=p,
            n_ctx=3,  # a, b, =
            # ... other config
        )

        return HookedTransformer(cfg)

    def generate_analysis_dataset(self, params: dict) -> torch.Tensor:
        """Generate full (a, b) grid for analysis."""
        p = params["prime"]
        # All pairs (a, b) where 0 <= a, b < p
        a = torch.arange(p)
        b = torch.arange(p)
        grid_a, grid_b = torch.meshgrid(a, b, indexing='ij')
        equals_token = torch.full_like(grid_a, p)  # or designated token

        dataset = torch.stack([
            grid_a.flatten(),
            grid_b.flatten(),
            equals_token.flatten()
        ], dim=1)

        return dataset

    def get_variant_directory_name(self, params: dict) -> str:
        """Generate variant directory name from pattern."""
        return self.variant_pattern.format(**params)
```

### Directory Structure After Implementation

```
model_families/
  modulo_addition_1layer/
    family.json
    reports/
      default.json

results/
  modulo_addition_1layer/
    modulo_addition_1layer_p113_seed42/
      checkpoints/
        epoch_100.safetensors
        epoch_500.safetensors
        ...
      artifacts/
        dominant_frequencies/
          epoch_100.npz
          ...
      metadata.json
      config.json
```

## Conditions of Satisfaction

- [ ] `model_families/modulo_addition_1layer/family.json` created and valid
- [ ] `ModuloAddition1LayerFamily` class implements `ModelFamily` protocol
- [ ] `create_model()` produces equivalent model to current `ModuloAdditionSpecification`
- [ ] `generate_analysis_dataset()` produces correct (a, b) grid
- [ ] `FamilyRegistry` discovers the family on startup
- [ ] Existing results (if any) are discoverable as variants
- [ ] All three analyzers can run against a loaded variant
- [ ] New variant can be created by specifying `prime` and `seed` parameters
- [ ] Integration test: train small variant, analyze, verify artifacts created

## Constraints

**Must have:**
- Backwards compatibility: existing trained models remain accessible
- Model architecture matches Neel Nanda's grokking experiment spec
- Analysis dataset generation is deterministic

**Must avoid:**
- Breaking existing analysis scripts during transition
- Changing checkpoint format (safetensors)

**Flexible:**
- Migration strategy for existing results directory structure
- Whether to auto-migrate or support both old and new paths temporarily

## Validation Checklist

To confirm the design works:

1. **Family Discovery:**
   ```python
   registry = FamilyRegistry(Path("model_families"))
   family = registry.get_family("modulo_addition_1layer")
   assert family.name == "modulo_addition_1layer"
   ```

2. **Variant Creation:**
   ```python
   variant = Variant(family, {"prime": 113, "seed": 42})
   assert variant.name == "modulo_addition_1layer_p113_seed42"
   ```

3. **Model Creation:**
   ```python
   model = family.create_model({"prime": 113, "seed": 42})
   assert model.cfg.d_vocab == 113
   ```

4. **Analysis Execution:**
   ```python
   analyzers = AnalyzerRegistry.get_for_family(family)
   for analyzer in analyzers:
       result = analyzer.analyze(model, dataset, params)
       assert result["data"] is not None
   ```

## Notes

This is the integration sub-requirement that validates REQ_021a and REQ_021b work together. If issues are found during implementation, they should be resolved by updating the relevant foundational sub-requirement, not by adding workarounds here.
