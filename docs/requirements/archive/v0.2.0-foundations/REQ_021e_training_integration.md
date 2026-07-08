# REQ_021e: Training Integration

**Status:** Active
**Priority:** High
**Parent:** [REQ_021](REQ_021_model_families.md)
**Dependencies:** REQ_021a (Core Abstractions), REQ_021c (Modulo Addition Family)
**Last Updated:** 2026-02-03

## Problem Statement

The training flow currently bypasses the Model Family abstraction:
- Training uses `ModuloAdditionSpecification` directly
- Directory naming is hardcoded, not derived from `variant_pattern`
- `family.create_model()` is not used for model instantiation
- Trained models require manual discovery to appear in the dashboard

For the family abstraction to be complete, training must flow through the same abstractions as analysis and visualization.

## Scope

This sub-requirement covers:
1. Training tab uses family selection
2. Training creates variants through the family
3. Model instantiation via `family.create_model()`
4. Proper variant directory structure
5. Seamless integration with existing analysis pipeline

## Current State

```
Training Tab                    Analysis Tab
     │                               │
     ▼                               ▼
ModuloAdditionSpecification    FamilyRegistry
     │                               │
     ▼                               ▼
  results/                      Variant discovery
     │                               │
     └───────── Gap ─────────────────┘
```

## Proposed Design

### Target Architecture

```
Training Tab                    Analysis Tab
     │                               │
     ▼                               ▼
FamilyRegistry ◄───────────────► FamilyRegistry
     │                               │
     ▼                               ▼
family.create_model()           Variant discovery
     │                               │
     ▼                               ▼
Variant.train()                 Variant selection
     │                               │
     └──────────► results/ ◄─────────┘
```

### Training Flow

1. **Family Selection** - User selects family from dropdown (same registry as Analysis tab)
2. **Parameter Input** - UI shows family's domain parameters with defaults
3. **Variant Creation** - `registry.create_variant(family, params)`
4. **Model Instantiation** - `family.create_model(params)` creates the model
5. **Training Execution** - Training loop runs, saving to `variant.checkpoints_dir`
6. **Metadata Persistence** - Losses and config saved to variant directory
7. **Automatic Discovery** - Variant appears in Analysis tab on refresh

### Variant Training Method

Add a `train()` method to Variant or create a family-aware Trainer:

```python
# Option A: Method on Variant
class Variant:
    def train(
        self,
        num_epochs: int,
        checkpoint_epochs: list[int] | None = None,
        training_fraction: float = 0.3,
        data_seed: int = 598,
        device: str | None = None,  # Auto-detect if None
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TrainingResult:
        """Train this variant's model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.family.create_model(self.params, device=device)
        dataset = self.family.generate_training_dataset(self.params, device=device)
        # ... training loop

# Option B: Family-aware Trainer
class FamilyTrainer:
    def __init__(self, variant: Variant, device: str | None = None):
        self.variant = variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, num_epochs: int, ...) -> TrainingResult:
        model = self.variant.family.create_model(self.variant.params, device=self.device)
        # ... training loop
```

### Training Parameters

Domain parameters come from family, training hyperparameters are separate:

| Source | Parameters |
|--------|------------|
| Family (domain) | `prime`, `seed` (from `domain_parameters`) |
| Training (common) | `num_epochs`, `checkpoint_epochs`, `training_fraction`, `data_seed`, `learning_rate`, `weight_decay` |
| Runtime | `device` (auto-detect CUDA if available, fallback to CPU) |

### Device Handling

Training must preserve the existing GPU acceleration behavior:

```python
# Auto-detect device (current behavior to preserve)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pass to model creation
model = family.create_model(params, device=device)

# Ensure dataset is on same device
dataset = family.generate_training_dataset(params, device=device)
```

The `create_model()` method already accepts a `device` parameter (implemented in REQ_021c). Training integration must use this consistently.

### Dashboard UI Updates

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING TAB                                                │
├─────────────────────────────────────────────────────────────┤
│  Family: [Modulo Addition (1 Layer) ▼]                       │
├─────────────────────────────────────────────────────────────┤
│  Domain Parameters:                Training Parameters:      │
│  ┌─────────────────────┐          ┌─────────────────────┐   │
│  │ Prime (p): [113]    │          │ Epochs: [25000]     │   │
│  │ Seed: [42]          │          │ Train Frac: [0.3]   │   │
│  └─────────────────────┘          │ Data Seed: [598]    │   │
│                                   │ Checkpoints: [...]  │   │
│                                   └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Variant: modulo_addition_1layer_p113_seed42                 │
│  Status: Ready to train                                      │
│                                                              │
│  [Start Training]                                            │
└─────────────────────────────────────────────────────────────┘
```

### Migration Strategy

1. Keep `ModuloAdditionSpecification` working (backward compatibility)
2. Add training capability to family abstraction
3. Update dashboard Training tab to use family-based training
4. Deprecate `ModuloAdditionSpecification` in documentation
5. Eventually remove in future version

### Family Protocol Extension

```python
class ModelFamily(Protocol):
    # Existing methods...

    def generate_training_dataset(
        self,
        params: dict[str, Any],
        training_fraction: float = 0.3,
        data_seed: int = 598,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test split for training.

        Returns:
            (train_data, train_labels, test_data, test_labels)
        """
        ...

    def get_training_config(self) -> dict[str, Any]:
        """Return default training hyperparameters.

        Returns:
            Dict with learning_rate, weight_decay, etc.
        """
        ...
```

## Conditions of Satisfaction

- [ ] Training tab has family selector dropdown
- [ ] Domain parameters populated from `family.domain_parameters`
- [ ] Variant name derived from `family.variant_pattern`
- [ ] Model created via `family.create_model()`
- [ ] Checkpoints saved to `variant.checkpoints_dir`
- [ ] Metadata saved to `variant.metadata_path`
- [ ] Config saved to `variant.config_path`
- [ ] Trained variant appears in Analysis tab on refresh
- [ ] GPU training preserved (auto-detect CUDA, fallback to CPU)
- [ ] Existing training functionality preserved (no regression)
- [ ] All tests pass

## Constraints

**Must have:**
- Family-based training produces identical models to current approach
- Trained variants discoverable by FamilyRegistry
- Progress reporting works during training

**Must avoid:**
- Breaking existing trained models in results/
- Forcing migration of existing ModuloAdditionSpecification usage
- Changing model architecture or training dynamics

**Flexible:**
- Exact location of training logic (Variant method vs separate Trainer)
- Whether to add generate_training_dataset to Protocol vs concrete class
- UI layout for parameter inputs

## Testing Strategy

1. **Unit Tests:**
   - Variant.train() creates correct directory structure
   - Model from family.create_model() matches expected architecture
   - Training produces valid checkpoints

2. **Integration Tests:**
   - Train via dashboard → appears in variant list
   - Train → analyze → visualize flow works end-to-end
   - Multiple variants of same family train independently

3. **Regression Tests:**
   - Existing ModuloAdditionSpecification still works
   - Previously trained models still discoverable
   - Analysis artifacts unchanged

## Notes

This requirement completes the family abstraction loop:

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Training  │ ──► │  Analysis  │ ──► │ Visualize  │
└────────────┘     └────────────┘     └────────────┘
      │                  │                  │
      ▼                  ▼                  ▼
┌─────────────────────────────────────────────────┐
│              Family Abstraction                  │
│  (ModelFamily, Variant, FamilyRegistry)         │
└─────────────────────────────────────────────────┘
```

After this requirement, adding a new model family means:
1. Create `family.json` with architecture and parameters
2. Implement `create_model()` and dataset methods
3. Register family-specific analyzers (if any)
4. Everything else "just works"
