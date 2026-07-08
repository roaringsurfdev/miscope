# REQ_021a: Core Abstractions

**Status:** Active
**Priority:** High
**Parent:** [REQ_021](REQ_021_model_families.md)
**Dependencies:** None
**Last Updated:** 2026-02-03

## Problem Statement

The workbench needs foundational abstractions to represent model families and their variants. These abstractions must support:
- Explicit family registration and discovery
- Variant lifecycle management (untrained → trained → analyzed)
- Consistent directory structure conventions
- Type-safe protocol for family implementations

## Scope

This sub-requirement covers:
1. `ModelFamily` protocol definition
2. `Variant` class with state tracking
3. Directory structure conventions and path resolution
4. `family.json` schema validation

## Proposed Design

### ModelFamily Protocol

```python
from typing import Protocol, Any
from pathlib import Path
import torch
from transformer_lens import HookedTransformer

class ParameterSpec(TypedDict):
    type: str  # "int", "float", "str"
    description: str
    default: Any

class ModelFamily(Protocol):
    """Protocol defining the contract for a model family."""

    @property
    def name(self) -> str:
        """Unique identifier, used as directory key."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for UI."""
        ...

    @property
    def description(self) -> str:
        """Brief description of the family."""
        ...

    @property
    def architecture(self) -> dict[str, Any]:
        """Architectural properties (n_layers, n_heads, etc.)."""
        ...

    @property
    def domain_parameters(self) -> dict[str, ParameterSpec]:
        """Parameters that vary across variants."""
        ...

    @property
    def analyzers(self) -> list[str]:
        """Analyzer identifiers valid for this family."""
        ...

    @property
    def visualizations(self) -> list[str]:
        """Visualization identifiers valid for this family."""
        ...

    @property
    def variant_pattern(self) -> str:
        """Pattern for variant directory names, e.g., '{name}_p{prime}_seed{seed}'."""
        ...

    def create_model(self, params: dict[str, Any]) -> HookedTransformer:
        """Instantiate a model with the given domain parameters."""
        ...

    def generate_analysis_dataset(self, params: dict[str, Any]) -> torch.Tensor:
        """Generate the analysis dataset (probe) for a variant."""
        ...

    def get_variant_directory_name(self, params: dict[str, Any]) -> str:
        """Generate variant directory name from parameters."""
        ...
```

### Variant Class

```python
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

class VariantState(Enum):
    UNTRAINED = "untrained"
    TRAINED = "trained"
    ANALYZED = "analyzed"

@dataclass
class Variant:
    """A specific trained model within a family."""

    family: ModelFamily
    params: dict[str, Any]

    @property
    def name(self) -> str:
        """Variant directory name derived from family pattern."""
        return self.family.get_variant_directory_name(self.params)

    @property
    def state(self) -> VariantState:
        """Current state based on filesystem presence."""
        ...

    @property
    def results_dir(self) -> Path:
        """Path to variant's results directory."""
        ...

    @property
    def checkpoints_dir(self) -> Path:
        """Path to checkpoints subdirectory."""
        ...

    @property
    def artifacts_dir(self) -> Path:
        """Path to analysis artifacts subdirectory."""
        ...

    def get_available_checkpoints(self) -> list[int]:
        """List checkpoint epochs available on disk."""
        ...

    def load_checkpoint(self, epoch: int) -> HookedTransformer:
        """Load model weights from a specific checkpoint."""
        ...
```

### Directory Conventions

The `ModelFamily.name` serves as the directory key at all levels:

```
model_families/{family.name}/
  family.json
  reports/

results/{family.name}/
  {variant_directory_name}/
    checkpoints/
    artifacts/
    metadata.json
    config.json
```

### FamilyRegistry

```python
class FamilyRegistry:
    """Discovers and manages registered model families."""

    def __init__(self, model_families_dir: Path):
        self._families: dict[str, ModelFamily] = {}
        self._load_families(model_families_dir)

    def get_family(self, name: str) -> ModelFamily:
        """Get a family by name."""
        ...

    def list_families(self) -> list[ModelFamily]:
        """List all registered families."""
        ...

    def get_variants(self, family: ModelFamily) -> list[Variant]:
        """Discover variants for a family from results directory."""
        ...
```

## Conditions of Satisfaction

- [ ] `ModelFamily` protocol defined with all required methods/properties
- [ ] `Variant` class implemented with state detection from filesystem
- [ ] `VariantState` enum covers untrained/trained/analyzed states
- [ ] `FamilyRegistry` can discover families from `model_families/` directory
- [ ] `FamilyRegistry` can discover variants from `results/` directory
- [ ] Path resolution uses `ModelFamily.name` consistently
- [ ] `family.json` schema documented and validated on load
- [ ] Unit tests for path resolution and state detection

## Constraints

**Must have:**
- Protocol-based design (not base class) for flexibility
- State derived from filesystem, not stored separately
- Thread-safe registry access

**Must avoid:**
- Circular dependencies with analysis or dashboard code
- Hardcoded paths (use configuration)

**Flexible:**
- Exact validation library for family.json (pydantic, jsonschema, etc.)
- Caching strategy for family/variant discovery

## Notes

This is the foundational sub-requirement. REQ_021c and REQ_021d depend on these abstractions being stable before implementation.
