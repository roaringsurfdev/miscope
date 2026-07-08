"""Type definitions for model families."""

from enum import Enum
from typing import Any, TypedDict


class ParameterSpec(TypedDict, total=False):
    """Specification for a domain parameter.

    Attributes:
        type: The parameter type ("int", "float", "str")
        description: Human-readable description
        default: Default value for this parameter
    """

    type: str
    description: str
    default: Any


class AnalysisDatasetSpec(TypedDict, total=False):
    """Specification for the analysis dataset.

    Attributes:
        type: Identifier for the dataset type
        description: Human-readable description
    """

    type: str
    description: str


class ArchitectureSpec(TypedDict, total=False):
    """Specification for model architecture.

    Attributes:
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension
        d_mlp: MLP dimension
        act_fn: Activation function name
    """

    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    act_fn: str


class VariantState(Enum):
    """State of a variant based on filesystem presence.

    States:
        UNTRAINED: No checkpoints or model files exist
        TRAINED: Checkpoints exist but no analysis artifacts
        ANALYZED: Both checkpoints and analysis artifacts exist
    """

    UNTRAINED = "untrained"
    TRAINED = "trained"
    ANALYZED = "analyzed"
