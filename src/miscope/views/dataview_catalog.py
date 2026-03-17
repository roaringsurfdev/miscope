"""REQ_054: Data View Catalog — Core types and registry.

DataViewDefinition pairs data loading and a static schema for a named dataview.
DataViewCatalog is the registry of all available dataviews.
BoundDataView is the Data Presenter — epoch fixed, exposes data() and schema.
DataView is the lightweight container returned by BoundDataView.data().
DataViewSchema describes all fields in a DataView without requiring IO.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from miscope.views.catalog import AnalyzerRequirement, _check_requirement

if TYPE_CHECKING:
    from miscope.families.variant import Variant


@dataclass
class DataViewField:
    """Describes a single field in a DataView.

    Attributes:
        name: Field name (matches attribute on DataView).
        field_type: "dataframe" for tabular/scalar data, "ndarray" for tensor data.
        description: Short human-readable description of what this field contains.
        shape_or_columns: For ndarrays, a description of shape (e.g., "(n_epochs, n_freqs)").
            For DataFrames, a list of column names.
    """

    name: str
    field_type: str  # "dataframe" or "ndarray"
    description: str
    shape_or_columns: str | list[str] = field(default="")


@dataclass
class DataViewSchema:
    """Describes all fields in a DataView — available before any data is loaded.

    Attributes:
        fields: List of DataViewField descriptions.
    """

    fields: list[DataViewField]

    def field_names(self) -> list[str]:
        """Return sorted list of all field names."""
        return sorted(f.name for f in self.fields)

    def get_field(self, name: str) -> DataViewField:
        """Look up a field descriptor by name."""
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"Unknown field '{name}'. Available fields: {self.field_names()}")


class DataView:
    """Lightweight container for named data fields.

    Fields are either pd.DataFrame (tabular/scalar data) or np.ndarray (tensor data).
    Access fields by attribute: ``view.losses``, ``view.coefficients``.
    """

    def __init__(self, schema: DataViewSchema, **fields: pd.DataFrame | np.ndarray) -> None:
        self._schema = schema
        for name, value in fields.items():
            setattr(self, name, value)

    @property
    def schema(self) -> DataViewSchema:
        """Schema describing all fields in this DataView."""
        return self._schema

    def __repr__(self) -> str:
        field_names = self._schema.field_names()
        return f"DataView(fields={field_names})"


@dataclass
class DataViewDefinition:
    """Pairs data loading and a static schema for a named dataview.

    Attributes:
        name: Unique dataview identifier (e.g., "training.metadata.loss_curves").
        load_data: Callable(variant, epoch) -> DataView. Loads all data.
            Epoch may be None for cross-epoch/metadata views.
        schema: Static schema describing all fields. Available before any IO.
        epoch_source_analyzer: Analyzer name used to resolve a None epoch to
            the first available artifact epoch. None for cross-epoch and
            metadata-based views where no resolution is needed.
        required_analyzers: Artifact requirements that must be satisfied for
            this dataview to be available. Empty list means always available.
    """

    name: str
    load_data: Callable[..., DataView]
    schema: DataViewSchema
    epoch_source_analyzer: str | None = field(default=None)
    required_analyzers: list[AnalyzerRequirement] = field(default_factory=list)

    def is_available_for(self, variant: Variant) -> bool:
        """Return True if all required artifacts exist for this variant."""
        for req in self.required_analyzers:
            if not _check_requirement(req, variant):
                return False
        return True


class DataViewCatalog:
    """Registry of named dataviews available in miscope."""

    def __init__(self) -> None:
        self._dataviews: dict[str, DataViewDefinition] = {}

    def register(self, dataview_def: DataViewDefinition) -> None:
        """Register a dataview definition."""
        self._dataviews[dataview_def.name] = dataview_def

    def get(self, name: str) -> DataViewDefinition:
        """Look up a dataview by name.

        Raises:
            KeyError: If name not found, with message listing available dataviews.
        """
        if name not in self._dataviews:
            available = sorted(self._dataviews.keys())
            raise KeyError(f"Unknown dataview '{name}'. Available dataviews: {available}")
        return self._dataviews[name]

    def names(self) -> list[str]:
        """Return sorted list of all registered dataview names."""
        return sorted(self._dataviews.keys())

    def available_names_for(self, variant: Variant) -> list[str]:
        """Return sorted list of dataview names whose requirements are met for this variant."""
        return sorted(
            name for name, dv_def in self._dataviews.items() if dv_def.is_available_for(variant)
        )


class BoundDataView:
    """A dataview definition with variant and epoch already bound.

    Returned by EpochContext.dataview(). Callers do not pass epoch — it was
    fixed at EpochContext creation time.
    """

    def __init__(
        self,
        dataview_def: DataViewDefinition,
        variant: Variant,
        epoch: int | None,
    ) -> None:
        self._dataview_def = dataview_def
        self._variant = variant
        self._epoch = epoch

    @property
    def schema(self) -> DataViewSchema:
        """Schema describing all fields — no IO required."""
        return self._dataview_def.schema

    def data(self) -> DataView:
        """Load and return the DataView container."""
        return self._dataview_def.load_data(self._variant, self._epoch)

    def __repr__(self) -> str:
        return (
            f"BoundDataView(name={self._dataview_def.name!r}, "
            f"variant={self._variant.name!r}, epoch={self._epoch!r})"
        )


def _resolve_dataview_epoch(
    epoch: int | None,
    dataview_def: DataViewDefinition,
    variant: Variant,
) -> int | None:
    """Resolve None epoch to first available artifact epoch for per-epoch dataviews."""
    if epoch is not None or dataview_def.epoch_source_analyzer is None:
        return epoch
    epochs = variant.artifacts.get_epochs(dataview_def.epoch_source_analyzer)
    return epochs[0] if epochs else None


# Module-level default catalog. Populated by miscope.views.dataview_universal on import.
_dataview_catalog = DataViewCatalog()
