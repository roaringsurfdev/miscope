"""``miscope.warehouse`` — the internal warehouse over analyzer artifacts (REQ_110).

Two planes, one shared catalog relation, driven by the REQ_107 output
declarations:

- **Columnar** (110-A): ``columnar`` fields become long-format Parquet rows
  (semantic tables where REQ_110 designed one, a generic coord-keyed table
  otherwise). The columnar half of the catalog is co-emitted with each table.
- **Tensor** (110-B): ``tensor`` fields stay ``.npz`` blobs (format unchanged)
  but are indexed as coordinate-keyed descriptors. You select tensors by SQL
  over descriptors and a :class:`TensorResolver` materializes only what you
  selected, verifying shape/dtype against the bytes.

Entry points::

    from miscope.warehouse import materialize_variant_columnar, read_table
    materialize_variant_columnar(variant)            # write the columnar warehouse
    read_table(variant, "pca_results").to_wide(...)  # in-memory long-format surface

    from miscope.warehouse import materialize_variant_tensors, read_tensor_catalog
    materialize_variant_tensors(variant)             # index the tensor blobs
    read_tensor_catalog(variant)                     # the descriptor relation (no payload)
"""

from __future__ import annotations

from miscope.warehouse.derived import DerivedMaterializeReport, materialize_variant_derived
from miscope.warehouse.reader import WarehouseTable, read_table
from miscope.warehouse.run_sets import (
    OrphanRecipe,
    RunSetRecord,
    live_recipe_signatures,
    orphaned_recipe_dirs,
    read_run_sets,
    record_run_set,
    run_set_id,
)
from miscope.warehouse.tensor_catalog import (
    DtypeDrift,
    TensorCatalogReport,
    TensorCatalogRow,
    TensorRef,
    TensorResolver,
    materialize_variant_tensors,
    read_tensor_catalog,
)
from miscope.warehouse.writer import MaterializeReport, materialize_variant_columnar

__all__ = [
    "materialize_variant_columnar",
    "MaterializeReport",
    "materialize_variant_derived",
    "DerivedMaterializeReport",
    "read_table",
    "WarehouseTable",
    "materialize_variant_tensors",
    "read_tensor_catalog",
    "TensorRef",
    "TensorCatalogRow",
    "TensorResolver",
    "TensorCatalogReport",
    "DtypeDrift",
    "read_run_sets",
    "record_run_set",
    "run_set_id",
    "RunSetRecord",
    "live_recipe_signatures",
    "orphaned_recipe_dirs",
    "OrphanRecipe",
]
