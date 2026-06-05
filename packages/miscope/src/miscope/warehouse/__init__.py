"""``miscope.warehouse`` — the internal columnar warehouse (REQ_110A).

Turns analyzers' ``.npz`` artifacts into long-format Parquet driven by the
REQ_107 output declarations: ``columnar`` fields become queryable rows (semantic
tables where REQ_110 designed one, a generic coord-keyed table otherwise);
``tensor`` fields are left as blobs for 110-B. The columnar half of the shared
catalog relation is co-emitted with each table.

Entry points::

    from miscope.warehouse import materialize_variant_columnar, read_table
    materialize_variant_columnar(variant)        # write the warehouse
    read_table(variant, "pca_results").to_wide(...)  # in-memory long-format surface
"""

from __future__ import annotations

from miscope.warehouse.reader import WarehouseTable, read_table
from miscope.warehouse.writer import MaterializeReport, materialize_variant_columnar

__all__ = [
    "materialize_variant_columnar",
    "MaterializeReport",
    "read_table",
    "WarehouseTable",
]
