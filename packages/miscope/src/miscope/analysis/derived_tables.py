"""Derived-table declarations (REQ_141) — collected by ``miscope.registry``.

Importing this module registers every derived table via
:func:`~miscope.analysis.derived_table.register_derived_table` side effects, the
same import-side-effect pattern the analyzer and DataView surfaces use. The
registry's ``build_index`` imports this module so the index sees the full set.

Declarations are pure data (SQL text over warehouse table names + an output
schema); the executor that materializes a spec lives in
:mod:`miscope.warehouse.derived`. The neuron-frequency vertical slice (the
``transient_frequency`` derived table and the peak-membership long table) is added
here as REQ_141 proceeds.
"""

from __future__ import annotations
