"""Generic-table discriminator hints (REQ_110A).

Semantic-table routing lives in :mod:`.mapping_semantic` (the ``CLAIMS`` that pull
fields into the designed tables). This module covers only the *generic* fallback:
the mechanical ``group_type`` tag for an analyzer's ``site``/``group`` column when
a field is not claimed by any semantic table (``operation_type`` stays null for
generic rows — it is a measurement discriminator the generic plane cannot infer).
"""

from __future__ import annotations

from collections.abc import Iterable

from miscope.warehouse.schema import GROUP_TYPE, GroupType

# The mechanical group_type for each analyzer's site/group column.
_GENERIC_GROUP_TYPE: dict[str, GroupType] = {
    "weight_spectra": GroupType.WEIGHT_MATRIX,
    "repr_geometry": GroupType.ACTIVATION_SITE,
    "centroid_fourier_alignment": GroupType.ACTIVATION_SITE,
    "activation_frequency_norm": GroupType.ACTIVATION_SITE,
    "activation_dmd": GroupType.ACTIVATION_SITE,
    "weight_basis_projection": GroupType.WEIGHT_MATRIX,
    "neuron_group_pca": GroupType.FREQUENCY_GROUP,
    "intragroup_manifold": GroupType.FREQUENCY_GROUP,
    "freq_group_weight_geometry": GroupType.FREQUENCY_GROUP,
    "parameter_trajectory": GroupType.WEIGHT_COMPONENT,
    "parameter_dmd": GroupType.FREQUENCY_GROUP,
}


def generic_discriminators(analyzer_name: str, columns: Iterable[str]) -> dict[str, object]:
    """The constant discriminators to stamp on a generic table for this analyzer."""
    cols = set(columns)
    gt = _GENERIC_GROUP_TYPE.get(analyzer_name)
    if gt is not None and ("site" in cols or "group" in cols):
        return {GROUP_TYPE: gt.value}
    return {}
