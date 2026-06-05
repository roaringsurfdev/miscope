"""Key-decomposition rules against the cases in the decomposition map (REQ_110A).

Pure: no disk IO. Builds ``OutputField``s matching the real declarations and
asserts each npz-key style parses to the right ``(field, prefix coords)``.
"""

from __future__ import annotations

from miscope.analysis.output_schema import Coord
from miscope.analysis.output_schema import OutputField as F
from miscope.warehouse.decompose import KeyStyle, assign_keys, get_decomp


def _by_key(matches):
    return {m.npz_key: m for m in matches}


def test_flat_default_no_prefix():
    fields = (F.columnar("quality_score", "float32", ("variant", "epoch"), ""),)
    m = _by_key(assign_keys("fourier_frequency_quality", fields, ("quality_score",)))
    assert m["quality_score"].field.name == "quality_score"
    assert m["quality_score"].prefix_coords == {}


def test_prefix_us_site_with_underscores():
    # repr_geometry: {site}_{field}, single underscore, site contains '_'.
    fields = (
        F.columnar("radii", "float64", ("variant", "epoch", "site", "row_id"), ""),
        F.columnar("mean_radius", "float64", ("variant", "epoch", "site"), ""),
    )
    keys = ("resid_pre_radii", "resid_pre_mean_radius")
    m = _by_key(assign_keys("repr_geometry", fields, keys))
    assert m["resid_pre_radii"].field.name == "radii"
    assert m["resid_pre_radii"].prefix_coords == {"site": "resid_pre"}
    # Longest-name match: mean_radius is not stolen by radii.
    assert m["resid_pre_mean_radius"].field.name == "mean_radius"


def test_reversed_us_field_first():
    # weight_spectra: {field}_{site}, field='sv', site='W_E'.
    fields = (F.columnar("sv", "float32", ("variant", "epoch", "site", "row_id"), ""),)
    m = _by_key(assign_keys("weight_spectra", fields, ("sv_W_E", "sv_W_in")))
    assert m["sv_W_E"].prefix_coords == {"site": "W_E"}
    assert m["sv_W_in"].prefix_coords == {"site": "W_in"}


def test_prefix_dus_group_double_underscore():
    # parameter_trajectory: {group}__{field}.
    fields = (
        F.columnar("explained_variance", "float64", ("variant", "group", "row_id"), ""),
        F.columnar("epochs", "int64", ("variant", "epoch"), ""),
    )
    keys = ("all__explained_variance", "embedding__explained_variance", "epochs")
    m = _by_key(assign_keys("parameter_trajectory", fields, keys))
    assert m["all__explained_variance"].prefix_coords == {"group": "all"}
    assert m["epochs"].prefix_coords == {}  # flat field unaffected by the style


def test_dmd_group_and_site_gated_on_double_underscore():
    # parameter_dmd: group_{freq}__{matrix}__{field}; flat group-only fields stay flat.
    fields = (
        F.columnar("n_components", "int64", ("variant", "group", "site"), ""),
        F.columnar("windowed__max_modes", "int64", ("variant", "group", "site"), ""),
        F.columnar("populated_groups", "int64", ("variant", "group"), ""),
    )
    keys = (
        "group_8__W_in__n_components",
        "group_8__W_in__windowed__max_modes",
        "populated_groups",
    )
    m = _by_key(assign_keys("parameter_dmd", fields, keys))
    assert m["group_8__W_in__n_components"].prefix_coords == {"group": "group_8", "site": "W_in"}
    # Field name containing '__' is parsed correctly (rest after group+site).
    assert m["group_8__W_in__windowed__max_modes"].field.name == "windowed__max_modes"
    # group-only field is flat, not a mis-parsed DMD unit.
    assert m["populated_groups"].prefix_coords == {}
    assert not m["populated_groups"].loose


def test_loose_rescues_site_replicated_frequencies():
    # frequencies declared site-independent but written per-site on disk.
    fields = (
        F.columnar("frequencies", "int32", ("variant", "epoch", "frequency"), ""),
        F.tensor("power", "float64", ("variant", "epoch", "site"), ""),
    )
    keys = ("attn_pattern_frequencies", "mlp_out_frequencies", "attn_pattern_power")
    matches = assign_keys("activation_basis_projection", fields, keys)
    m = _by_key(matches)
    # strict match for the site-keyed tensor field
    assert m["attn_pattern_power"].field.name == "power"
    assert not m["attn_pattern_power"].loose
    # loose rescue for the site-replicated columnar field — no declared site coord
    assert m["attn_pattern_frequencies"].field.name == "frequencies"
    assert m["attn_pattern_frequencies"].loose
    assert m["attn_pattern_frequencies"].prefix_coords == {}


def test_unknown_keys_dropped():
    fields = (F.columnar("k", "int32", ("variant", "epoch"), ""),)
    matches = assign_keys("fourier_frequency_quality", fields, ("k", "totally_foreign_key"))
    assert {m.npz_key for m in matches} == {"k"}


def test_decomp_registry_has_expected_styles():
    assert get_decomp("weight_spectra").style is KeyStyle.REVERSED_US
    assert get_decomp("parameter_dmd").style is KeyStyle.DMD
    assert get_decomp("global_centroid_pca").style is KeyStyle.PREFIX_DUS
    assert get_decomp("not_a_real_analyzer").style is KeyStyle.FLAT
    assert Coord.SITE in get_decomp("repr_geometry").key_coords
