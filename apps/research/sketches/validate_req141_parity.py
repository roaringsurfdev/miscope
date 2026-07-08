"""REQ_141 parity check — the neuron-frequency slice vs. the baselines (read-only).

Non-destructive: reads each baseline's existing ``activation_frequency_norm`` +
the legacy ``neuron_dynamics`` / ``transient_frequency`` artifacts, recomputes the
new per-epoch attribution, the reshaped neuron_dynamics tail, and the three derived
transient tables entirely in memory, and asserts value parity. Writes nothing.

Run: ``uv run python apps/research/sketches/validate_req141_parity.py``
"""

from __future__ import annotations

import numpy as np

import miscope
import miscope.query
from miscope.analysis.analyzers.neuron_dynamics import (
    _compute_commitment_epochs,
    _compute_switch_counts,
)
from miscope.analysis.derived_tables import (
    COMMITTED_COUNTS_TABLE,
    TRANSIENT_FREQUENCIES_TABLE,
    TRANSIENT_PEAK_MEMBERS_TABLE,
)
from miscope.analysis.library.fourier_basis import get_fourier_basis

BASELINES = [(113, 999, 598), (109, 485, 598), (101, 999, 598)]


def _new_attribution(variant, epochs, prime):
    """Recompute the per-epoch (dominant_freq, max_frac) the new analyzer would emit."""
    doms, fracs = [], []
    for ep in epochs:
        proj = variant.artifacts.load_epoch(
            "activation_frequency_norm", ep, fields=["mlp_out_freq_norm"]
        )
        norm = proj["mlp_out_freq_norm"]
        doms.append(np.argmax(norm, axis=0))
        fracs.append(np.max(norm, axis=0))
    return np.stack(doms), np.stack(fracs)


def _derived(variant):
    """Run the three transient derived queries over the baseline attribution table."""
    with miscope.query.open_variant(variant, ["neuron_frequency_attribution"]) as con:
        cc = con.df(COMMITTED_COUNTS_TABLE.query)
        con.con.register("committed_counts", cc)
        tf = con.df(TRANSIENT_FREQUENCIES_TABLE.query)
        members = con.df(TRANSIENT_PEAK_MEMBERS_TABLE.query)
    return cc, tf, members


def check(prime, seed, dseed):
    fam = miscope.load_family("modulo_addition_1layer")
    v = fam.get_variant(prime=prime, seed=seed, data_seed=dseed)
    old = v.artifacts.load_cross_epoch("neuron_dynamics")
    epochs = [int(e) for e in old["epochs"]]

    # 1. per-epoch attribution
    new_dom, new_frac = _new_attribution(v, epochs, prime)
    assert np.array_equal(new_dom, old["dominant_freq"]), "dominant_freq mismatch"
    assert np.allclose(new_frac, old["max_frac"], rtol=1e-3), "max_frac mismatch"

    # 2. reshaped neuron_dynamics tail
    thr = 3.0 / get_fourier_basis(prime).n_frequencies
    assert np.isclose(thr, float(np.ravel(old["threshold"])[0])), "threshold mismatch"
    sw = _compute_switch_counts(new_dom, new_frac, thr)
    ce = _compute_commitment_epochs(new_dom, new_frac, np.array(epochs), thr)
    assert np.array_equal(sw, old["switch_counts"]), "switch_counts mismatch"
    assert np.array_equal(ce, old["commitment_epochs"], equal_nan=True), "commitment mismatch"

    # 3. derived transient tables vs the legacy transient_frequency artifact
    old_tf = v.artifacts.load_cross_epoch("transient_frequency")
    _, tf, members = _derived(v)
    tf = tf.sort_values("frequency").reset_index(drop=True)
    old_ever = np.asarray(old_tf["ever_qualified_freqs"])
    assert np.array_equal(tf["frequency"].to_numpy(), old_ever), "ever_qualified mismatch"
    assert np.array_equal(tf["is_final"].to_numpy().astype(bool), old_tf["is_final"]), "is_final"
    assert np.array_equal(tf["peak_epoch"].to_numpy(), old_tf["peak_epoch"]), "peak_epoch mismatch"
    assert np.array_equal(tf["peak_count"].to_numpy(), old_tf["peak_count"]), "peak_count mismatch"
    assert np.array_equal(
        tf["homeless_count"].to_numpy(), old_tf["homeless_count"]
    ), "homeless_count mismatch"

    # peak members (ragged) — compare per ever-qualified frequency
    flat, offsets = old_tf["peak_members_flat"], old_tf["peak_members_offsets"]
    by_freq = {int(f): g["member_neuron"].to_numpy() for f, g in members.groupby("frequency")}
    for i, freq in enumerate(old_ever):
        old_members = np.sort(flat[offsets[i] : offsets[i + 1]])
        new_members = np.sort(by_freq.get(int(freq), np.array([], dtype=int)))
        assert np.array_equal(old_members, new_members), f"peak_members mismatch freq {freq}"

    n_transient = int((~tf["is_final"].to_numpy().astype(bool)).sum())
    print(
        f"  p{prime}/s{seed}/ds{dseed}: OK — {len(epochs)} epochs, d_mlp={new_dom.shape[1]}, "
        f"{len(old_ever)} ever-qualified ({n_transient} transient)"
    )


if __name__ == "__main__":
    print("REQ_141 parity (read-only) on the three baselines:")
    for p, s, d in BASELINES:
        check(p, s, d)
    print("All baselines parity-clean.")
