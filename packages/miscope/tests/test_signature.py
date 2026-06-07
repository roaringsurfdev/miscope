"""Tests for REQ_145: provenance signatures — the one freshness predicate's core.

CoS coverage:
- compute_signature is deterministic and order/boundary sensitive.
- build_record folds each input axis (code version, recipe, checkpoint, upstreams)
  so a change in any axis changes the signature; the decision is input-derived only.
- explain_change attributes the axis that moved (skip transparency).
- SigRecord round-trips through its JSON form (the manifest storage shape).
"""

from __future__ import annotations

from miscope.analysis.signature import (
    SigRecord,
    build_record,
    compute_signature,
    digest,
    explain_change,
)


def test_compute_signature_deterministic():
    assert compute_signature(["a", "b"]) == compute_signature(["a", "b"])


def test_compute_signature_order_sensitive():
    assert compute_signature(["a", "b"]) != compute_signature(["b", "a"])


def test_compute_signature_boundary_sensitive():
    # NUL-delimited so concatenation ambiguity cannot collide.
    assert compute_signature(["ab", "c"]) != compute_signature(["a", "bc"])


def test_digest_is_order_independent():
    assert digest(["x", "y", "z"]) == digest(["z", "y", "x"])


def test_build_record_code_version_changes_sig():
    a = build_record(code_version=1, recipe="", checkpoint="100:5:9")
    b = build_record(code_version=2, recipe="", checkpoint="100:5:9")
    assert a.sig != b.sig


def test_build_record_checkpoint_changes_sig():
    a = build_record(code_version=1, checkpoint="100:5:9")
    b = build_record(code_version=1, checkpoint="100:5:10")  # mtime moved
    assert a.sig != b.sig


def test_build_record_recipe_changes_sig():
    a = build_record(code_version=1, recipe="", checkpoint="c")
    b = build_record(code_version=1, recipe="rs_abc", checkpoint="c")
    assert a.sig != b.sig


def test_build_record_upstream_changes_sig():
    a = build_record(code_version=1, upstream_sigs=["u1", "u2"])
    b = build_record(code_version=1, upstream_sigs=["u1", "u2-changed"])
    assert a.sig != b.sig
    assert a.upstream_digest != b.upstream_digest


def test_build_record_upstream_order_independent():
    a = build_record(code_version=1, upstream_sigs=["u1", "u2"])
    b = build_record(code_version=1, upstream_sigs=["u2", "u1"])
    assert a.sig == b.sig


def test_explain_change_missing():
    new = build_record(code_version=1, checkpoint="c")
    assert explain_change(None, new) == "missing"


def test_explain_change_fresh():
    rec = build_record(code_version=1, checkpoint="c")
    assert explain_change(rec, rec) == "fresh: signature unchanged"


def test_explain_change_code_version():
    old = build_record(code_version=1, checkpoint="c")
    new = build_record(code_version=2, checkpoint="c")
    assert explain_change(old, new) == "stale: code v1->v2"


def test_explain_change_checkpoint():
    old = build_record(code_version=1, checkpoint="c1")
    new = build_record(code_version=1, checkpoint="c2")
    assert explain_change(old, new) == "stale: checkpoint changed"


def test_explain_change_upstream():
    old = build_record(code_version=1, upstream_sigs=["u1"])
    new = build_record(code_version=1, upstream_sigs=["u2"])
    assert explain_change(old, new) == "stale: upstream changed"


def test_sigrecord_roundtrip():
    rec = build_record(code_version=3, recipe="rs", upstream_sigs=["a", "b"])
    assert SigRecord.from_json(rec.to_json()) == rec


def test_sigrecord_from_json_none_for_legacy():
    assert SigRecord.from_json(None) is None
    assert SigRecord.from_json({}) is None  # no 'sig' key — a legacy/partial entry
