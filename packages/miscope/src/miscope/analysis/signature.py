"""Provenance signatures for incremental refresh (REQ_145).

A node's recompute decision is an **input-derived** signature: a stable hash of
the producer's code version, its recipe (REQ_138), the identity of what it reads
(an upstream signature, or — for a model-driven primary — a checkpoint
fingerprint), and nothing of its own output. Stamped at write time, compared at
plan time. A downstream folds in its upstreams' signatures, so a changed upstream
automatically changes the downstream signature — forward propagation through the
dependency DAG does the invalidation, no reverse walk required for execution.

This module is the **swappable seam** (REQ_145 Must-have): the whole signature
mechanism is `compute_signature` + `SigRecord` + a tiny set of builders, with zero
bleed into analyzer bodies. A later adoption of an off-the-shelf orchestration tool
(should that ever be re-justified) maps onto this boundary rather than unwinding it.

The decision is non-circular by construction: a node's signature never depends on
its own output, only on its declared inputs. An optional output data-version hash
(fork c) would be a robustness layer on top — deliberately deferred.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Truncated hex length for stored signatures — 16 hex chars (64 bits) is ample
# collision resistance for one variant's analyzer/table set while keeping the
# manifest compact and human-scannable.
_SIG_LEN = 16

# Manifest key for a cross-epoch analyzer's single artifact (per-epoch keys are
# the epoch number as a string, which never collides with this sentinel).
CROSS_EPOCH_KEY = "__cross_epoch__"


def compute_signature(components: Iterable[str]) -> str:
    """Hash an ordered sequence of string components into a stable signature.

    Components are NUL-delimited so ``["ab", "c"]`` and ``["a", "bc"]`` hash
    differently. Deterministic across processes (no salt, fixed algorithm).
    """
    h = hashlib.sha256()
    for component in components:
        h.update(component.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:_SIG_LEN]


def digest(signatures: Iterable[str]) -> str:
    """Order-independent digest of a *set* of upstream signatures.

    Sorted before hashing so the fold over a node's upstreams (or a table's
    feeders) does not depend on iteration order — only on the multiset of
    contributing signatures.
    """
    return compute_signature(sorted(signatures))


@dataclass(frozen=True)
class SigRecord:
    """A stamped signature plus the axes it was built from (for attribution).

    Storing the axes — not just the final ``sig`` — lets the planner answer
    *why* a node is stale by a field compare (``explain_change``) instead of an
    opaque hash mismatch: code version moved, the checkpoint changed, or an
    upstream changed. Serializes to/from plain JSON dicts so the storage
    primitive (the signature manifest) stays free of this type.
    """

    sig: str
    code_version: int
    recipe: str = ""
    checkpoint: str | None = None
    upstream_digest: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "sig": self.sig,
            "code_version": self.code_version,
            "recipe": self.recipe,
            "checkpoint": self.checkpoint,
            "upstream_digest": self.upstream_digest,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> SigRecord | None:
        """Rebuild a record from a manifest entry; ``None`` for a missing/legacy entry."""
        if not data or "sig" not in data:
            return None
        return cls(
            sig=str(data["sig"]),
            code_version=int(data.get("code_version", 0)),
            recipe=str(data.get("recipe", "")),
            checkpoint=data.get("checkpoint"),
            upstream_digest=data.get("upstream_digest"),
        )


def build_record(
    *,
    code_version: int,
    recipe: str = "",
    checkpoint: str | None = None,
    upstream_sigs: Iterable[str] | None = None,
) -> SigRecord:
    """Assemble a :class:`SigRecord` from a node's input axes.

    - Model-driven primary (per epoch): pass ``checkpoint`` (the epoch's
      fingerprint), no ``upstream_sigs``.
    - Artifact-derived / cross-epoch: pass ``upstream_sigs`` (the stored or
      projected signatures of every declared upstream it reads), no ``checkpoint``.
    - ``code_version`` is the producer's manual version (``AnalyzerSpec.version`` or,
      for a derived table, a hash folding its SQL text into the version — composed
      by the caller).
    """
    upstream_digest = digest(upstream_sigs) if upstream_sigs is not None else None
    components = [
        f"v={code_version}",
        f"recipe={recipe}",
        f"ckpt={checkpoint or ''}",
        f"up={upstream_digest or ''}",
    ]
    return SigRecord(
        sig=compute_signature(components),
        code_version=code_version,
        recipe=recipe,
        checkpoint=checkpoint,
        upstream_digest=upstream_digest,
    )


def explain_change(old: SigRecord | None, new: SigRecord) -> str:
    """Human-readable reason a node is fresh or stale — skip transparency (CoS).

    A generic fallback: the planner, which knows the per-upstream old/new
    signatures, can name the specific changed upstream before falling back here.
    """
    if old is None:
        return "missing"
    if old.sig == new.sig:
        return "fresh: signature unchanged"
    if old.code_version != new.code_version:
        return f"stale: code v{old.code_version}->v{new.code_version}"
    if old.checkpoint != new.checkpoint:
        return "stale: checkpoint changed"
    if old.upstream_digest != new.upstream_digest:
        return "stale: upstream changed"
    if old.recipe != new.recipe:
        return "stale: recipe changed"
    return "stale: signature changed"
