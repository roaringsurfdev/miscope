"""Analyzer Registry — Specs + factories (REQ_120).

The Registry is the single index of "what analyzers exist." Each analyzer
self-registers via the ``@register_analyzer(SPEC)`` decorator. Callers
ask the Registry for Specs (declarative metadata) or factories
(instantiation closures). The Planner consults Specs to make load-time
decisions (skip the forward pass when no analyzer needs activations);
entry points enumerate analyzers via the Registry instead of hand-coding
imports.

This module supersedes the class-based registry that lived at
``miscope.analysis.analyzers.registry``; that module is preserved as a
re-export shim for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    from miscope.families.protocols import ModelFamily

T = TypeVar("T", bound=type)

# Module-level storage. The Registry is a singleton accessed through the
# AnalyzerRegistry class methods; these dicts are the backing state.
_specs: dict[str, AnalyzerSpec] = {}
_factories: dict[str, Callable[[], Any]] = {}


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def register_analyzer(spec: AnalyzerSpec) -> Callable[[T], T]:
    """Decorator that registers an analyzer class under its Spec.

    Usage:

        SPEC = AnalyzerSpec(name="foo", inputs=(...), ...)

        @register_analyzer(SPEC)
        class FooAnalyzer:
            name = "foo"

            def analyze(self, ctx): ...

    The Spec's ``name`` must match the class's ``name`` attribute. The
    decorator stores both a Spec and a factory closure; either can be
    retrieved via ``AnalyzerRegistry``.
    """

    def decorator(cls: T) -> T:
        cls_name = getattr(cls, "name", None)
        if cls_name is not None and cls_name != spec.name:
            raise ValueError(
                f"Spec name mismatch: SPEC.name={spec.name!r} but "
                f"{cls.__name__}.name={cls_name!r}. Keep them in sync."
            )
        _specs[spec.name] = spec
        _factories[spec.name] = lambda: cls()  # type: ignore[operator]
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Spec-based query API (preferred)
# ---------------------------------------------------------------------------


class AnalyzerRegistry:
    """Registry of analyzer Specs and factories.

    Spec-based query API — the single canonical surface (REQ_132):
        get_spec(name)           — single Spec
        has_spec(name)           — membership test
        list_specs()             — every registered Spec
        get_factory(name)        — instantiation closure
        create(name)             — invoke the factory
        list_for_family(family)  — Specs declared by a family's flat list
        list_all_names()         — names of every registered analyzer
        is_registered(name)      — membership test (alias of has_spec)
        clear()                  — reset (testing)

    The pipeline derives execution phase from each Spec's ``inputs``; the
    Registry exposes no phase/category vocabulary.
    """

    @classmethod
    def get_spec(cls, name: str) -> AnalyzerSpec:
        if name not in _specs:
            raise KeyError(f"No Spec for analyzer '{name}'. Registered: {sorted(_specs)}")
        return _specs[name]

    @classmethod
    def has_spec(cls, name: str) -> bool:
        return name in _specs

    @classmethod
    def list_specs(cls) -> list[AnalyzerSpec]:
        return list(_specs.values())

    @classmethod
    def get_factory(cls, name: str) -> Callable[[], Any]:
        if name not in _factories:
            raise KeyError(f"No factory for analyzer '{name}'. Registered: {sorted(_factories)}")
        return _factories[name]

    @classmethod
    def create(cls, name: str) -> Any:
        """Instantiate an analyzer by name (via its factory)."""
        return cls.get_factory(name)()

    @classmethod
    def list_for_family(cls, family: ModelFamily) -> list[AnalyzerSpec]:
        """Return the registered Specs declared by a family.

        Reads the family's single flat ``analyzers`` list (REQ_132) and
        returns the registered Specs. Names with no registered Spec are
        silently dropped (matches today's lenient behavior). Execution
        order is derived downstream from each Spec's ``inputs``.
        """
        declared = list(getattr(family, "analyzers", []))
        return [_specs[n] for n in declared if n in _specs]

    @classmethod
    def list_all_names(cls) -> list[str]:
        """Names of every registered analyzer regardless of category."""
        return sorted(_specs)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in _specs

    @classmethod
    def clear(cls) -> None:
        """Clear all registered analyzers. Mainly for testing."""
        _specs.clear()
        _factories.clear()
