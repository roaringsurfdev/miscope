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

from miscope.analysis.spec import AnalyzerSpec, Category

if TYPE_CHECKING:
    from miscope.analysis.protocols import Analyzer, CrossEpochAnalyzer, SecondaryAnalyzer
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

        SPEC = AnalyzerSpec(name="foo", category="primary", ...)

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

    Spec-based API (preferred):
        get_spec(name)           — single Spec
        list_specs()             — every registered Spec
        list_specs_by_category() — filtered by primary/secondary/cross_epoch
        get_factory(name)        — instantiation closure
        create(name)             — invoke the factory
        list_for_family(family)  — Specs filtered by a family's declarations

    Legacy class-based API (preserved for backwards compatibility):
        register(cls), register_secondary(cls), register_cross_epoch(cls)
        get(name), get_secondary(name), get_cross_epoch(name)
        get_for_family(family), get_secondary_for_family(family),
        get_cross_epoch_for_family(family)
    """

    # ---- Spec-based API ---------------------------------------------------

    @classmethod
    def get_spec(cls, name: str) -> AnalyzerSpec:
        if name not in _specs:
            raise KeyError(
                f"No Spec for analyzer '{name}'. Registered: {sorted(_specs)}"
            )
        return _specs[name]

    @classmethod
    def has_spec(cls, name: str) -> bool:
        return name in _specs

    @classmethod
    def list_specs(cls) -> list[AnalyzerSpec]:
        return list(_specs.values())

    @classmethod
    def list_specs_by_category(cls, category: Category) -> list[AnalyzerSpec]:
        return [s for s in _specs.values() if s.category == category]

    @classmethod
    def get_factory(cls, name: str) -> Callable[[], Any]:
        if name not in _factories:
            raise KeyError(
                f"No factory for analyzer '{name}'. Registered: {sorted(_factories)}"
            )
        return _factories[name]

    @classmethod
    def create(cls, name: str) -> Any:
        """Instantiate an analyzer by name (via its factory)."""
        return cls.get_factory(name)()

    @classmethod
    def list_for_family(cls, family: ModelFamily) -> list[AnalyzerSpec]:
        """Return Specs declared by a family, filtered by analyzer availability.

        Concatenates the family's ``analyzers``, ``secondary_analyzers``, and
        ``cross_epoch_analyzers`` lists (each may be missing) and returns the
        registered Specs. Names with no registered Spec are silently dropped
        (matches today's lenient behavior).
        """
        declared = (
            list(getattr(family, "analyzers", []))
            + list(getattr(family, "secondary_analyzers", []))
            + list(getattr(family, "cross_epoch_analyzers", []))
        )
        return [_specs[n] for n in declared if n in _specs]

    # ---- Legacy class-based API (back-compat) -----------------------------

    @classmethod
    def register(cls, analyzer_class: type) -> type:
        """Register a primary analyzer class without an explicit Spec.

        Legacy entry point — preserved for callers that pre-date REQ_120.
        When the analyzer already has a Spec registered (via the decorator),
        this is a no-op. Otherwise a conservative default Spec is created
        with ``requires_model_weights=True`` and ``requires_activation_cache=True``.
        """
        return cls._legacy_register(analyzer_class, category="primary")

    @classmethod
    def register_secondary(cls, analyzer_class: type) -> type:
        return cls._legacy_register(analyzer_class, category="secondary")

    @classmethod
    def register_cross_epoch(cls, analyzer_class: type) -> type:
        return cls._legacy_register(analyzer_class, category="cross_epoch")

    @classmethod
    def _legacy_register(cls, analyzer_class: type, category: Category) -> type:
        name = getattr(analyzer_class, "name", None)
        if name is None:
            raise ValueError(
                f"Analyzer {analyzer_class.__name__} must have a 'name' attribute"
            )
        if name in _specs:
            # Decorator-registered Spec wins; treat the legacy call as a no-op.
            return analyzer_class

        # Synthesize a conservative default Spec. ``requires`` is inferred
        # from existing class attributes (``depends_on`` for secondary,
        # ``requires`` for cross-epoch).
        requires: tuple[str, ...] = ()
        if category == "secondary":
            depends_on = getattr(analyzer_class, "depends_on", None)
            if depends_on:
                requires = (depends_on,)
        elif category == "cross_epoch":
            requires = tuple(getattr(analyzer_class, "requires", ()) or ())

        hooks_attr = getattr(analyzer_class, "required_hooks", None)
        required_hooks = tuple(hooks_attr) if hooks_attr else ()
        produces_summary = hasattr(analyzer_class, "get_summary_keys")

        spec = AnalyzerSpec(
            name=name,
            category=category,
            requires=requires,
            requires_model_weights=(category == "primary"),
            requires_activation_cache=(category == "primary"),
            required_hooks=required_hooks,
            produces_summary=produces_summary,
        )
        _specs[name] = spec
        _factories[name] = lambda: analyzer_class()
        return analyzer_class

    @classmethod
    def get(cls, name: str) -> Analyzer:
        spec = _specs.get(name)
        if spec is None or spec.category != "primary":
            available = sorted(
                n for n, s in _specs.items() if s.category == "primary"
            )
            raise KeyError(f"Analyzer '{name}' not found. Available: {available}")
        return cls.create(name)

    @classmethod
    def get_secondary(cls, name: str) -> SecondaryAnalyzer:
        spec = _specs.get(name)
        if spec is None or spec.category != "secondary":
            available = sorted(
                n for n, s in _specs.items() if s.category == "secondary"
            )
            raise KeyError(
                f"Secondary analyzer '{name}' not found. Available: {available}"
            )
        return cls.create(name)

    @classmethod
    def get_cross_epoch(cls, name: str) -> CrossEpochAnalyzer:
        spec = _specs.get(name)
        if spec is None or spec.category != "cross_epoch":
            available = sorted(
                n for n, s in _specs.items() if s.category == "cross_epoch"
            )
            raise KeyError(
                f"Cross-epoch analyzer '{name}' not found. Available: {available}"
            )
        return cls.create(name)

    @classmethod
    def get_for_family(cls, family: ModelFamily) -> list[Analyzer]:
        names = getattr(family, "analyzers", [])
        return [
            cls.create(n)
            for n in names
            if n in _specs and _specs[n].category == "primary"
        ]

    @classmethod
    def get_secondary_for_family(
        cls, family: ModelFamily
    ) -> list[SecondaryAnalyzer]:
        names = getattr(family, "secondary_analyzers", [])
        return [
            cls.create(n)
            for n in names
            if n in _specs and _specs[n].category == "secondary"
        ]

    @classmethod
    def get_cross_epoch_for_family(
        cls, family: ModelFamily
    ) -> list[CrossEpochAnalyzer]:
        names = getattr(family, "cross_epoch_analyzers", [])
        return [
            cls.create(n)
            for n in names
            if n in _specs and _specs[n].category == "cross_epoch"
        ]

    @classmethod
    def list_all(cls) -> list[str]:
        """Names of primary analyzers (legacy semantics)."""
        return sorted(n for n, s in _specs.items() if s.category == "primary")

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
