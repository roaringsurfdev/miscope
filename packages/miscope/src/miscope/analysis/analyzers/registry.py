"""Back-compat re-export shim for the analyzer Registry (REQ_120 / REQ_121).

The Registry's canonical home is :mod:`miscope.analysis.registry`. This
module re-exports the public surface so existing imports
(``from miscope.analysis.analyzers.registry import AnalyzerRegistry``)
continue to work. New code should import from
:mod:`miscope.analysis.registry` directly.
"""

from __future__ import annotations

from miscope.analysis.registry import AnalyzerRegistry, register_analyzer

__all__ = ["AnalyzerRegistry", "register_analyzer", "register_default_analyzers"]


def register_default_analyzers() -> None:
    """Register every built-in analyzer.

    Iterates over the analyzer modules under ``miscope.analysis.analyzers``
    and re-registers each module's ``SPEC`` + factory. The module-level
    ``@register_analyzer`` decorator handles first-import registration;
    this function exists so tests that ``clear()`` the registry can
    rehydrate it without depending on side-effects from already-cached
    module imports.
    """
    import importlib
    import pkgutil

    import miscope.analysis.analyzers as analyzers_pkg

    for module_info in pkgutil.iter_modules(analyzers_pkg.__path__):
        if module_info.name in {"registry", "__init__"}:
            continue
        full_name = f"{analyzers_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(full_name)
        spec = getattr(module, "SPEC", None)
        if spec is None:
            continue
        # Find the analyzer class whose `name` matches the SPEC.
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and getattr(attr, "name", None) == spec.name:
                from miscope.analysis import registry as reg_mod

                reg_mod._specs[spec.name] = spec
                reg_mod._factories[spec.name] = lambda cls=attr: cls()
                break


# Auto-register default analyzers on import
register_default_analyzers()
