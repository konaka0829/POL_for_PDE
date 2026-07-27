"""Small explicit registry for matrix experiment plugins."""
from __future__ import annotations

from importlib import import_module
from typing import Any


_PLUGIN_FACTORIES: dict[str, str] = {
    "paper1_e1_resolution_v1": (
        "pol.paper1.matrix_plugins.e1_resolution:E1ResolutionPlugin"
    ),
}


def register_matrix_plugin(
    plugin_id: str, factory_path: str, *, replace: bool = False
) -> None:
    """Register an importable ``module:factory`` path.

    Callable objects are deliberately rejected because parent-only registry
    state is not authoritative in ``spawn`` workers.
    """
    if not plugin_id or (plugin_id in _PLUGIN_FACTORIES and not replace):
        raise ValueError(f"matrix plugin already registered or invalid: {plugin_id}")
    if not isinstance(factory_path, str) or ":" not in factory_path:
        raise ValueError("matrix plugin factory must be an importable module:path")
    _PLUGIN_FACTORIES[plugin_id] = factory_path


def matrix_plugin_factory_path(plugin_id: str) -> str:
    try:
        return _PLUGIN_FACTORIES[plugin_id]
    except KeyError as exc:
        raise ValueError(f"unknown matrix experiment plugin: {plugin_id}") from exc


def load_matrix_plugin(factory_path: str, *, expected_id: str) -> Any:
    module_name, attribute = factory_path.split(":", 1)
    factory = getattr(import_module(module_name), attribute)
    plugin = factory()
    if getattr(plugin, "plugin_id", None) != expected_id:
        raise ValueError(f"matrix plugin id mismatch: {expected_id}")
    return plugin


def get_matrix_plugin(plugin_id: str) -> Any:
    """Return a known plugin without introducing a dynamic plugin framework."""
    try:
        factory_path = _PLUGIN_FACTORIES[plugin_id]
    except KeyError as exc:
        raise ValueError(f"unknown matrix experiment plugin: {plugin_id}") from exc
    return load_matrix_plugin(factory_path, expected_id=plugin_id)
