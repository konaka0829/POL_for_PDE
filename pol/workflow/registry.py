"""Small explicit registry for matrix experiment plugins."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable


_PLUGIN_FACTORIES: dict[str, str | Callable[[], Any]] = {
    "paper1_e1_resolution_v1": (
        "pol.paper1.matrix_plugins.e1_resolution:E1ResolutionPlugin"
    ),
}


def register_matrix_plugin(
    plugin_id: str, factory: Callable[[], Any], *, replace: bool = False
) -> None:
    """Register an explicit plugin factory, primarily for local/test plugins."""
    if not plugin_id or (plugin_id in _PLUGIN_FACTORIES and not replace):
        raise ValueError(f"matrix plugin already registered or invalid: {plugin_id}")
    _PLUGIN_FACTORIES[plugin_id] = factory


def get_matrix_plugin(plugin_id: str) -> Any:
    """Return a known plugin without introducing a dynamic plugin framework."""
    try:
        factory = _PLUGIN_FACTORIES[plugin_id]
    except KeyError as exc:
        raise ValueError(f"unknown matrix experiment plugin: {plugin_id}") from exc
    if isinstance(factory, str):
        module_name, attribute = factory.split(":", 1)
        factory = getattr(import_module(module_name), attribute)
    plugin = factory()
    if getattr(plugin, "plugin_id", None) != plugin_id:
        raise ValueError(f"matrix plugin id mismatch: {plugin_id}")
    return plugin
