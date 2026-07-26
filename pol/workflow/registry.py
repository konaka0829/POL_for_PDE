"""Small explicit registry for matrix experiment plugins."""
from __future__ import annotations

from typing import Any


def get_matrix_plugin(plugin_id: str) -> Any:
    """Return a known plugin without introducing a dynamic plugin framework."""
    if plugin_id == "paper1_e1_resolution_v1":
        from pol.paper1.matrix_plugins.e1_resolution import E1ResolutionPlugin

        return E1ResolutionPlugin()
    raise ValueError(f"unknown matrix aggregation plugin: {plugin_id}")
