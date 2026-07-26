"""Strict lazy registry for plot recipes."""
from __future__ import annotations

from .types import PlotRecipe


def get_plot_recipe(recipe_id: str) -> PlotRecipe:
    """Return a known recipe without importing unrelated scientific code."""
    if recipe_id == "paper1.e1.standard.v1":
        from pol.paper1.plot_recipes.e1_standard import RECIPE
    elif recipe_id == "paper1.e2.standard.v1":
        from pol.paper1.plot_recipes.e2_standard import RECIPE
    elif recipe_id == "paper1.e1.resolution_sweep.v1":
        from pol.paper1.plot_recipes.e1_resolution_sweep import RECIPE
    else:
        raise ValueError(f"unknown plot recipe id: {recipe_id}")
    return RECIPE
