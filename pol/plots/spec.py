"""One strict parser and compatibility normalizer for plot task specs."""
from __future__ import annotations

from typing import Any, Mapping

from .registry import get_plot_recipe
from .types import PlotTaskSpec


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {path}")
    return value


def parse_plot_block(
    value: object,
    *,
    experiment_kind: str,
    path: str = "$.plots",
) -> tuple[bool, bool, tuple[PlotTaskSpec, ...]]:
    """Parse canonical tasks and normalize the legacy flat scalar form."""
    block = _object(value, path)
    allowed = {"enabled", "required", "recipes"}
    unknown = sorted(set(block) - allowed)
    missing = sorted(allowed - set(block))
    if unknown:
        raise ValueError(f"unknown key at {path}: {unknown[0]}")
    if missing:
        raise ValueError(f"missing required key at {path}.{missing[0]}")
    enabled, required, recipes = (
        block["enabled"],
        block["required"],
        block["recipes"],
    )
    if not isinstance(enabled, bool):
        raise ValueError(f"expected boolean at {path}.enabled")
    if not isinstance(required, bool):
        raise ValueError(f"expected boolean at {path}.required")
    if not isinstance(recipes, list):
        raise ValueError(f"expected array at {path}.recipes")
    if enabled != bool(recipes):
        raise ValueError(
            f"{path}.enabled must match whether recipes are present"
        )
    if required and not enabled:
        raise ValueError(f"{path}.required cannot be true when plots are disabled")

    tasks: list[PlotTaskSpec] = []
    for index, item in enumerate(recipes):
        item_path = f"{path}.recipes[{index}]"
        raw = _object(item, item_path)
        recipe_id = raw.get("id")
        if not isinstance(recipe_id, str) or not recipe_id:
            raise ValueError(f"expected non-empty string at {item_path}.id")
        if "settings" in raw:
            if set(raw) != {"id", "settings"}:
                extra = sorted(set(raw) - {"id", "settings"})
                raise ValueError(f"unknown key at {item_path}: {extra[0]}")
            settings = _object(raw["settings"], f"{item_path}.settings")
        else:
            # paper1-run-v2 compatibility form.
            if set(raw) != {"id", "formats", "dpi"}:
                extra = sorted(set(raw) - {"id", "formats", "dpi"})
                if extra:
                    raise ValueError(f"unknown key at {item_path}: {extra[0]}")
                raise ValueError(f"invalid legacy plot task at {item_path}")
            settings = {"formats": raw["formats"], "dpi": raw["dpi"]}
        recipe = get_plot_recipe(recipe_id)
        if experiment_kind not in recipe.supported_experiment_kinds:
            raise ValueError(
                f"unsupported plot recipe at {item_path}.id: {recipe_id}"
            )
        try:
            validated = recipe.validate_settings(settings)
        except ValueError as exc:
            raise ValueError(f"invalid settings at {item_path}.settings: {exc}") from exc
        tasks.append(PlotTaskSpec(recipe_id, validated))
    ids = [task.recipe_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate plot recipe at {path}.recipes")
    return enabled, required, tuple(tasks)
