"""Artifact-only renderer for standard E1 figures."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pol.plots.types import PlotContext, PlotRecipe, PlotResult


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in tuple(row.items()):
            try:
                row[key] = int(value)
            except ValueError:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def _render(context: PlotContext) -> PlotResult:
    from pol.paper1.e1_plotting import create_e1_plots

    tables = {
        "mode_comparison": _rows(context.input_dir / "mode_comparison.csv"),
        "selected_results": _rows(context.input_dir / "selected_results.csv"),
        "readout_diagnostics": _rows(
            context.input_dir / "readout_diagnostics.csv"
        ),
        "noise_summary": _rows(context.input_dir / "noise_summary.csv"),
    }
    formats = tuple(context.settings.get("formats", ["png"]))
    dpi = int(context.settings.get("dpi", 160))
    return PlotResult(
        tuple(create_e1_plots(context.output_dir, tables, formats=formats, dpi=dpi))
    )


def _validate(settings: dict[str, Any]) -> dict[str, Any]:
    if set(settings) != {"formats", "dpi"}:
        raise ValueError("E1 standard plot settings require formats and dpi")
    formats = settings["formats"]
    dpi = settings["dpi"]
    if not isinstance(formats, list) or not formats or any(
        item not in {"png", "pdf", "svg"} for item in formats
    ):
        raise ValueError("formats must be a non-empty png/pdf/svg array")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    return {"formats": list(formats), "dpi": dpi}


RECIPE = PlotRecipe(
    recipe_id="paper1.e1.standard.v1",
    version="1",
    supported_experiment_kinds=("e1",),
    required_input_files=(
        "mode_comparison.csv",
        "selected_results.csv",
        "readout_diagnostics.csv",
        "noise_summary.csv",
        "resolved_config.json",
    ),
    render=_render,
    validate_settings=_validate,
)
