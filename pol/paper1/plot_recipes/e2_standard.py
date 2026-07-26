"""Artifact-only renderer for standard E2 figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pol.plots.types import PlotContext, PlotRecipe, PlotResult


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in tuple(row.items()):
            if value == "":
                row[key] = None
                continue
            try:
                row[key] = int(value)
            except ValueError:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _render(context: PlotContext) -> PlotResult:
    from pol.paper1.config import load_config_json
    from pol.paper1.e2_plotting import create_e2_plots

    summary = _json(context.input_dir / "e2_summary.json")
    result = {
        "test_sweep": _rows(context.input_dir / "test_sweep.csv"),
        "model3_test_aggregate": _rows(
            context.input_dir / "model3_test_aggregate.csv"
        ),
        "shared_representatives": _json(
            context.input_dir / "shared_representatives.json"
        ),
        "model_specific_optima": _json(
            context.input_dir / "model_specific_optima.json"
        ),
        "convergence_results": _rows(
            context.input_dir / "convergence_results.csv"
        ),
        "convergence_summary": _json(
            context.input_dir / "convergence_summary.json"
        ),
        "pilot_n_sur": summary["actual_final_sweep_n_sur"],
        "selection_record_hash": summary["selection_record_hash"],
        "frozen_plan_hash": summary["frozen_plan_hash"],
    }
    config = load_config_json(context.input_dir / "resolved_config.json")
    formats = tuple(context.settings.get("formats", ["png", "pdf"]))
    dpi = int(context.settings.get("dpi", 180))
    return PlotResult(
        tuple(
            create_e2_plots(
                context.output_dir,
                result,
                config,
                formats=formats,
                dpi=dpi,
            )
        )
    )


def _validate(settings: dict[str, Any]) -> dict[str, Any]:
    from pol.paper1.plot_recipes.e1_standard import _validate as validate_standard

    return validate_standard(settings)


RECIPE = PlotRecipe(
    recipe_id="paper1.e2.standard.v1",
    version="1",
    supported_experiment_kinds=("e2",),
    required_input_files=(
        "test_sweep.csv",
        "model3_test_aggregate.csv",
        "shared_representatives.json",
        "model_specific_optima.json",
        "convergence_results.csv",
        "convergence_summary.json",
        "e2_summary.json",
        "resolved_config.json",
        "selection_record.json",
        "frozen_evaluation_plan.pt",
    ),
    render=_render,
    validate_settings=_validate,
)
