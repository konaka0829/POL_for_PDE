"""Artifact-only renderer for E1 matrix aggregate figures."""
from __future__ import annotations

from pol.plots.types import PlotContext, PlotRecipe, PlotResult


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate(settings: dict) -> dict:
    expected = {
        "q",
        "noise_level",
        "line_slice",
        "target_observation_surrogate_nx",
        "surrogate_line_target_data_nx",
        "surrogate_line_observation_dims",
        "formats",
        "dpi",
    }
    unknown = sorted(set(settings) - expected)
    missing = sorted(expected - set(settings))
    if unknown:
        raise ValueError(f"unknown E1 sweep plot setting: {unknown[0]}")
    if missing:
        raise ValueError(f"missing E1 sweep plot setting: {missing[0]}")
    _positive(settings["q"], "q")
    noise = settings["noise_level"]
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or noise < 0:
        raise ValueError("noise_level must be nonnegative")
    line = settings["line_slice"]
    if not isinstance(line, dict) or set(line) != {
        "target_data_nx",
        "surrogate_internal_nx",
    }:
        raise ValueError("line_slice has invalid keys")
    for key, value in line.items():
        _positive(value, f"line_slice.{key}")
    for key in (
        "target_observation_surrogate_nx",
        "surrogate_line_target_data_nx",
        "dpi",
    ):
        _positive(settings[key], key)
    dimensions = settings["surrogate_line_observation_dims"]
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("surrogate_line_observation_dims must be non-empty")
    for value in dimensions:
        _positive(value, "surrogate_line_observation_dims")
    formats = settings["formats"]
    if not isinstance(formats, list) or not formats or any(
        item not in {"png", "pdf", "svg"} for item in formats
    ):
        raise ValueError("formats must be a non-empty png/pdf/svg array")
    return dict(settings)


def _render(context: PlotContext) -> PlotResult:
    from pol.paper1.e1_sweep_plotting import generate_aggregate_plots

    manifest = generate_aggregate_plots(
        context.output_dir,
        dict(context.settings),
        input_dir=context.input_dir,
        write_manifest=False,
    )
    outputs = tuple(
        item for item in manifest["plots"] if item.get("status") == "pass"
    )
    if manifest["status"] != "pass":
        reasons = [
            str(item.get("reason"))
            for item in manifest["plots"]
            if item.get("status") != "pass"
        ]
        raise ValueError(f"aggregate plot generation failed: {reasons}")
    return PlotResult(outputs)


RECIPE = PlotRecipe(
    recipe_id="paper1.e1.resolution_sweep.v1",
    version="1",
    supported_experiment_kinds=("e1_matrix",),
    required_input_files=(
        "sweep_selected_results.csv",
        "sweep_readout_diagnostics.csv",
        "sweep_noise_summary.csv",
    ),
    render=_render,
    validate_settings=_validate,
)
