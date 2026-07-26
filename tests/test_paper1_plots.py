from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest

from pol.paper1.run_spec import load_run_spec
from pol.plots.registry import get_plot_recipe
from pol.plots.runtime import execute_plot_tasks
from pol.plots.types import PlotRecipe, PlotRenderError, PlotResult, PlotTaskSpec


ROOT = Path(__file__).resolve().parents[1]


def test_registry_rejects_unknown_recipe() -> None:
    with pytest.raises(ValueError, match="unknown plot recipe"):
        get_plot_recipe("not.registered")


def test_plot_recipes_do_not_import_scientific_compute_modules() -> None:
    forbidden = {
        "pol.paper1.e0",
        "pol.paper1.e1",
        "pol.paper1.e2",
        "pol.paper1.solvers",
        "pol.paper1.readouts",
        "pol.burgers_spectral_1d",
    }
    directory = ROOT / "pol/paper1/plot_recipes"
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert not imports & forbidden, path


def test_v2_plot_block_and_v1_compatibility(tmp_path: Path) -> None:
    v1 = json.loads(
        (ROOT / "configs/runs/paper1_e1_smoke.json").read_text(encoding="utf-8")
    )
    v1["run"] = {"name": "v1", "output_root": str(tmp_path / "runs")}
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(v1), encoding="utf-8")
    loaded_v1 = load_run_spec(v1_path, repo_root=ROOT)
    assert loaded_v1.plots_enabled
    assert loaded_v1.plot_tasks[0].recipe_id == "paper1.e1.standard.v1"

    v2 = json.loads(json.dumps(v1))
    v2["schema_version"] = "paper1-run-v2"
    v2["run"]["name"] = "v2"
    del v2["execution"]["skip_plots"]
    v2["plots"] = {
        "enabled": True,
        "required": True,
        "recipes": [
            {
                "id": "paper1.e1.standard.v1",
                "formats": ["png"],
                "dpi": 123,
            }
        ],
    }
    v2_path = tmp_path / "v2.json"
    v2_path.write_text(json.dumps(v2), encoding="utf-8")
    loaded_v2 = load_run_spec(v2_path, repo_root=ROOT)
    assert loaded_v2.plots_required
    assert loaded_v2.plot_tasks[0].settings["dpi"] == 123


def test_plot_runtime_reuse_output_tamper_and_input_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "compute"
    input_dir.mkdir()
    (input_dir / "data.csv").write_text("x\n1\n", encoding="utf-8")
    calls = []

    def render(context):
        calls.append(dict(context.settings))
        (context.output_dir / "figure.png").write_bytes(
            f"png-{context.settings['dpi']}".encode()
        )
        return PlotResult(({"relative_path": "figure.png", "format": "png"},))

    recipe = PlotRecipe(
        "test.plot.v1",
        "1",
        ("test",),
        ("data.csv",),
        render,
        lambda settings: settings,
    )
    monkeypatch.setattr("pol.plots.runtime.get_plot_recipe", lambda _: recipe)
    task = PlotTaskSpec("test.plot.v1", {"dpi": 100})
    first = execute_plot_tasks(
        experiment_kind="test",
        input_dir=input_dir,
        figures_dir=tmp_path / "figures",
        tasks=(task,),
    )
    second = execute_plot_tasks(
        experiment_kind="test",
        input_dir=input_dir,
        figures_dir=tmp_path / "figures",
        tasks=(task,),
    )
    assert first[0]["executed_or_reused"] == "executed"
    assert second[0]["executed_or_reused"] == "reused"
    assert len(calls) == 1
    manifest_path = tmp_path / "figures/test.plot.v1/plot_manifest.json"
    manifest_before_reuse = manifest_path.read_bytes()
    execute_plot_tasks(
        experiment_kind="test",
        input_dir=input_dir,
        figures_dir=tmp_path / "figures",
        tasks=(task,),
    )
    assert manifest_path.read_bytes() == manifest_before_reuse

    output = tmp_path / "figures/test.plot.v1/figure.png"
    output.write_bytes(b"tampered")
    third = execute_plot_tasks(
        experiment_kind="test",
        input_dir=input_dir,
        figures_dir=tmp_path / "figures",
        tasks=(task,),
    )
    assert third[0]["executed_or_reused"] == "executed"
    (input_dir / "data.csv").write_text("x\n2\n", encoding="utf-8")
    fourth = execute_plot_tasks(
        experiment_kind="test",
        input_dir=input_dir,
        figures_dir=tmp_path / "figures",
        tasks=(task,),
    )
    assert fourth[0]["executed_or_reused"] == "executed"
    changed = replace(task, settings={"dpi": 200})
    fifth = execute_plot_tasks(
        experiment_kind="test",
        input_dir=input_dir,
        figures_dir=tmp_path / "figures",
        tasks=(changed,),
    )
    assert fifth[0]["executed_or_reused"] == "executed"
    assert len(calls) == 4


def test_artifact_only_renderers_map_required_inputs() -> None:
    assert "mode_comparison.csv" in get_plot_recipe(
        "paper1.e1.standard.v1"
    ).required_input_files
    assert "frozen_evaluation_plan.pt" in get_plot_recipe(
        "paper1.e2.standard.v1"
    ).required_input_files
    assert "sweep_selected_results.csv" in get_plot_recipe(
        "paper1.e1.resolution_sweep.v1"
    ).required_input_files


def test_plot_runtime_rejects_unsafe_renderer_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "compute"
    input_dir.mkdir()
    (input_dir / "data").write_text("ok")

    def render(context):
        return PlotResult(({"relative_path": "../escape.png"},))

    recipe = PlotRecipe(
        "unsafe.v1",
        "1",
        ("test",),
        ("data",),
        render,
        lambda settings: settings,
    )
    monkeypatch.setattr("pol.plots.runtime.get_plot_recipe", lambda _: recipe)
    with pytest.raises(ValueError, match="unsafe or duplicate"):
        execute_plot_tasks(
            experiment_kind="test",
            input_dir=input_dir,
            figures_dir=tmp_path / "figures",
            tasks=(PlotTaskSpec("unsafe.v1", {}),),
        )
    assert not (tmp_path / "escape.png").exists()


def test_plot_runtime_rejects_symlink_figures_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "compute"
    input_dir.mkdir()
    (input_dir / "data").write_text("ok")
    victim = tmp_path / "victim"
    victim.mkdir()
    figures = tmp_path / "figures"
    try:
        figures.symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    recipe = PlotRecipe(
        "safe.v1",
        "1",
        ("test",),
        ("data",),
        lambda context: PlotResult(()),
        lambda settings: settings,
    )
    monkeypatch.setattr("pol.plots.runtime.get_plot_recipe", lambda _: recipe)
    with pytest.raises(ValueError, match="must not be a symlink"):
        execute_plot_tasks(
            experiment_kind="test",
            input_dir=input_dir,
            figures_dir=figures,
            tasks=(PlotTaskSpec("safe.v1", {}),),
        )
    assert list(victim.iterdir()) == []


def test_plot_runtime_records_partial_format_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "compute"
    input_dir.mkdir()
    (input_dir / "data").write_text("ok")

    def render(context):
        (context.output_dir / "figure.png").write_bytes(b"png")
        raise PlotRenderError(
            "pdf failed",
            outputs=(
                {
                    "relative_path": "figure.png",
                    "format": "png",
                    "status": "created",
                },
            ),
            failures=(
                {
                    "relative_path": "figure.pdf",
                    "format": "pdf",
                    "status": "fail",
                    "reason": "backend failure",
                },
            ),
        )

    recipe = PlotRecipe(
        "partial.v1",
        "1",
        ("test",),
        ("data",),
        render,
        lambda settings: settings,
    )
    monkeypatch.setattr("pol.plots.runtime.get_plot_recipe", lambda _: recipe)
    with pytest.raises(PlotRenderError, match="pdf failed"):
        execute_plot_tasks(
            experiment_kind="test",
            input_dir=input_dir,
            figures_dir=tmp_path / "figures",
            tasks=(PlotTaskSpec("partial.v1", {}),),
        )
    manifest = json.loads(
        (tmp_path / "figures/partial.v1/plot_manifest.json").read_text()
    )
    assert manifest["outputs"][0]["format"] == "png"
    assert manifest["format_failures"][0]["format"] == "pdf"
