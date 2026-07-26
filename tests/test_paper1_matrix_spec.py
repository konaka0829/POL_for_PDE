from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pol.workflow.matrix import matrix_plan_to_dict
from pol.workflow.matrix_spec import expand_matrix, load_matrix_spec
from pol.workflow.registry import get_matrix_plugin


ROOT = Path(__file__).resolve().parents[1]


def _raw() -> dict:
    return json.loads(
        (
            ROOT / "configs/runs/paper1_e1_resolution_sweep_smoke.json"
        ).read_text(encoding="utf-8")
    )


def _load(tmp_path: Path, raw: dict):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_matrix_spec(path, repo_root=ROOT)


def _expand(spec):
    plugin = get_matrix_plugin(spec.aggregation_kind)
    return expand_matrix(
        spec, base=plugin.load_base(spec.base_config), plugin=plugin
    )


def test_smoke_and_main_plans_have_expected_counts() -> None:
    smoke = load_matrix_spec(
        ROOT / "configs/runs/paper1_e1_resolution_sweep_smoke.json",
        repo_root=ROOT,
    )
    main = load_matrix_spec(
        ROOT / "configs/runs/paper1_e1_resolution_sweep_main.json",
        repo_root=ROOT,
    )
    smoke_plan = matrix_plan_to_dict(smoke, repo_root=ROOT)
    main_plan = matrix_plan_to_dict(main, repo_root=ROOT)
    assert smoke_plan["unique_valid_cells"] == 3
    assert smoke_plan["full_observation_cells"] == 2
    assert smoke_plan["contains_n_tar_gt_J"]
    assert smoke_plan["contains_n_tar_lt_J"]
    assert main_plan["raw_run_counts"] == {
        "full_observation_resolution": 64,
        "target_observation_grid": 64,
        "surrogate_resolution_J65": 8,
        "surrogate_resolution_J96": 6,
    }
    assert main_plan["unique_valid_cells"] == 130


def test_matrix_plan_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    raw = _raw()
    raw["run"] = {"name": "plan", "output_root": str(tmp_path / "output")}
    spec = _load(tmp_path, raw)
    plan = matrix_plan_to_dict(spec, repo_root=ROOT)
    assert plan["execution_mode"] == "process_isolated_matrix"
    assert plan["unique_valid_cells"] == 3
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update({"unknown": 1}), r"unknown key at \$"),
        (
            lambda raw: raw["execution"].update({"workers": 2}),
            r"unknown key at \$\.execution",
        ),
        (
            lambda raw: raw["execution"].update({"jobs": True}),
            r"positive integer at \$\.execution\.jobs",
        ),
        (
            lambda raw: raw["matrix"]["experiments"].append(
                {
                    "name": "bad",
                    "fixed": {"spatial.target_data_nx": 32},
                    "grid": [
                        {
                            "path": "spatial.target_data_nx",
                            "values": [16],
                        }
                    ],
                    "copy": {},
                }
            ),
            r"conflicting override target",
        ),
    ],
)
def test_strict_schema_errors_include_paths(
    tmp_path: Path, mutation, message: str
) -> None:
    raw = _raw()
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, raw)


def test_non_scalar_unknown_and_bool_overrides_are_rejected(tmp_path: Path) -> None:
    for path, value, message in (
        ("spatial", 1, "not a scalar leaf"),
        ("spatial.missing", 1, "does not exist"),
        ("spatial.target_data_nx", True, "type mismatch"),
    ):
        raw = _raw()
        raw["matrix"]["explicit_runs"] = [{path: value}]
        with pytest.raises(ValueError, match=message):
            _expand(_load(tmp_path, raw))


def test_copy_cycle_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["matrix"]["explicit_runs"] = []
    raw["matrix"]["experiments"] = [
        {
            "name": "cycle",
            "fixed": {"spatial.surrogate_internal_nx": 32},
            "grid": [],
            "copy": {
                "spatial.target_data_nx": "spatial.observation_dim",
                "spatial.observation_dim": "spatial.target_data_nx",
            },
        }
    ]
    with pytest.raises(ValueError, match="copy cycle"):
        _expand(_load(tmp_path, raw))


def test_axis_and_explicit_order_dedup_and_membership(tmp_path: Path) -> None:
    raw = _raw()
    raw["matrix"]["experiments"] = [
        {
            "name": "declared_grid",
            "fixed": {"spatial.surrogate_internal_nx": 32},
            "grid": [
                {
                    "path": "spatial.target_data_nx",
                    "values": [32, 16],
                },
                {
                    "path": "spatial.observation_dim",
                    "values": [16, 32],
                },
            ],
            "copy": {},
        }
    ]
    raw["matrix"]["explicit_runs"] = [
        {
            "spatial.target_data_nx": 32,
            "spatial.surrogate_internal_nx": 32,
            "spatial.observation_dim": 16,
        }
    ]
    cells, invalid, counts = _expand(_load(tmp_path, raw))
    assert not invalid
    assert counts == {"declared_grid": 4, "explicit_runs": 1}
    assert [
        (cell.metadata["n_tar"], cell.metadata["J"]) for cell in cells
    ] == [(32, 16), (32, 32), (16, 16), (16, 32)]
    assert cells[0].experiment_memberships == (
        "declared_grid",
        "explicit_runs",
    )
    assert [cell.run_index for cell in cells] == list(range(4))


def test_invalid_policy_skip_records_invalid_cell(tmp_path: Path) -> None:
    raw = _raw()
    raw["matrix"]["invalid_run_policy"] = "skip"
    raw["matrix"]["explicit_runs"] = [
        {
            "spatial.target_data_nx": 32,
            "spatial.surrogate_internal_nx": 16,
            "spatial.observation_dim": 32,
        },
        _raw()["matrix"]["explicit_runs"][0],
    ]
    cells, invalid, _ = _expand(_load(tmp_path, raw))
    assert len(cells) == 1
    assert len(invalid) == 1
    assert invalid[0]["failure_type"] == "ValueError"
