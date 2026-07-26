from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pol.paper1.run_spec import load_run_spec


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["e0", "e1", "e2"])
def test_repository_specs_load(kind: str) -> None:
    spec = load_run_spec(
        ROOT / f"configs/runs/paper1_{kind}_smoke.json", repo_root=ROOT
    )
    assert spec.kind == kind
    assert spec.output_root == (ROOT / "outputs/paper1_runs").resolve()


def _raw(kind: str = "e2") -> dict[str, object]:
    return json.loads(
        (ROOT / f"configs/runs/paper1_{kind}_smoke.json").read_text(encoding="utf-8")
    )


def _load(tmp_path: Path, value: object):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return load_run_spec(path, repo_root=ROOT)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update({"surprise": 1}), "$: surprise"),
        (lambda d: d["execution"].update({"num_workers": 1}), "$.execution: num_workers"),
        (lambda d: d.update({"schema_version": "other"}), "$.schema_version"),
        (lambda d: d["run"].update({"name": "../bad"}), "$.run.name"),
        (lambda d: d["experiment"].update({"kind": "e3"}), "$.experiment.kind"),
        (lambda d: d["prerequisites"].clear(), "$.prerequisites.e0_config"),
        (lambda d: d["execution"].update({"torch_threads": 0}), "$.execution.torch_threads"),
        (lambda d: d["execution"].update({"batch_size": False}), "$.execution.batch_size"),
    ],
)
def test_strict_errors(tmp_path: Path, mutate, message: str) -> None:
    value = _raw()
    mutate(value)
    with pytest.raises(ValueError, match=message.replace("$", r"\$")):
        _load(tmp_path, value)


def test_nested_unknown_and_e0_prerequisite_rejected(tmp_path: Path) -> None:
    value = _raw("e1")
    value["execution"]["batch_size"] = 64
    with pytest.raises(ValueError, match=r"\$\.execution: batch_size"):
        _load(tmp_path, value)
    value = _raw("e0")
    value["prerequisites"]["e0_config"] = "configs/paper1_e0_smoke.json"
    with pytest.raises(ValueError, match=r"\$\.prerequisites: e0_config"):
        _load(tmp_path, value)


def test_wrong_sections_and_relative_resolution(tmp_path: Path) -> None:
    value = _raw("e1")
    value["experiment"]["config"] = "configs/paper1_e0_smoke.json"
    with pytest.raises(ValueError, match="does not contain section e1"):
        _load(tmp_path, value)
    value = _raw("e1")
    value["prerequisites"]["e0_config"] = "configs/paper1_e1_smoke.json"
    with pytest.raises(ValueError, match="does not contain section e0"):
        _load(tmp_path, value)
    spec = _load(tmp_path, _raw("e0"))
    assert spec.experiment_config == (ROOT / "configs/paper1_e0_smoke.json").resolve()


def test_non_object_and_missing_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"at \$"):
        _load(tmp_path, [])
    value = copy.deepcopy(_raw())
    del value["run"]["name"]
    with pytest.raises(ValueError, match=r"\$\.run\.name"):
        _load(tmp_path, value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda e0, e2: e2["domain"].update({"length": 2.0}),
            r"domain\.length=2\.0",
        ),
        (
            lambda e0, e2: e2["target"].update({"fine_dt": 0.001}),
            r"target\.\(dt, fine_dt\).+not present",
        ),
        (
            lambda e0, e2: (
                e0["e0"].update({"q_reference_check": 15}),
                e0["e0"]["model1_identity"].update({"target_output_dim": 15}),
            ),
            r"e0\.q_reference_check=15.+target_output_dim=17",
        ),
    ],
)
def test_e2_static_prerequisite_mismatch_is_rejected(
    tmp_path: Path, mutation, message: str
) -> None:
    e0 = json.loads(
        (ROOT / "configs/paper1_e0_smoke.json").read_text(encoding="utf-8")
    )
    e2 = json.loads(
        (ROOT / "configs/paper1_e2_smoke.json").read_text(encoding="utf-8")
    )
    mutation(e0, e2)
    e0_path = tmp_path / "e0.json"
    e2_path = tmp_path / "e2.json"
    e0_path.write_text(json.dumps(e0), encoding="utf-8")
    e2_path.write_text(json.dumps(e2), encoding="utf-8")
    run = _raw("e2")
    run["experiment"]["config"] = str(e2_path)
    run["prerequisites"]["e0_config"] = str(e0_path)
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, run)


def test_repository_e2_main_static_prerequisite_is_compatible() -> None:
    spec = load_run_spec(
        ROOT / "configs/runs/paper1_e2_main.json", repo_root=ROOT
    )
    assert spec.kind == "e2"
