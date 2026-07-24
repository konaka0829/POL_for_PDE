import copy
import json

import pytest
import torch

from pol.paper1.config import config_from_dict, load_config_json
from pol.paper1.e2 import (
    E2_SCHEMA_VERSION,
    dry_run_cost_summary,
    select_ridge,
    stable_hash,
    tensor_hash,
    validate_frozen_evaluation_plan,
)


def test_selection_provenance_and_hash_ignore_runtime(monkeypatch):
    torch.manual_seed(4)
    x = torch.randn(12, 5, dtype=torch.float64)
    y = torch.randn(12, 3, dtype=torch.float64)
    _, _, first = select_ridge(
        x[:8], y[:8], x[8:], y[8:], (0.0, 1e-6),
        tolerance=0.0, svd_rcond=None)
    monkeypatch.setattr("time.perf_counter", lambda: 1e30)
    _, _, second = select_ridge(
        x[:8], y[:8], x[8:], y[8:], (0.0, 1e-6),
        tolerance=0.0, svd_rcond=None)
    assert first == second
    assert stable_hash(first) == stable_hash(second)
    assert all("runtime" not in key for row in first for key in row)


def test_ridge_path_uses_one_svd_and_matches_single_zeta(monkeypatch):
    torch.manual_seed(5)
    x = torch.randn(14, 6, dtype=torch.float64)
    y = torch.randn(14, 4, dtype=torch.float64)
    calls = 0
    original = torch.linalg.svd

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted)
    zetas = (0.0, 1e-8, 1e-4)
    _, _, path = select_ridge(
        x[:10], y[:10], x[10:], y[10:], zetas,
        tolerance=0.0, svd_rcond=None, _return_candidate_models=True)
    assert calls == 1
    path_predictions = {
        row["zeta"]: row["_model"](x[10:]).clone() for row in path}
    for zeta in zetas:
        model, _, _ = select_ridge(
            x[:10], y[:10], x[10:], y[10:], (zeta,),
            tolerance=0.0, svd_rcond=None)
        torch.testing.assert_close(path_predictions[zeta], model(x[10:]))


def test_main_cost_path_reuse_is_independent_of_zeta_count():
    config = load_config_json("configs/paper1_e2_main.json")
    cost = dry_run_cost_summary(config)
    assert cost["axis_row_count"] == 100
    assert cost["unique_physical_point_count_upper_bound"] == 50
    assert cost["model2_svd_count"] == 50
    assert cost["model3_svd_count_after_path_reuse"] == 4500
    assert cost["model3_svd_count_legacy_single_zeta"] == 27000
    raw = config.to_dict()
    raw["e2"]["ridge"]["zetas"] = tuple((*raw["e2"]["ridge"]["zetas"], 1e-3))
    more = dry_run_cost_summary(config_from_dict(raw))
    assert more["model3_svd_count_after_path_reuse"] == cost["model3_svd_count_after_path_reuse"]


def test_model3_tie_break_must_be_exact():
    raw = json.loads(open("configs/paper1_e2_smoke.json").read())
    raw["e2"]["model3"]["tie_break"] = ["first_in_config_order"]
    with pytest.raises(ValueError, match="must exactly equal"):
        config_from_dict(raw)


def test_frozen_plan_rejects_tensor_tamper(tmp_path):
    models = {"point": {"W": torch.eye(2, dtype=torch.float64)}}
    hashes = {"models.point.W": tensor_hash(models["point"]["W"])}
    hashable = {
        "schema_version": E2_SCHEMA_VERSION,
        "selection_record_hash": "s",
        "tensor_hashes": hashes,
        "model_tensor_hashes": hashes,
    }
    payload = {
        "schema_version": E2_SCHEMA_VERSION,
        "selection_record_hash": "s", "models": models,
        "tensor_hashes": hashes, "plan_content_hash": stable_hash(hashable)}
    path = tmp_path / "plan.pt"
    torch.save(payload, path)
    validate_frozen_evaluation_plan(path, expected_selection_hash="s")
    changed = copy.deepcopy(payload)
    changed["models"]["point"]["W"][0, 0] += 1
    torch.save(changed, path)
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        validate_frozen_evaluation_plan(path, expected_selection_hash="s")
