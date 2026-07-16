import math
import torch

from pol.paper1.config import load_config_json
from pol.paper1.e0 import run_reference_convergence
from pol.paper1.initial_conditions import build_master_grf_initial_conditions
from pol.paper1.solvers import solve_burgers_final_state


def test_reference_schema_shared_hash_selection_and_effective_step():
    cfg = load_config_json("configs/paper1_e0_smoke.json")
    result = run_reference_convergence(cfg, build_master_grf_initial_conditions(cfg))
    assert result["spatial_status"] == result["temporal_status"] == "pass"
    assert {row["kind"] for row in result["rows"]} == {"spatial", "temporal"}
    assert len({row["master_hash"] for row in result["rows"]}) == 1
    row = result["rows"][0]
    assert row["substeps_per_outer"] == math.ceil(row["requested_dt"] / row["requested_fine_dt"])
    assert row["effective_inner_step"] == row["requested_dt"] / row["substeps_per_outer"]


def test_solver_nan_inf_path():
    with torch.no_grad():
        bad = torch.zeros(1, 16); bad[0, 0] = float("nan")
    try:
        solve_burgers_final_state(bad, nu=.01, T=.01, dt=.01, fine_dt=.005, solver="split_step", dealias=True, domain_length=1.0)
    except FloatingPointError as exc:
        assert "sample indices" in str(exc)
    else:
        raise AssertionError("NaN input must fail")
