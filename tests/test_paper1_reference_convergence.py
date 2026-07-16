import math
from dataclasses import replace
import torch

from pol.paper1.config import load_config_json
from pol.paper1.e0 import run_reference_convergence
from pol.paper1.e0 import E0SolverCache
from pol.paper1.initial_conditions import build_master_grf_initial_conditions
from pol.paper1.solvers import solve_burgers_final_state
from pol.paper1.solvers import BurgersFinalStateResult, BurgersSolverMetadata


def test_reference_schema_shared_hash_selection_and_effective_step():
    cfg = load_config_json("configs/paper1_e0_smoke.json")
    result = run_reference_convergence(cfg, build_master_grf_initial_conditions(cfg))
    assert result["spatial_status"] == result["temporal_status"] == "pass"
    assert {row["kind"] for row in result["rows"]} == {"spatial", "temporal", "joint"}
    assert result["joint_status"] == "pass"
    assert result["joint_row"]["status"] == "pass"
    assert result["cache_stats"]["hits"] >= 1
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


class _JointFailCache:
    def __init__(self): self.calls = 0
    def solve(self, u0, **kwargs):
        self.calls += 1
        fine = kwargs["fine_dt"] == 0.0025
        coarse_nx = u0.shape[-1] == 32
        factor = 1.0 + (0.06 if coarse_nx else 0.0) + (0.06 if not fine else 0.0)
        values = torch.full_like(u0, factor)
        sub = math.ceil(kwargs["dt"] / kwargs["fine_dt"])
        meta = BurgersSolverMetadata("split_step", kwargs["dt"], kwargs["fine_dt"], kwargs["dt"] / sub, 2, sub, kwargs["dealias"], kwargs["domain_length"], "float64", "cpu")
        return BurgersFinalStateResult(values, meta)
    def stats(self): return {"hits": 0, "misses": self.calls, "unique_solve_count": self.calls}


def test_joint_can_fail_after_independent_axes_pass():
    cfg = load_config_json("configs/paper1_e0_smoke.json")
    cfg = replace(cfg, e0=replace(cfg.e0, reference_tolerances=replace(cfg.e0.reference_tolerances, mean_relative_l2=.1, max_relative_l2=.1, low_mode_relative_l2=.1)))
    master = build_master_grf_initial_conditions(cfg)
    result = run_reference_convergence(cfg, master, cache=_JointFailCache())
    assert result["spatial_status"] == result["temporal_status"] == "pass"
    assert result["joint_status"] == "fail"
    assert result["joint_row"]["status"] == "fail"


def test_production_eligibility_excludes_too_coarse_candidate():
    cfg = load_config_json("configs/paper1_e0_smoke.json")
    cfg = replace(cfg, spatial=replace(cfg.spatial, target_data_nx=64))
    result = run_reference_convergence(cfg, build_master_grf_initial_conditions(cfg))
    assert not result["rows"][0]["eligible_for_production"]
    assert result["selected_spatial"]["candidate_nx"] == 64


def test_solver_cache_calls_identical_real_solver_once(monkeypatch):
    import pol.paper1.e0 as e0_module
    calls = 0
    real = e0_module.solve_burgers_final_state
    def counted(*args, **kwargs):
        nonlocal calls; calls += 1
        return real(*args, **kwargs)
    monkeypatch.setattr(e0_module, "solve_burgers_final_state", counted)
    cache = E0SolverCache(); u0 = torch.zeros(1, 16)
    kwargs = dict(nu=.01, T=.01, dt=.01, fine_dt=.005, solver="split_step", dealias=True, domain_length=1.0)
    first = cache.solve(u0, **kwargs); second = cache.solve(u0, **kwargs)
    assert calls == 1 and cache.stats() == {"hits": 1, "misses": 1, "unique_solve_count": 1}
    assert first.values.data_ptr() != second.values.data_ptr()
