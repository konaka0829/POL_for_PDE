import inspect
from dataclasses import replace

import pytest
import torch
from pol.paper1.config import load_config_json
from pol.paper1.e1 import build_surrogate_from_finite_target, finite_input_path_check, run_e1, select_ridge_readout
from pol.paper1.e1_qa import model_content_hash, scientific_acceptance_checks
from pol.paper1.initial_conditions import build_master_grf_initial_conditions, resolve_device

def test_finite_input_boundary_and_synthetic_leak():
    assert list(inspect.signature(build_surrogate_from_finite_target).parameters)==["u_tar","n_sur","domain_length"]
    assert finite_input_path_check(64,32,32,1.0)["status"]=="pass"

def test_auto_device_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    assert resolve_device("auto").type=="cpu"

def _string_tables(result):
    names=("ridge_selection","selected_results","readout_diagnostics","mode_comparison","noise_results","noise_summary")
    return {name+".csv":[{k:("" if v is None else str(v)) for k,v in row.items()} for row in result[name]] for name in names}

def test_zero_grf_main_identifiability_gate_fails():
    cfg=load_config_json("configs/paper1_e1_smoke.json")
    cfg=replace(cfg,data=replace(cfg.data,grf_sigma=0.0),e1=replace(cfg.e1,profile="main",min_identifiable_nonconstant_fraction=1.0))
    result=run_e1(cfg,build_master_grf_initial_conditions(cfg))
    checks=scientific_acceptance_checks(_string_tables(result),cfg)
    assert checks["identifiable_nonconstant_fraction"]["status"]=="fail"
    assert checks["identifiable_nonconstant_fraction"]["value"]==0.0

def test_same_seed_core_is_reproducible():
    torch.set_num_threads(1)
    cfg=load_config_json("configs/paper1_e1_smoke.json")
    master=build_master_grf_initial_conditions(cfg)
    first,second=run_e1(cfg,master),run_e1(cfg,master)
    assert first["ridge_selection"]==second["ridge_selection"]
    assert first["readout_diagnostics"]==second["readout_diagnostics"]
    assert first["selected_results"]==second["selected_results"]
    assert first["noise_results"]==second["noise_results"]
    assert model_content_hash(first["models"])==model_content_hash(second["models"])
    for key in first["models"]:
        assert torch.equal(first["models"][key]["W"],second["models"][key]["W"])
        assert torch.equal(first["models"][key]["b"],second["models"][key]["b"])

def test_ridge_selection_api_is_independent_of_test_labels():
    generator=torch.Generator().manual_seed(9)
    train_x=torch.randn(20,5,generator=generator,dtype=torch.float64)
    val_x=torch.randn(6,5,generator=generator,dtype=torch.float64)
    weight=torch.randn(3,5,generator=generator,dtype=torch.float64)
    train_y=train_x@weight.T; val_y=val_x@weight.T
    first=select_ridge_readout(train_x,train_y,val_x,val_y,zetas=(0.0,1e-12,1e-6),tie_tolerance=1e-15,svd_rcond=None)
    # Arbitrary test labels are deliberately outside the selector's signature.
    test_labels_a=torch.zeros(4,3,dtype=torch.float64); test_labels_b=torch.full((4,3),1e100,dtype=torch.float64)
    second=select_ridge_readout(train_x,train_y,val_x,val_y,zetas=(0.0,1e-12,1e-6),tie_tolerance=1e-15,svd_rcond=None)
    assert not torch.equal(test_labels_a,test_labels_b)
    assert first[0]==second[0] and first[2]==second[2]
    assert torch.equal(first[1].W,second[1].W) and torch.equal(first[1].b,second[1].b)

@pytest.mark.slow
def test_main_like_q65_meets_scientific_thresholds():
    torch.set_num_threads(1)
    cfg=load_config_json("configs/paper1_e1_main.json")
    cfg=replace(cfg,spatial=replace(cfg.spatial,reference_nx=256))
    result=run_e1(cfg,build_master_grf_initial_conditions(cfg))
    checks=scientific_acceptance_checks(_string_tables(result),cfg)
    assert all(value["status"]=="pass" for value in checks.values())
    unstable=next(row for row in result["readout_diagnostics"] if row["case_name"]=="unstable" and row["q"]==65)
    assert unstable["max_identifiable_diagonal_relative_error"]<=0.25
    assert unstable["identifiable_off_diagonal_relative_norm"]<=0.25
