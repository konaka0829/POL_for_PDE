import inspect
import torch
from pol.paper1.e1 import build_surrogate_from_finite_target, finite_input_path_check
from pol.paper1.initial_conditions import resolve_device

def test_finite_input_boundary_and_synthetic_leak():
    assert list(inspect.signature(build_surrogate_from_finite_target).parameters)==["u_tar","n_sur","domain_length"]
    assert finite_input_path_check(64,32,32,1.0)["status"]=="pass"

def test_auto_device_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    assert resolve_device("auto").type=="cpu"
