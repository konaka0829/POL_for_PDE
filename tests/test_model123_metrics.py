import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.model123_1d.experiments import ExperimentConfig, run_experiment
from pol.model123_1d.metrics import dataset_abs_l2h_error, dataset_rel_l2h_mean, rms_l2


def test_dataset_abs_l2h_error_matches_manual_value():
    pred = torch.tensor([[1.0, 3.0], [2.0, 6.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    # h = 1/2, per-sample errors = sqrt(2.5), sqrt(8.5)
    expected = math.sqrt((2.5 + 8.5) / 2.0)
    assert abs(dataset_abs_l2h_error(pred, target) - expected) < 1e-12


def test_abs_l2h_scales_with_domain_length():
    pred = torch.ones(3, 64, dtype=torch.float64)
    target = torch.zeros_like(pred)
    err_l1 = dataset_abs_l2h_error(pred, target, domain_length=1.0)
    err_l2 = dataset_abs_l2h_error(pred, target, domain_length=2.0)
    assert abs(err_l2 - math.sqrt(2.0) * err_l1) < 1e-12


def test_dataset_rel_l2h_mean_matches_manual_value():
    pred = torch.tensor([[2.0, 1.0], [3.0, 5.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    # numerators: sqrt(0.5), sqrt(10); denominators: 1, 1
    expected = 0.5 * (math.sqrt(0.5) + math.sqrt(10.0))
    assert abs(dataset_rel_l2h_mean(pred, target) - expected) < 1e-11


def test_rms_l2_is_alias_for_dataset_abs_l2h_error():
    pred = torch.tensor([[2.0, 1.0], [3.0, 5.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    assert abs(rms_l2(pred, target) - dataset_abs_l2h_error(pred, target)) < 1e-12


def test_run_experiment_returns_abs_and_relative_metrics(tmp_path):
    cfg = ExperimentConfig(
        total_samples=12,
        ntrain=8,
        ntest=4,
        nx=32,
        dt=1e-2,
        fine_dt=2e-3,
        batch_size=2,
        obs="full",
        J=8,
        K=2,
        reservoir="reaction_diffusion",
        model3_m=8,
        out_dir=str(tmp_path),
    )
    result = run_experiment(cfg)
    assert result["main_metric"] == "abs_l2h"
    assert "E1_train_abs_l2h" in result
    assert "E1_test_rel_l2h_mean" in result
