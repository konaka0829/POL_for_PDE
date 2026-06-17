import csv
import json
import sys
from pathlib import Path

import pytest
import torch

from scripts.run_baseline_suite import main as baseline_main
from scripts.run_burgers_calibration_suite import coefficient_columns, main as calibration_main
from scripts.run_e0_smoke_suite import main as e0_main
from scripts.run_nonlinear_surrogate_suite import add_dlin_columns, spectral_error_rows
from scripts.suite_common import best_by_validation


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_b0_like_zero_split(path: Path) -> None:
    nx = 64
    torch.save(
        {
            "u0_train": torch.zeros(12, nx),
            "y_train": torch.zeros(12, nx),
            "u0_val": torch.zeros(6, nx),
            "y_val": torch.zeros(6, nx),
            "u0_test": torch.zeros(6, nx),
            "y_test": torch.zeros(6, nx),
            "metadata": {
                "T": 0.1,
                "dt": 0.01,
                "target_nu": 0.01,
                "nx": nx,
                "domain_length": 1.0,
                "ic_type": "grf",
                "solver": "split_step",
                "dealias": True,
                "target_equation": "burgers",
            },
        },
        path,
    )


def test_e0_smoke_suite_dry_run_writes_commands(tmp_path):
    out_dir = tmp_path / "e0"
    rc = e0_main(
        [
            "--config",
            "configs/B0_smoke.json",
            "--output-dir",
            str(out_dir),
            "--dry-run",
            "--python",
            sys.executable,
        ]
    )
    assert rc == 0
    rows = _read_csv(out_dir / "e0_smoke_summary.csv")
    assert {row["task"] for row in rows} == {
        "generate_b0_dataset",
        "zeta_static_model2",
        "zeta_static_model3",
        "headroom",
        "model1_burgers_matching_defect",
    }
    commands = json.loads((out_dir / "commands.json").read_text(encoding="utf-8"))
    assert len(commands) == 5
    assert all(record["status"] == "dry_run" for record in commands)


def test_baseline_suite_dry_run_cartesian_product(tmp_path):
    out_dir = tmp_path / "baseline"
    rc = baseline_main(
        [
            "--config",
            "configs/B0_smoke.json",
            "--data-file",
            str(tmp_path / "dummy.pt"),
            "--output-dir",
            str(out_dir),
            "--models",
            "model2,model3",
            "--reservoirs",
            "static,heat,advection",
            "--alpha-values",
            "1.0",
            "--dry-run",
            "--python",
            sys.executable,
        ]
    )
    assert rc == 0
    rows = _read_csv(out_dir / "baseline_summary.csv")
    assert len(rows) == 6
    assert len({(row["model"], row["reservoir"], row["alpha"]) for row in rows}) == 6
    assert all(row["status"] == "dry_run" for row in rows)
    commands = json.loads((out_dir / "commands.json").read_text(encoding="utf-8"))
    assert len(commands) == 6


@pytest.mark.slow
def test_baseline_suite_tiny_actual_run(tmp_path):
    data = tmp_path / "data.pt"
    out_dir = tmp_path / "baseline_actual"
    _write_b0_like_zero_split(data)
    rc = baseline_main(
        [
            "--config",
            "configs/B0_smoke.json",
            "--data-file",
            str(data),
            "--output-dir",
            str(out_dir),
            "--models",
            "model2",
            "--reservoirs",
            "static",
            "--zeta-grid",
            "1e-8",
            "--alpha-values",
            "1.0",
            "--python",
            sys.executable,
        ]
    )
    assert rc == 0
    rows = _read_csv(out_dir / "baseline_summary.csv")
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["selection_metric_name"] == "val_absL2h"


def test_burgers_calibration_coefficients():
    cols = coefficient_columns(alpha=0.5, target_nu=0.01, res_burgers_nu=0.02, res_burgers_b=2.0)
    assert cols["effective_nu"] == pytest.approx(0.01)
    assert cols["effective_b"] == pytest.approx(1.0)
    assert cols["mismatch_nu"] == pytest.approx(0.0)
    assert cols["mismatch_b"] == pytest.approx(0.0)
    assert cols["scaled_mismatch_norm"] == pytest.approx(0.0)


def test_burgers_calibration_dry_run_grid_and_schema(tmp_path):
    out_dir = tmp_path / "calibration"
    rc = calibration_main(
        [
            "--config",
            "configs/B0_smoke.json",
            "--data-file",
            str(tmp_path / "dummy.pt"),
            "--output-dir",
            str(out_dir),
            "--models",
            "model1",
            "--alpha-values",
            "0.5,1.0",
            "--res-burgers-nu-values",
            "0.01,0.02",
            "--res-burgers-b-values",
            "1.0",
            "--dry-run",
            "--python",
            sys.executable,
        ]
    )
    assert rc == 0
    rows = _read_csv(out_dir / "calibration_summary.csv")
    assert len(rows) == 4
    for key in ["effective_nu", "effective_b", "mismatch_nu", "mismatch_b", "scaled_mismatch_norm"]:
        assert key in rows[0]


def test_best_by_validation_ignores_test_for_selection():
    rows = [
        {"model": "model2", "reservoir": "static", "val_absL2h": 0.2, "test_absL2h": 0.01},
        {"model": "model2", "reservoir": "static", "val_absL2h": 0.1, "test_absL2h": 0.9},
    ]
    best = best_by_validation(rows, ["model", "reservoir"])
    assert best[0]["val_absL2h"] == 0.1
    assert best[0]["test_absL2h"] == 0.9


def test_best_by_validation_requires_validation_metric():
    rows = [
        {"model": "model2", "reservoir": "static", "test_absL2h": 0.01},
    ]
    with pytest.raises(ValueError, match="val_absL2h"):
        best_by_validation(rows, ["model", "reservoir"])


def test_e3_dlin_columns_and_spectral_error_helper():
    row = {"test_absL2h": 0.25}
    add_dlin_columns(row, {"D_lin_abs_l2h": 0.5, "headroom_H": 0.3, "linear_explained_variance": 0.7})
    assert row["improvement_over_dlin_abs"] == pytest.approx(0.25)
    assert row["ratio_to_dlin"] == pytest.approx(0.5)
    assert row["beats_dlin"] is True

    x = torch.arange(8, dtype=torch.float64) / 8.0
    target = torch.zeros(2, 8, dtype=torch.float64)
    pred = torch.sin(2.0 * torch.pi * x).repeat(2, 1)
    rows = spectral_error_rows(pred, target)
    assert rows[1]["mean_fft_error_sq"] > 0.0
    assert rows[0]["mean_fft_error_sq"] < rows[1]["mean_fft_error_sq"]
