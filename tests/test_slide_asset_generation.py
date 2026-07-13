import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from test_utils import run_cli

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.plotting import plot_feature_vector, plot_single_waveform, plot_spacetime, plot_waveform_overlay


def _assert_all_formats(stem: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        path = stem.with_suffix("." + ext)
        assert path.exists()
        assert path.stat().st_size > 0


def _write_tiny_mat(path: Path, *, n: int = 10, s: int = 16, T: float = 0.02) -> None:
    x = np.linspace(0.0, 1.0, s, endpoint=False, dtype=np.float64)
    a = []
    u = []
    for idx in range(n):
        values = 0.1 * np.sin(2.0 * np.pi * (idx % 3 + 1) * x)
        a.append(values)
        u.append(values)
    savemat(
        path,
        {
            "a": np.asarray(a, dtype=np.float32),
            "u": np.asarray(u, dtype=np.float32),
            "T": np.asarray([[T]], dtype=np.float64),
            "dt": np.asarray([[0.01]], dtype=np.float64),
            "nu": np.asarray([[0.01]], dtype=np.float64),
            "domain_length": np.asarray([[1.0]], dtype=np.float64),
            "solver": np.asarray(["split_step"]),
            "dealias": np.asarray([[1]], dtype=np.int32),
            "equation": np.asarray(["burgers"]),
        },
    )


def _write_tiny_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "domain": {"length": 1.0, "periodic": True, "nx": 16},
                "target": {"equation": "burgers", "nu": 0.01, "T": 0.02, "dt": 0.01, "solver": "split_step", "dealias": True},
                "data": {"ic_type": "synthetic", "ntrain": 4, "nval": 2, "ntest": 4},
                "readout": {"ridge_convention": "normalized_empirical_l2h_unweighted_frobenius", "ridge_dtype": "float64"},
            }
        ),
        encoding="utf-8",
    )


def test_slide_plotting_utilities_save_all_formats(tmp_path):
    x = np.linspace(0.0, 1.0, 16, endpoint=False)
    y = np.sin(2.0 * np.pi * x)
    plot_single_waveform(x, y, str(tmp_path / "single"))
    plot_waveform_overlay(x, [{"y": y, "label": "a"}, {"y": 0.5 * y, "label": "b", "linestyle": "--"}], str(tmp_path / "overlay"))
    plot_feature_vector(np.arange(32), str(tmp_path / "feature"), max_points=8)
    t = np.linspace(0.0, 0.1, 5)
    u_xt = np.stack([np.sin(2.0 * np.pi * x + ti) for ti in t], axis=0)
    plot_spacetime(x, t, u_xt, str(tmp_path / "spacetime"))
    for name in ("single", "overlay", "feature", "spacetime"):
        _assert_all_formats(tmp_path / name)


def test_slide_scripts_help():
    for script in [
        "scripts/make_model123_stage_waveform_assets.py",
        "scripts/make_slide_spacetime_assets.py",
        "scripts/rerun_best_model123_for_slide_assets.py",
    ]:
        proc = subprocess.run([sys.executable, script, "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_save_predictions_and_slide_asset_smoke(tmp_path):
    data_file = tmp_path / "tiny.mat"
    config_file = tmp_path / "tiny_config.json"
    out_dir = tmp_path / "run"
    stage_dir = tmp_path / "stage_assets"
    spacetime_dir = tmp_path / "spacetime_assets"
    _write_tiny_mat(data_file)
    _write_tiny_config(config_file)
    cmd = [
        sys.executable,
        "model123_burgers_1d.py",
        "--config",
        str(config_file),
        "--model",
        "model2",
        "--reservoir",
        "reaction_diffusion",
        "--data-file",
        str(data_file),
        "--train-split",
        "0.6",
        "--ntrain",
        "4",
        "--nval",
        "2",
        "--ntest",
        "4",
        "--T",
        "0.02",
        "--Ttilde",
        "0.02",
        "--dt",
        "0.01",
        "--batch-size",
        "4",
        "--K",
        "1",
        "--obs",
        "full",
        "--ridge-zeta",
        "1e-8",
        "--data-dtype",
        "preserve",
        "--sim-dtype",
        "float64",
        "--ridge-dtype",
        "float64",
        "--device",
        "cpu",
        "--out-dir",
        str(out_dir),
        "--save-model",
        "--save-predictions",
    ]
    proc = run_cli(cmd, cwd=REPO_ROOT, timeout=90)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (out_dir / "predictions.pt").exists()
    assert (out_dir / "sample_metrics.csv").exists()
    payload = torch.load(out_dir / "predictions.pt", map_location="cpu")
    assert {"x_test", "y_test", "pred_test", "per_sample_absL2h", "per_sample_relL2"}.issubset(payload.keys())

    proc = run_cli(
        [
            sys.executable,
            "scripts/make_model123_stage_waveform_assets.py",
            "--run-dir",
            str(out_dir),
            "--sample-index",
            "0",
            "--out-dir",
            str(stage_dir),
            "--shared-ylim",
            "--device",
            "cpu",
            "--dtype",
            "float64",
        ],
        cwd=REPO_ROOT,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (stage_dir / "asset_manifest.json").exists()
    _assert_all_formats(stage_dir / "model2_sample000_target_vs_prediction")
    _assert_all_formats(stage_dir / "model2_sample000_model2_phi_waveform")

    proc = run_cli(
        [
            sys.executable,
            "scripts/make_slide_spacetime_assets.py",
            "--config",
            str(config_file),
            "--predictions",
            str(out_dir / "predictions.pt"),
            "--run-dir",
            str(out_dir),
            "--trajectory",
            "both",
            "--sample-index",
            "0",
            "--num-frames",
            "3",
            "--out-dir",
            str(spacetime_dir),
            "--device",
            "cpu",
            "--dtype",
            "float64",
        ],
        cwd=REPO_ROOT,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (spacetime_dir / "asset_manifest.json").exists()
    _assert_all_formats(spacetime_dir / "model2_sample000_target_burgers_spacetime")
    _assert_all_formats(spacetime_dir / "model2_sample000_reaction_diffusion_surrogate_spacetime")
