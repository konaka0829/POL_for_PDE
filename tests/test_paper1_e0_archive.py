from dataclasses import replace
import subprocess
import sys

import pytest
import torch

from pol.paper1.config import load_config_json, save_config_json
from pol.paper1.e0 import load_master_initial_conditions, save_master_initial_conditions
from pol.paper1.initial_conditions import build_master_grf_initial_conditions, initial_conditions_at_resolution


def _archive(tmp_path):
    cfg = load_config_json("configs/paper1_e0_smoke.json")
    master = build_master_grf_initial_conditions(cfg)
    path = tmp_path / "master.pt"
    save_master_initial_conditions(master, path, tmp_path / "manifest.json", cfg)
    return cfg, master, path


def test_archive_load_and_lower_resolution_fourier_preservation(tmp_path):
    cfg, master, path = _archive(tmp_path)
    loaded = load_master_initial_conditions(path, cfg)
    assert torch.equal(loaded.values_master, master.values_master)
    lower = replace(cfg, e0=None, spatial=replace(cfg.spatial, reference_nx=32, target_data_nx=32))
    reused = load_master_initial_conditions(path, lower)
    down = initial_conditions_at_resolution(reused, 32)
    original_hat = torch.fft.fft(master.values_master, norm="forward")
    down_hat = torch.fft.fft(down, norm="forward")
    assert torch.allclose(down_hat[:, 1:16], original_hat[:, 1:16], atol=1e-12)


@pytest.mark.parametrize("mutation,match", [
    ("hash", "tensor hash"), ("shape", "shape"), ("dtype", "dtype"),
    ("metadata_dtype", "dtype"), ("nan", "finite"), ("ids", "sample IDs"),
])
def test_archive_rejects_tampering(tmp_path, mutation, match):
    cfg, _, path = _archive(tmp_path)
    payload = torch.load(path, weights_only=False)
    if mutation == "hash": payload["values"][0, 0] += 1
    elif mutation == "shape": payload["values"] = payload["values"][:, :-1]
    elif mutation == "dtype": payload["values"] = payload["values"].float()
    elif mutation == "metadata_dtype": payload["metadata"]["dtype"] = "float32"
    elif mutation == "nan": payload["values"][0, 0] = float("nan")
    elif mutation == "ids": payload["sample_ids"][0] = 7
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match): load_master_initial_conditions(path, cfg)


def test_generate_dataset_cli_archive_handoff(tmp_path):
    cfg, _, archive = _archive(tmp_path)
    config_path = tmp_path / "production.json"; save_config_json(replace(cfg, e0=None), config_path)
    out = tmp_path / "dataset"
    proc = subprocess.run([sys.executable, "tests/paper1_recipe_driver.py", "dataset", "--config", str(config_path), "--master-initial-conditions", str(archive), "--output-dir", str(out), "--no-target", "--overwrite"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "master_dataset.pt").exists()
