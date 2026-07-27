import json
import subprocess
import sys

import pytest
import torch

from pol.paper1.config import load_config_json
from pol.paper1.datasets import _tensor_hash
from pol.paper1.datasets import build_master_dataset, load_master_dataset, save_master_dataset
from pol.paper1.schemas import stable_hash_json


def test_split_disjointness_and_coverage():
    cfg = load_config_json("configs/paper1_smoke.json")
    ds = build_master_dataset(cfg, generate_target=False)
    assigned = torch.cat([ds.train_indices, ds.val_indices, ds.test_indices])
    assert assigned.numel() == cfg.data.total_samples
    assert torch.unique(assigned).numel() == cfg.data.total_samples
    assert set(assigned.tolist()) == set(range(cfg.data.total_samples))


def test_save_load_round_trip_and_hashes(tmp_path):
    cfg = load_config_json("configs/paper1_smoke.json")
    ds = build_master_dataset(cfg, generate_target=False)
    save_master_dataset(ds, tmp_path, overwrite=True)
    got = load_master_dataset(tmp_path)
    assert got.config == cfg
    assert torch.equal(got.sample_ids, ds.sample_ids)
    assert torch.allclose(got.u0_master, ds.u0_master)
    assert got.metadata["dataset_hash"] == ds.metadata["dataset_hash"]
    assert stable_hash_json(got.metadata["split"]) == got.metadata["split_hash"]


def test_tampered_manifest_hash_detection(tmp_path):
    cfg = load_config_json("configs/paper1_smoke.json")
    ds = build_master_dataset(cfg, generate_target=False)
    save_master_dataset(ds, tmp_path, overwrite=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["tensor_hashes"]["u0_master"] = "bad"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tensor hash"):
        load_master_dataset(tmp_path)


def test_tampered_split_hash_detection(tmp_path):
    cfg = load_config_json("configs/paper1_smoke.json")
    ds = build_master_dataset(cfg, generate_target=False)
    save_master_dataset(ds, tmp_path, overwrite=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["split"]["permutation"] = list(reversed(manifest["split"]["permutation"]))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split_hash"):
        load_master_dataset(tmp_path)


def test_tampered_tensor_shape_detection(tmp_path):
    cfg = load_config_json("configs/paper1_smoke.json")
    ds = build_master_dataset(cfg, generate_target=False)
    save_master_dataset(ds, tmp_path, overwrite=True)
    payload = torch.load(tmp_path / "master_dataset.pt", map_location="cpu", weights_only=False)
    payload["u0_master"] = payload["u0_master"][:, :-1]
    bad_hash = _tensor_hash(payload["u0_master"])
    payload["metadata"]["tensor_hashes"]["u0_master"] = bad_hash
    torch.save(payload, tmp_path / "master_dataset.pt")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["tensor_hashes"]["u0_master"] = bad_hash
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="u0_master shape"):
        load_master_dataset(tmp_path)


def test_tiny_target_generation_shape():
    cfg = load_config_json("configs/paper1_smoke.json")
    ds = build_master_dataset(cfg, generate_target=True)
    assert ds.y_target_master is not None
    assert ds.y_target_master.shape == (cfg.data.total_samples, cfg.spatial.target_master_nx)


def test_smoke_cli_creates_required_files(tmp_path):
    out = tmp_path / "paper1_cli"
    proc = subprocess.run(
        [
            sys.executable,
            "tests/paper1_recipe_driver.py", "dataset",
            "--config",
            "configs/paper1_smoke.json",
            "--output-dir",
            str(out),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (out / "master_dataset.pt").exists()
    assert (out / "manifest.json").exists()
    assert (out / "resolved_config.json").exists()
    loaded = load_master_dataset(out)
    assert loaded.u0_master.shape == (12, 64)
    assert loaded.y_target_master is not None
    assert loaded.metadata["runtime"]["torch_version"]


def test_cli_refuses_overwrite_before_generation(tmp_path):
    out = tmp_path / "paper1_cli_existing"
    out.mkdir()
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "tests/paper1_recipe_driver.py", "dataset",
            "--config",
            "configs/paper1_smoke.json",
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--overwrite" in proc.stderr
    assert not (out / "master_dataset.pt").exists()
