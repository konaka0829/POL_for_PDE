from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .config import Paper1Config, canonical_config_json, config_from_dict, save_config_json
from .initial_conditions import MasterInitialConditions, build_master_grf_initial_conditions, initial_conditions_at_resolution
from .schemas import SCHEMA_VERSION, dataset_schema_metadata, stable_hash_json
from .solvers import solve_burgers_final_state


@dataclass
class Paper1MasterDataset:
    config: Paper1Config
    sample_ids: torch.Tensor
    train_indices: torch.Tensor
    val_indices: torch.Tensor
    test_indices: torch.Tensor
    u0_master: torch.Tensor
    u0_hat_master: torch.Tensor
    y_target_master: torch.Tensor | None
    metadata: dict[str, object]


def _tensor_hash(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("ascii"))
    h.update(json.dumps(list(t.shape)).encode("ascii"))
    if t.is_complex():
        h.update(torch.view_as_real(t).numpy().tobytes())
    else:
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _split_indices(config: Paper1Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(config.data.seed)
    perm = torch.randperm(config.data.total_samples, generator=gen, dtype=torch.long)
    n_train = config.data.n_train
    n_val = config.data.n_val
    train = perm[:n_train].clone()
    val = perm[n_train : n_train + n_val].clone()
    test = perm[n_train + n_val :].clone()
    meta = {
        "split_seed": config.data.seed,
        "split_order": "torch.randperm",
        "permutation": perm.tolist(),
        "n_train": config.data.n_train,
        "n_val": config.data.n_val,
        "n_test": config.data.n_test,
    }
    return train, val, test, meta


def _simulate_target_master_batch(config: Paper1Config, u0: torch.Tensor) -> torch.Tensor:
    return solve_burgers_final_state(
        u0, nu=config.target.nu, T=config.target.T, dt=config.target.dt,
        fine_dt=config.target.fine_dt, solver=config.target.solver,
        dealias=config.target.dealias, domain_length=config.domain.length,
    ).values


def _simulate_target_master(config: Paper1Config, u0: torch.Tensor, *, batch_size: int = 20) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    chunks: list[torch.Tensor] = []
    for start in range(0, u0.shape[0], batch_size):
        batch = u0[start : start + batch_size]
        chunks.append(_simulate_target_master_batch(config, batch).detach().cpu())
    return torch.cat(chunks, dim=0).to(dtype=u0.dtype)


def _metadata(config: Paper1Config, split_meta: dict[str, object], tensor_hashes: dict[str, str], y_generated: bool) -> dict[str, object]:
    split_hash = stable_hash_json(split_meta)
    config_hash = hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()
    base = {
        **dataset_schema_metadata(),
        "config": config.to_dict(),
        "config_hash": config_hash,
        "split": split_meta,
        "split_hash": split_hash,
        "tensor_hashes": tensor_hashes,
        "dataset_hash": stable_hash_json(
            {
                "schema_version": SCHEMA_VERSION,
                "config_hash": config_hash,
                "split_hash": split_hash,
                "tensor_hashes": tensor_hashes,
                "y_target_master_generated": y_generated,
            }
        ),
        "domain_length": config.domain.length,
        "target": config.target.__dict__,
        "spatial": config.spatial.__dict__,
        "data": config.data.__dict__,
        "ic_coordinate_convention": "normalized_periodic_coordinate_x_over_L",
        "fft_normalization": "torch.fft.rfft(values, norm='forward')",
    }
    return base


def build_master_dataset(
    config: Paper1Config,
    *,
    generate_target: bool = True,
    master_initial_conditions: MasterInitialConditions | None = None,
) -> Paper1MasterDataset:
    config.validate()
    master = build_master_grf_initial_conditions(config) if master_initial_conditions is None else master_initial_conditions
    if master.sample_ids.numel() != config.data.total_samples or not torch.equal(master.sample_ids.cpu(), torch.arange(config.data.total_samples)):
        raise ValueError("master initial-condition sample IDs/count do not match config")
    if master.domain_length != config.domain.length or master.seed != config.data.seed:
        raise ValueError("master initial-condition domain length or seed does not match config")
    if master.master_nx < config.spatial.reference_nx:
        raise ValueError("master initial-condition maximum nx is smaller than reference_nx")
    if master.master_nx != config.spatial.reference_nx:
        values = initial_conditions_at_resolution(master, config.spatial.reference_nx)
        master = MasterInitialConditions(master.sample_ids, values, torch.fft.rfft(values, dim=-1, norm="forward"), config.spatial.reference_nx, master.domain_length, master.seed)
    train, val, test, split_meta = _split_indices(config)
    y = _simulate_target_master(config, master.values_master) if generate_target else None
    tensor_hashes = {
        "sample_ids": _tensor_hash(master.sample_ids),
        "train_indices": _tensor_hash(train),
        "val_indices": _tensor_hash(val),
        "test_indices": _tensor_hash(test),
        "u0_master": _tensor_hash(master.values_master),
        "u0_hat_master": _tensor_hash(master.fourier_master),
    }
    if y is not None:
        tensor_hashes["y_target_master"] = _tensor_hash(y)
    metadata = _metadata(config, split_meta, tensor_hashes, y is not None)
    return Paper1MasterDataset(
        config=config,
        sample_ids=master.sample_ids.detach().cpu(),
        train_indices=train,
        val_indices=val,
        test_indices=test,
        u0_master=master.values_master.detach().cpu(),
        u0_hat_master=master.fourier_master.detach().cpu(),
        y_target_master=None if y is None else y.detach().cpu(),
        metadata=metadata,
    )


def _payload(dataset: Paper1MasterDataset) -> dict[str, Any]:
    return {
        "sample_ids": dataset.sample_ids,
        "train_indices": dataset.train_indices,
        "val_indices": dataset.val_indices,
        "test_indices": dataset.test_indices,
        "u0_master": dataset.u0_master,
        "u0_hat_master": dataset.u0_hat_master,
        "y_target_master": dataset.y_target_master,
        "metadata": dataset.metadata,
    }


def save_master_dataset(dataset: Paper1MasterDataset, output_dir: str | Path, *, overwrite: bool = False) -> None:
    out = Path(output_dir)
    pt = out / "master_dataset.pt"
    manifest = out / "manifest.json"
    resolved = out / "resolved_config.json"
    if out.exists() and not overwrite and (pt.exists() or manifest.exists() or resolved.exists()):
        raise FileExistsError(f"{out} already contains paper1 dataset outputs; pass overwrite=True")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(_payload(dataset), pt)
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(dataset.metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    save_config_json(dataset.config, resolved)


def _verify_loaded(payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing schema_version")
    meta = payload.get("metadata")
    if meta is None:
        raise ValueError("payload metadata is missing")
    for key in ("dataset_hash", "split_hash", "config_hash"):
        if meta.get(key) != manifest.get(key):
            raise ValueError(f"manifest/payload {key} mismatch")
    cfg = config_from_dict(manifest["config"])
    config_hash = hashlib.sha256(canonical_config_json(cfg).encode("utf-8")).hexdigest()
    if manifest.get("config_hash") != config_hash:
        raise ValueError("config_hash does not match manifest config")
    split_meta = manifest.get("split", {})
    split_hash = stable_hash_json(split_meta)
    if manifest.get("split_hash") != split_hash:
        raise ValueError("split_hash does not match manifest split")
    tensor_hashes = manifest.get("tensor_hashes", {})
    for key, expected in tensor_hashes.items():
        if key not in payload or payload[key] is None:
            raise ValueError(f"payload tensor {key} is missing")
        if _tensor_hash(payload[key]) != expected:
            raise ValueError(f"tensor hash mismatch for {key}")
    n = cfg.data.total_samples
    if payload["sample_ids"].numel() != n or payload["u0_master"].shape[0] != n:
        raise ValueError("sample count mismatch")
    expected_u0_shape = (n, cfg.spatial.reference_nx)
    expected_hat_shape = (n, cfg.spatial.reference_nx // 2 + 1)
    if tuple(payload["u0_master"].shape) != expected_u0_shape:
        raise ValueError("u0_master shape mismatch")
    if tuple(payload["u0_hat_master"].shape) != expected_hat_shape:
        raise ValueError("u0_hat_master shape mismatch")
    y_target = payload.get("y_target_master")
    if y_target is not None and tuple(y_target.shape) != expected_u0_shape:
        raise ValueError("y_target_master shape mismatch")
    if payload["train_indices"].numel() != cfg.data.n_train:
        raise ValueError("train split length mismatch")
    if payload["val_indices"].numel() != cfg.data.n_val:
        raise ValueError("val split length mismatch")
    if payload["test_indices"].numel() != cfg.data.n_test:
        raise ValueError("test split length mismatch")
    assigned = torch.cat([payload["train_indices"], payload["val_indices"], payload["test_indices"]], dim=0)
    if assigned.numel() != n or torch.unique(assigned).numel() != n:
        raise ValueError("split indices are not a disjoint full cover")
    if set(assigned.tolist()) != set(range(n)):
        raise ValueError("split indices do not cover sample IDs exactly")
    expected_dataset_hash = stable_hash_json(
        {
            "schema_version": SCHEMA_VERSION,
            "config_hash": config_hash,
            "split_hash": split_hash,
            "tensor_hashes": tensor_hashes,
            "y_target_master_generated": payload.get("y_target_master") is not None,
        }
    )
    if manifest.get("dataset_hash") != expected_dataset_hash:
        raise ValueError("dataset_hash does not match manifest contents")


def load_master_dataset(output_dir: str | Path) -> Paper1MasterDataset:
    out = Path(output_dir)
    payload = torch.load(out / "master_dataset.pt", map_location="cpu", weights_only=False)
    with (out / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    _verify_loaded(payload, manifest)
    config = config_from_dict(manifest["config"])
    return Paper1MasterDataset(
        config=config,
        sample_ids=payload["sample_ids"],
        train_indices=payload["train_indices"],
        val_indices=payload["val_indices"],
        test_indices=payload["test_indices"],
        u0_master=payload["u0_master"],
        u0_hat_master=payload["u0_hat_master"],
        y_target_master=payload.get("y_target_master"),
        metadata=manifest,
    )
