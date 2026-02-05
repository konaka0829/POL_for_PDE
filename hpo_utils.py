"""Utility helpers for RFM hyperparameter optimization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from rfm_core import predict_from_features
from utilities3 import LpLoss


def parse_csv_list(value: str, cast_type: type = int) -> list:
    """Parse a comma-separated list into a list of values."""
    if value is None or value == "":
        return []
    return [cast_type(item.strip()) for item in value.split(",") if item.strip() != ""]


def loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample from log-uniform [low, high]."""
    if low <= 0 or high <= 0:
        raise ValueError("loguniform bounds must be positive.")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_config(rng: np.random.Generator, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample a hyperparameter configuration from a search space."""
    config: dict[str, Any] = {}
    for key, spec in space.items():
        spec_type = spec.get("type")
        if spec_type == "uniform":
            config[key] = float(rng.uniform(spec["low"], spec["high"]))
        elif spec_type == "loguniform":
            config[key] = loguniform(rng, spec["low"], spec["high"])
        elif spec_type == "choice":
            choice = rng.choice(spec["choices"])
            if isinstance(choice, np.generic):
                choice = choice.item()
            config[key] = choice
        else:
            raise ValueError(f"Unsupported spec type '{spec_type}' for key '{key}'.")
    return config


def split_indices(
    n_total: int,
    val_split: float,
    seed: int,
    *,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train/val indices."""
    if not 0 < val_split < 1:
        raise ValueError("val_split must be in (0, 1).")
    indices = np.arange(n_total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    split_idx = int(n_total * (1 - val_split))
    return indices[:split_idx], indices[split_idx:]


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader from tensors."""
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def eval_rel_l2(
    alpha: torch.Tensor,
    features: Any,
    loader: torch.utils.data.DataLoader,
    m: int,
    device: torch.device,
) -> float:
    """Compute mean relative L2 error for a loader."""
    loss = LpLoss(size_average=False)
    total = 0.0
    count = 0
    with torch.no_grad():
        for a, y in loader:
            a = a.to(device)
            y = y.to(device)
            phi = features(a)
            if phi.ndim > 3:
                phi = phi.reshape(phi.shape[0], phi.shape[1], -1)
            pred = predict_from_features(alpha, phi, m)
            pred = pred.reshape(y.shape)
            total += loss(pred.reshape(pred.shape[0], -1), y.reshape(y.shape[0], -1)).item()
            count += y.shape[0]
    return total / count


@dataclass
class TrialResult:
    """Container for trial results."""

    trial: int
    config: dict[str, Any]
    val_mean: float
    val_std: float
    test_mean: float | None
    test_std: float | None
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_results(path: str, results: Sequence[TrialResult], best: TrialResult | None) -> None:
    """Save HPO results to JSON or CSV."""
    payload = {
        "results": [res.to_dict() for res in results],
        "best": best.to_dict() if best else None,
    }
    if path.endswith(".csv"):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0].keys()) if results else [])
            if results:
                writer.writeheader()
                for res in payload["results"]:
                    writer.writerow(res)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
