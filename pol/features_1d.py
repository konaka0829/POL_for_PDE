from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch


def build_time_grid(
    *,
    Tr: float,
    dt: float,
    K: int,
    feature_times: str,
) -> tuple[List[float], List[int]]:
    if Tr <= 0.0 or dt <= 0.0:
        raise ValueError("Tr and dt must be positive")

    if feature_times.strip():
        times = [float(v.strip()) for v in feature_times.split(",") if v.strip()]
        if not times:
            raise ValueError("feature-times is empty")
    else:
        if K <= 0:
            raise ValueError("K must be positive when feature-times is not provided")
        times = np.linspace(dt, Tr, num=K).tolist()

    for t in times:
        if t <= 0.0 or t > Tr + 1e-12:
            raise ValueError(f"Feature time {t} must be in (0, Tr]")

    steps = [max(1, int(round(t / dt))) for t in times]
    # Keep unique steps in ascending order while preserving matching times at those steps.
    step_to_time = {}
    for t, s in zip(times, steps):
        step_to_time[s] = min(step_to_time.get(s, t), t)

    steps_sorted = sorted(step_to_time.keys())
    times_sorted = [step_to_time[s] for s in steps_sorted]
    return times_sorted, steps_sorted


def build_sensor_indices(
    s: int,
    obs: str,
    J: int,
    sensor_mode: str,
    sensor_seed: int,
) -> torch.Tensor:
    if obs == "full":
        return torch.arange(s, dtype=torch.long)

    if J <= 0 or J > s:
        raise ValueError(f"J must be in [1, {s}] for points observation")

    if sensor_mode == "equispaced":
        idx = np.linspace(0, s - 1, num=J, dtype=int)
    elif sensor_mode == "random":
        rng = np.random.default_rng(sensor_seed)
        idx = np.sort(rng.choice(s, size=J, replace=False))
    else:
        raise ValueError(f"Unsupported sensor mode: {sensor_mode}")

    return torch.from_numpy(idx.astype(np.int64))


def collect_observations(states: Sequence[torch.Tensor], obs: str, sensor_idx: torch.Tensor) -> List[torch.Tensor]:
    obs_list: List[torch.Tensor] = []
    for z in states:
        if obs == "full":
            obs_list.append(z)
        elif obs == "points":
            idx = sensor_idx.to(z.device)
            obs_list.append(z.index_select(dim=-1, index=idx))
        else:
            raise ValueError(f"Unsupported observation type: {obs}")
    return obs_list


def flatten_observations(obs_list: Sequence[torch.Tensor]) -> torch.Tensor:
    if not obs_list:
        raise ValueError("obs_list is empty")
    return torch.cat([o.reshape(o.shape[0], -1) for o in obs_list], dim=-1)


def standardize_features(phi: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (phi - mean) / (std + eps)
