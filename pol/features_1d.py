from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch

from pol.time_grid import require_time_aligned


def build_time_grid(
    *,
    Tr: float,
    dt: float,
    K: int,
    feature_times: str,
) -> tuple[List[float], List[int]]:
    if Tr <= 0.0 or dt <= 0.0:
        raise ValueError("Tr and dt must be positive")
    step_T = require_time_aligned(Tr, dt, "Tr")
    if step_T <= 0:
        raise ValueError("Tr must be positive")

    if feature_times.strip():
        times = [float(v.strip()) for v in feature_times.split(",") if v.strip()]
        if not times:
            raise ValueError("feature-times is empty")
        steps = [require_time_aligned(t, dt, f"feature_times[{idx}]") for idx, t in enumerate(times)]
    else:
        if K <= 0:
            raise ValueError("K must be positive when feature-times is not provided")
        if K > step_T:
            raise ValueError("K cannot exceed the number of positive dt steps in Tr")
        if K == 1:
            steps = [step_T]
        else:
            steps = np.linspace(0, step_T, num=K + 1)[1:]
            steps = [int(round(v)) for v in steps.tolist()]
            steps = sorted(set(steps))
        times = [step * dt for step in steps]

    for t, step in zip(times, steps):
        if t <= 0.0 or t > Tr + 1e-12:
            raise ValueError(f"Feature time {t} must be in (0, Tr]")
        if step <= 0:
            raise ValueError(f"Feature time {t} must be positive")

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

    if obs == "fourier":
        max_modes = s // 2 + 1
        if J <= 0 or J > max_modes:
            raise ValueError(f"J must be in [1, {max_modes}] for fourier observation")
        return torch.arange(J, dtype=torch.long)

    if obs == "proj":
        if J <= 0:
            raise ValueError("J must be positive for proj observation")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(sensor_seed)
        scale = 1.0 / np.sqrt(float(s))
        return scale * torch.randn((J, s), generator=gen, dtype=torch.float32)

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
        elif obs == "fourier":
            modes = sensor_idx.to(z.device)
            z_hat = torch.fft.rfft(z, dim=-1)
            sel = z_hat.index_select(dim=-1, index=modes)
            obs_list.append(torch.cat([sel.real, sel.imag], dim=-1))
        elif obs == "proj":
            proj = sensor_idx.to(device=z.device, dtype=z.dtype)
            obs_list.append(z @ proj.t())
        else:
            raise ValueError(f"Unsupported observation type: {obs}")
    return obs_list


def flatten_observations(obs_list: Sequence[torch.Tensor]) -> torch.Tensor:
    if not obs_list:
        raise ValueError("obs_list is empty")
    return torch.cat([o.reshape(o.shape[0], -1) for o in obs_list], dim=-1)


def standardize_features(phi: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (phi - mean) / (std + eps)
