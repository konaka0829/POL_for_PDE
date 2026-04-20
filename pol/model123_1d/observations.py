from __future__ import annotations

import warnings

import torch

from pol.features_1d import build_sensor_indices, collect_observations, flatten_observations


warnings.warn(
    "pol.model123_1d.observations is deprecated; use predictors.py + features_1d.py",
    DeprecationWarning,
    stacklevel=2,
)


def build_observation_operator(
    *,
    obs: str,
    nx: int,
    J: int,
    sensor_seed: int,
) -> torch.Tensor:
    return build_sensor_indices(
        s=nx,
        obs=obs,
        J=J,
        sensor_mode="equispaced",
        sensor_seed=sensor_seed,
    )


def observe_states(
    states: list[torch.Tensor],
    *,
    obs: str,
    operator: torch.Tensor,
) -> list[torch.Tensor]:
    return collect_observations(states, obs=obs, sensor_idx=operator)


def make_model2_features(
    states: list[torch.Tensor],
    *,
    obs: str,
    operator: torch.Tensor,
) -> torch.Tensor:
    return flatten_observations(observe_states(states, obs=obs, operator=operator))


def decode_model1_observation(
    observation: torch.Tensor,
    *,
    obs: str,
    nx: int,
    J: int,
) -> torch.Tensor:
    if obs == "full":
        return observation
    if obs != "fourier":
        raise ValueError("Model 1 currently supports obs='full' or obs='fourier'")
    if observation.shape[-1] != 2 * J:
        raise ValueError(f"Expected Fourier observation width {2 * J}, got {observation.shape[-1]}")

    real = observation[:, :J]
    imag = observation[:, J:]
    imag = imag.clone()
    imag[:, 0] = 0.0
    z = torch.complex(real, imag)
    full_hat = torch.zeros((observation.shape[0], nx // 2 + 1), device=observation.device, dtype=z.dtype)
    full_hat[:, :J] = z
    return torch.fft.irfft(full_hat, n=nx, dim=-1)
