from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pol.numerics.burgers import simulate_burgers_split_step
from pol.numerics.etdrk4 import simulate_burgers_etdrk4_trajectory
from pol.numerics.initial_conditions import (
    sample_gaussian_random_field_initial_conditions,
)
from pol.paper1.datasets import tensor_hash


GOLDEN = json.loads(
    (
        Path(__file__).parent
        / "fixtures/phase6_neutral_numerics_golden_v1.json"
    ).read_text(encoding="utf-8")
)["hashes"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nx", [15, 16])
@pytest.mark.parametrize("dealias", [False, True])
def test_split_step_deterministic_shape_dtype(
    dtype: torch.dtype, nx: int, dealias: bool
) -> None:
    x = torch.arange(nx, dtype=dtype) / nx
    u0 = (0.2 * torch.sin(2 * torch.pi * x))[None]
    trajectory = simulate_burgers_split_step(
        u0,
        dt=0.01,
        Tr=0.03,
        obs_steps=[1, 3],
        nu=0.05,
        fine_dt=0.005,
        dealias=dealias,
    )
    assert [tuple(item.shape) for item in trajectory] == [(1, nx), (1, nx)]
    assert all(item.dtype == dtype for item in trajectory)
    repeated = simulate_burgers_split_step(
        u0,
        dt=0.01,
        Tr=0.03,
        obs_steps=[1, 3],
        nu=0.05,
        fine_dt=0.005,
        dealias=dealias,
    )
    assert all(torch.equal(a, b) for a, b in zip(trajectory, repeated))
    key = f"burgers_{str(dtype).removeprefix('torch.')}_nx{nx}_dealias{int(dealias)}"
    assert [tensor_hash(item) for item in trajectory] == GOLDEN[key]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nx", [15, 16])
def test_etdrk4_trajectory_deterministic_shape_dtype(
    dtype: torch.dtype, nx: int
) -> None:
    x = torch.arange(nx, dtype=dtype) / nx
    u0 = (0.1 * torch.cos(2 * torch.pi * x))[None]
    values = simulate_burgers_etdrk4_trajectory(
        u0, nu=0.05, T=0.02, dt=0.01, obs_steps=[1, 2]
    )
    assert [tuple(item.shape) for item in values] == [(1, nx), (1, nx)]
    assert all(item.dtype == dtype and torch.isfinite(item).all() for item in values)
    key = f"etdrk4_{str(dtype).removeprefix('torch.')}_nx{nx}"
    assert [tensor_hash(item) for item in values] == GOLDEN[key]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nx", [15, 16])
def test_grf_fixed_seed_even_odd_batch_and_parameters(
    dtype: torch.dtype, nx: int
) -> None:
    kwargs = dict(
        num_samples=3,
        nx=nx,
        seed=19,
        gamma=2.25,
        tau=4.0,
        sigma=7.0,
        mean=0.5,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    first = sample_gaussian_random_field_initial_conditions(**kwargs)
    second = sample_gaussian_random_field_initial_conditions(**kwargs)
    assert first.shape == (3, nx)
    assert first.dtype == dtype
    assert torch.equal(first, second)
    key = f"grf_{str(dtype).removeprefix('torch.')}_nx{nx}"
    assert tensor_hash(first) == GOLDEN[key]


@pytest.mark.parametrize(
    "call,error",
    [
        (
            lambda: simulate_burgers_split_step(
                torch.zeros(4),
                dt=0.1,
                Tr=0.1,
                obs_steps=[1],
                nu=0.1,
                fine_dt=0.01,
            ),
            ValueError,
        ),
        (
            lambda: sample_gaussian_random_field_initial_conditions(
                0, 16, seed=0, device=torch.device("cpu")
            ),
            ValueError,
        ),
    ],
)
def test_neutral_numerics_invalid_inputs(call, error) -> None:
    with pytest.raises(error):
        call()
