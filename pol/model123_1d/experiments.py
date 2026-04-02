from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch

from pol.features_1d import build_time_grid

from .datasets import DatasetConfig, build_dataset, save_dataset_bundle
from .metrics import rms_l2
from .models import fit_model2, fit_model3
from .observations import build_observation_operator, decode_model1_observation, make_model2_features, observe_states
from .solvers import SurrogateSpec, simulate_surrogate_batches


@dataclass(frozen=True)
class ExperimentConfig:
    total_samples: int = 1200
    ntrain: int = 1000
    ntest: int = 200
    seed: int = 0
    nx: int = 256
    target_nu: float = 0.05
    T: float = 1.0
    dt: float = 1e-3
    fine_dt: float = 1e-4
    batch_size: int = 20
    obs: str = "full"
    J: int = 33
    K: int = 1
    feature_times: str = ""
    reservoir: str = "burgers"
    rd_nu: float = 1e-3
    rd_alpha: float = 1.0
    rd_beta: float = 1.0
    burgers_nu: float = 0.05
    burgers_b: float = 1.0
    ks_dealias: bool = False
    ks_b: float = 1.0
    ks_eta: float = 1.0
    ks_kappa: float = 1.0
    model3_m: int = 256
    model3_activation: str = "tanh"
    model3_seed: int = 0
    model3_weight_scale: float = 0.0
    model3_bias_scale: float = 1.0
    out_dir: str = "visualizations/model123_1d"
    save_dataset: bool = False
    device: str = "cpu"
    dtype: str = "float64"

    def dataset_config(self) -> DatasetConfig:
        return DatasetConfig(
            total_samples=self.total_samples,
            ntrain=self.ntrain,
            ntest=self.ntest,
            seed=self.seed,
            nx=self.nx,
            target_nu=self.target_nu,
            T=self.T,
            dt=self.dt,
            fine_dt=self.fine_dt,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

    def surrogate_spec(self) -> SurrogateSpec:
        return SurrogateSpec(
            family=self.reservoir,
            dt=self.dt,
            T=self.T,
            fine_dt=self.fine_dt,
            rd_nu=self.rd_nu,
            rd_alpha=self.rd_alpha,
            rd_beta=self.rd_beta,
            burgers_nu=self.burgers_nu,
            burgers_b=self.burgers_b,
            ks_dealias=self.ks_dealias,
            ks_b=self.ks_b,
            ks_eta=self.ks_eta,
            ks_kappa=self.ks_kappa,
        )


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def run_experiment(cfg: ExperimentConfig) -> dict[str, float | str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = _resolve_device(cfg.device)
    dataset = build_dataset(cfg.dataset_config(), device=device)
    if cfg.save_dataset:
        save_dataset_bundle(dataset, os.path.join(cfg.out_dir, "dataset.pt"))

    _, obs_steps = build_time_grid(Tr=cfg.T, dt=cfg.dt, K=cfg.K, feature_times=cfg.feature_times)
    operator = build_observation_operator(obs=cfg.obs, nx=cfg.nx, J=cfg.J, sensor_seed=cfg.seed)
    spec = cfg.surrogate_spec()

    states_train = simulate_surrogate_batches(
        dataset.u0_train.to(device=device, dtype=dataset.y_train.dtype),
        spec=spec,
        obs_steps=obs_steps,
        batch_size=cfg.batch_size,
    )
    states_test = simulate_surrogate_batches(
        dataset.u0_test.to(device=device, dtype=dataset.y_test.dtype),
        spec=spec,
        obs_steps=obs_steps,
        batch_size=cfg.batch_size,
    )

    model1_obs_train = observe_states([states_train[-1]], obs=cfg.obs, operator=operator)[0]
    model1_obs_test = observe_states([states_test[-1]], obs=cfg.obs, operator=operator)[0]
    model1_train = decode_model1_observation(model1_obs_train, obs=cfg.obs, nx=cfg.nx, J=cfg.J)
    model1_test = decode_model1_observation(model1_obs_test, obs=cfg.obs, nx=cfg.nx, J=cfg.J)

    model2_features_train = make_model2_features(states_train, obs=cfg.obs, operator=operator)
    model2_features_test = make_model2_features(states_test, obs=cfg.obs, operator=operator)

    targets_train = dataset.y_train.to(dtype=model2_features_train.dtype)
    targets_test = dataset.y_test.to(dtype=model2_features_test.dtype)

    _, model2_train, model2_test = fit_model2(model2_features_train, targets_train, model2_features_test)
    _, _, model3_train, model3_test = fit_model3(
        model2_features_train,
        targets_train,
        model2_features_test,
        hidden_dim=cfg.model3_m,
        activation=cfg.model3_activation,
        seed=cfg.model3_seed,
        weight_scale=cfg.model3_weight_scale,
        bias_scale=cfg.model3_bias_scale,
    )

    metrics = {
        "reservoir": cfg.reservoir,
        "obs": cfg.obs,
        "E1_train": rms_l2(model1_train, targets_train),
        "E1_test": rms_l2(model1_test, targets_test),
        "E2_train": rms_l2(model2_train, targets_train),
        "E2_test": rms_l2(model2_test, targets_test),
        "E3_train": rms_l2(model3_train, targets_train),
        "E3_test": rms_l2(model3_test, targets_test),
    }

    with open(os.path.join(cfg.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "metrics": metrics}, f, indent=2, sort_keys=True)
    return metrics
