from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch

from .datasets import DatasetConfig, build_dataset, save_dataset_bundle
from .metrics import dataset_abs_l2h_error, dataset_rel_l2h_mean
from .predictors import Model1Predictor1D, Model2Regressor1D, Model3Regressor1D, Model123Config


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


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.float64


def build_model123_config_from_experiment(cfg: ExperimentConfig) -> Model123Config:
    # Keep synthetic path exact against DatasetConfig target builder.
    burgers_scheme = "split_step" if cfg.reservoir == "burgers" else "semi_implicit"
    return Model123Config(
        reservoir=cfg.reservoir,
        Ttilde=cfg.T,
        dt=cfg.dt,
        K=cfg.K,
        feature_times=cfg.feature_times,
        obs=cfg.obs,
        J=cfg.J,
        sensor_mode="equispaced",
        sensor_seed=cfg.seed,
        ridge_lambda=0.0,
        ridge_dtype=_resolve_dtype(cfg.dtype),
        elm_hidden_dim=cfg.model3_m,
        elm_activation=cfg.model3_activation,
        elm_seed=cfg.model3_seed,
        elm_weight_scale=cfg.model3_weight_scale,
        elm_bias_scale=cfg.model3_bias_scale,
        rd_nu=cfg.rd_nu,
        rd_alpha=cfg.rd_alpha,
        rd_beta=cfg.rd_beta,
        res_burgers_nu=cfg.burgers_nu,
        res_burgers_b=cfg.burgers_b,
        ks_dealias=cfg.ks_dealias,
        ks_b=cfg.ks_b,
        ks_eta=cfg.ks_eta,
        ks_kappa=cfg.ks_kappa,
        burgers_scheme=burgers_scheme,
        burgers_fine_dt=cfg.fine_dt,
        burgers_dealias=False,
        device=str(cfg.device),
        dtype=_resolve_dtype(cfg.dtype),
    )


def _jsonable_model_config(model_cfg: Model123Config) -> dict[str, object]:
    payload = asdict(model_cfg)
    payload["dtype"] = str(model_cfg.dtype)
    payload["ridge_dtype"] = str(model_cfg.ridge_dtype)
    if isinstance(payload.get("device"), torch.device):
        payload["device"] = str(payload["device"])
    return payload


@torch.no_grad()
def _predict_all(model, loader) -> torch.Tensor:
    preds = []
    for xb, _ in loader:
        preds.append(model.predict(xb).detach().cpu())
    return torch.cat(preds, dim=0)


def run_experiment(cfg: ExperimentConfig) -> dict[str, float | str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = _resolve_device(cfg.device)

    dataset = build_dataset(cfg.dataset_config(), device=device)
    if cfg.save_dataset:
        save_dataset_bundle(dataset, os.path.join(cfg.out_dir, "dataset.pt"))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset.u0_train, dataset.y_train),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset.u0_test, dataset.y_test),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model_cfg = build_model123_config_from_experiment(cfg)
    model1 = Model1Predictor1D(s=cfg.nx, config=model_cfg)
    model2 = Model2Regressor1D(s=cfg.nx, config=model_cfg)
    model3 = Model3Regressor1D(s=cfg.nx, config=model_cfg)

    model2.fit(train_loader)
    model3.fit(train_loader)

    model1_train = _predict_all(model1, train_loader)
    model1_test = _predict_all(model1, test_loader)
    model2_train = _predict_all(model2, train_loader)
    model2_test = _predict_all(model2, test_loader)
    model3_train = _predict_all(model3, train_loader)
    model3_test = _predict_all(model3, test_loader)

    targets_train = dataset.y_train.to(dtype=model1_train.dtype)
    targets_test = dataset.y_test.to(dtype=model1_test.dtype)

    metrics = {
        "reservoir": cfg.reservoir,
        "obs": cfg.obs,
        "main_metric": "abs_l2h",
        "E1_train": dataset_abs_l2h_error(model1_train, targets_train),
        "E1_test": dataset_abs_l2h_error(model1_test, targets_test),
        "E2_train": dataset_abs_l2h_error(model2_train, targets_train),
        "E2_test": dataset_abs_l2h_error(model2_test, targets_test),
        "E3_train": dataset_abs_l2h_error(model3_train, targets_train),
        "E3_test": dataset_abs_l2h_error(model3_test, targets_test),
        "E1_train_abs_l2h": dataset_abs_l2h_error(model1_train, targets_train),
        "E1_test_abs_l2h": dataset_abs_l2h_error(model1_test, targets_test),
        "E2_train_abs_l2h": dataset_abs_l2h_error(model2_train, targets_train),
        "E2_test_abs_l2h": dataset_abs_l2h_error(model2_test, targets_test),
        "E3_train_abs_l2h": dataset_abs_l2h_error(model3_train, targets_train),
        "E3_test_abs_l2h": dataset_abs_l2h_error(model3_test, targets_test),
        "E1_train_rel_l2h_mean": dataset_rel_l2h_mean(model1_train, targets_train),
        "E1_test_rel_l2h_mean": dataset_rel_l2h_mean(model1_test, targets_test),
        "E2_train_rel_l2h_mean": dataset_rel_l2h_mean(model2_train, targets_train),
        "E2_test_rel_l2h_mean": dataset_rel_l2h_mean(model2_test, targets_test),
        "E3_train_rel_l2h_mean": dataset_rel_l2h_mean(model3_train, targets_train),
        "E3_test_rel_l2h_mean": dataset_rel_l2h_mean(model3_test, targets_test),
    }

    with open(os.path.join(cfg.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "model_config": _jsonable_model_config(model_cfg),
                "metrics": metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return metrics
