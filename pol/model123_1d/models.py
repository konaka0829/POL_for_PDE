from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from pol.elm import FixedRandomELM
from pol.ridge import fit_ridge_streaming, predict_linear


@dataclass
class AffineModel:
    weight: torch.Tensor

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return predict_linear(features, self.weight)


@dataclass
class ModelOutputs:
    model1_train: torch.Tensor
    model1_test: torch.Tensor
    model2_train: torch.Tensor
    model2_test: torch.Tensor
    model3_train: torch.Tensor
    model3_test: torch.Tensor


def _fit_affine_model(features: torch.Tensor, targets: torch.Tensor) -> AffineModel:
    warnings.warn(
        "pol.model123_1d.models is deprecated; use Model2Regressor1D/Model3Regressor1D in predictors.py",
        DeprecationWarning,
        stacklevel=2,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, targets),
        batch_size=max(1, int(features.shape[0])),
        shuffle=False,
    )
    state = fit_ridge_streaming(loader, lambda xb: xb, 0.0, dtype=features.dtype, regularize_bias=False)
    return AffineModel(weight=state["W"])


def fit_model2(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[AffineModel, torch.Tensor, torch.Tensor]:
    model = _fit_affine_model(train_features, train_targets)
    return model, model.predict(train_features), model.predict(test_features)


def fit_model3(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    *,
    hidden_dim: int,
    activation: str,
    seed: int,
    weight_scale: float,
    bias_scale: float,
) -> tuple[AffineModel, FixedRandomELM, torch.Tensor, torch.Tensor]:
    warnings.warn(
        "pol.model123_1d.models.fit_model3 is deprecated; use Model3Regressor1D in predictors.py",
        DeprecationWarning,
        stacklevel=2,
    )
    elm = FixedRandomELM(
        in_dim=train_features.shape[1],
        hidden_dim=hidden_dim,
        activation=activation,
        seed=seed,
        weight_scale=weight_scale,
        bias_scale=bias_scale,
        device=train_features.device,
        dtype=train_features.dtype,
    )
    train_lift = elm(train_features)
    test_lift = elm(test_features)
    train_aug = torch.cat([train_features, train_lift], dim=-1)
    test_aug = torch.cat([test_features, test_lift], dim=-1)
    model = _fit_affine_model(train_aug, train_targets)
    return model, elm, model.predict(train_aug), model.predict(test_aug)
