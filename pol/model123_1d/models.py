from __future__ import annotations

from dataclasses import dataclass

import torch

from pol.elm import FixedRandomELM


def _append_bias(features: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((features.shape[0], 1), device=features.device, dtype=features.dtype)
    return torch.cat([features, ones], dim=-1)


@dataclass
class AffineModel:
    weight: torch.Tensor

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return _append_bias(features) @ self.weight


def fit_affine_model(features: torch.Tensor, targets: torch.Tensor) -> AffineModel:
    weight = torch.linalg.pinv(_append_bias(features)) @ targets
    return AffineModel(weight=weight)


@dataclass
class ModelOutputs:
    model1_train: torch.Tensor
    model1_test: torch.Tensor
    model2_train: torch.Tensor
    model2_test: torch.Tensor
    model3_train: torch.Tensor
    model3_test: torch.Tensor


def fit_model2(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[AffineModel, torch.Tensor, torch.Tensor]:
    model = fit_affine_model(train_features, train_targets)
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
    model = fit_affine_model(train_aug, train_targets)
    return model, elm, model.predict(train_aug), model.predict(test_aug)
