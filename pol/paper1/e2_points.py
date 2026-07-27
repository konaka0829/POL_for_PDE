"""Typed dataset boundaries for E2 point and cache evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectionDatasetView:
    """Only tensors selection and convergence are permitted to inspect."""

    sample_ids: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    u0_train_validation: torch.Tensor
    target_train: torch.Tensor
    target_validation: torch.Tensor
    target_data_validation: torch.Tensor
    reference_validation: torch.Tensor


@dataclass(frozen=True)
class TestDatasetView:
    """Test-only tensors constructed after durable frozen-plan read-back."""

    sample_ids: torch.Tensor
    indices: torch.Tensor
    u0_test: torch.Tensor
    target_coefficients_test: torch.Tensor
    target_data_test: torch.Tensor
    reference_test: torch.Tensor


@dataclass(frozen=True)
class PointEvaluationResult:
    """Typed result of one physical surrogate point evaluation."""

    state_cache_key: str
    feature_cache_key: str
    solver_metadata: dict[str, object]
