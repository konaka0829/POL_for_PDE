"""Typed dataset boundaries for E2 point and cache evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from .e2_cache import tensor_hash
from .metrics import samplewise_l2_errors
from .model1 import decode_equispaced_point_observation_to_real_fourier
from .target_representation import real_fourier_synthesis


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


@dataclass
class ValidationPointContext:
    """Mutable attempt-local state owned by validation point evaluation."""

    config: Any
    view: SelectionDatasetView
    finite_u0: torch.Tensor
    sample_positions: torch.Tensor
    pilot_n_sur: int
    batch_size: int | None
    cache: Any
    solve_state: Callable[..., tuple[torch.Tensor, dict[str, Any], str]]
    build_features: Callable[..., tuple[torch.Tensor, str]]
    fit_point: Callable[..., tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]
    metric_row: Callable[..., dict[str, float]]
    point_models: dict[tuple[str, str, float, float], dict[str, Any]]
    point_selections: dict[tuple[str, str, float, float], dict[str, Any]]
    physical_models: dict[tuple[Any, ...], tuple[Any, Any, Any]]
    validation_rows: list[dict[str, Any]]
    model3_validation: list[dict[str, Any]]
    point_order: list[tuple[str, str, float, float]]
    failed_runs: list[dict[str, Any]]


def evaluate_validation_point(
    context: ValidationPointContext,
    family: str,
    axis: str,
    nu: float,
    time_value: float,
) -> None:
    """Solve/cache/fit/score one validation-only physical candidate."""
    config, view = context.config, context.view
    identity = (family, axis, float(nu), float(time_value))
    if identity in context.point_models:
        return
    try:
        physical_key = (
            family,
            float(nu),
            float(time_value),
            context.pilot_n_sur,
            tensor_hash(context.sample_positions.detach().cpu()),
        )
        if physical_key not in context.physical_models:
            state, solver_meta, state_digest = context.solve_state(
                family, nu, time_value
            )
            features, feature_digest = context.build_features(
                state, state_digest
            )
            features = features.to(view.u0_train_validation.device)
            train_validation = type("TrainValidationData", (), {})()
            train_validation.x_train = features[view.train_indices]
            train_validation.x_validation = features[view.validation_indices]
            train_validation.y_train = view.target_train
            train_validation.y_validation = view.target_validation
            train_validation.target_data_validation = (
                view.target_data_validation
            )
            train_validation.reference_validation = view.reference_validation
            models, selections, provenance = context.fit_point(
                train_validation
            )
            bundle = {
                "models": models,
                "features": features,
                "solver_metadata": solver_meta,
                "state_digest": state_digest,
                "feature_digest": feature_digest,
                "physical_key": physical_key,
                "point_result": PointEvaluationResult(
                    state_cache_key=state_digest,
                    feature_cache_key=feature_digest,
                    solver_metadata=solver_meta,
                ),
            }
            context.physical_models[physical_key] = (
                bundle,
                selections,
                provenance,
            )
        bundle, selections, provenance = context.physical_models[physical_key]
        context.point_models[identity] = bundle
        context.point_selections[identity] = selections
        context.model3_validation.extend(
            [
                {
                    "family": family,
                    "sweep_axis": axis,
                    "nu_tilde": nu,
                    "T_tilde": time_value,
                    **row,
                }
                for row in provenance
            ]
        )
        features, models = bundle["features"], bundle["models"]
        validation = view.validation_indices
        for model_name in ("model1", "model2", "model3"):
            if model_name == "model1":
                prediction = decode_equispaced_point_observation_to_real_fourier(
                    features[validation],
                    config.spatial.target_output_dim,
                    domain_length=config.domain.length,
                )
            elif model_name == "model2":
                prediction = models["model2"](features[validation])
            else:
                predictions = [
                    readout(random_map(features[validation]))
                    for random_map, readout, _ in models["model3"][
                        "seed_models"
                    ].values()
                ]
                prediction = torch.stack(predictions).mean(0)
            metrics = context.metric_row(
                prediction,
                view.target_validation,
                view.reference_validation,
                view.target_data_validation,
            )
            if model_name == "model3":
                ensemble = {
                    f"ensemble_prediction_{key}": value
                    for key, value in metrics.items()
                }
                formal_values = [
                    float(
                        samplewise_l2_errors(
                            real_fourier_synthesis(
                                readout(random_map(features[validation])),
                                config.spatial.reference_nx,
                                domain_length=config.domain.length,
                            ),
                            view.reference_validation,
                            domain_length=config.domain.length,
                        )["relative"].mean()
                    )
                    for random_map, readout, _ in models["model3"][
                        "seed_models"
                    ].values()
                ]
                formal = torch.tensor(formal_values, dtype=torch.float64)
                metrics = {
                    "formal_seed_metric_mean": float(formal.mean()),
                    "formal_seed_metric_std": (
                        float(formal.std(unbiased=True))
                        if len(formal_values) > 1
                        else 0.0
                    ),
                    "formal_seed_metric_max": float(formal.max()),
                    "formal_seed_count": len(formal_values),
                    **ensemble,
                }
            context.validation_rows.append(
                {
                    "family": family,
                    "sweep_axis": axis,
                    "parameter_value": nu
                    if axis == "nu_tilde"
                    else time_value,
                    "fixed_parameter": time_value
                    if axis == "nu_tilde"
                    else nu,
                    "nu_tilde": nu,
                    "T_tilde": time_value,
                    "model": model_name,
                    "validation_field_relative_l2_mean": (
                        metrics["formal_seed_metric_mean"]
                        if model_name == "model3"
                        else metrics["field_relative_l2_mean"]
                    ),
                    **metrics,
                    "selected_zeta": selections.get(model_name, {}).get(
                        "zeta"
                    ),
                    "state_cache_key": bundle["state_digest"],
                    "feature_cache_key": bundle["feature_digest"],
                }
            )
        context.point_order.append(identity)
    except Exception as exc:
        context.failed_runs.append(
            {
                "family": family,
                "sweep_axis": axis,
                "nu_tilde": nu,
                "T_tilde": time_value,
                "reason": str(exc),
            }
        )
        raise


def evaluate_validation_points(
    *,
    view: SelectionDatasetView,
    candidate_points: Iterable[tuple[str, str, float, float]],
    evaluate_one: Callable[[str, str, float, float], Any],
) -> tuple[Any, ...]:
    """Evaluate physical candidates with a selection-only data capability.

    The callback owns numerical kernels and cache objects, but this boundary
    makes it impossible to pass a test view or full dataset into point
    evaluation.
    """
    if not isinstance(view, SelectionDatasetView):
        raise TypeError("point evaluation requires SelectionDatasetView")
    results = []
    for family, axis, nu, time_value in candidate_points:
        results.append(
            evaluate_one(
                str(family), str(axis), float(nu), float(time_value)
            )
        )
    return tuple(results)
