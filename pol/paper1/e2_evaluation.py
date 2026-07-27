"""Frozen-plan and test-evaluation reference contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import math
import os
import tempfile

import torch
from scipy.stats import t as student_t

from .e2_cache import atomic_json, stable_hash
from .metrics import samplewise_l2_errors
from .random_features import RandomFeatureMap
from .target_representation import real_fourier_synthesis


@dataclass(frozen=True)
class FrozenPlanReference:
    path: Path
    selection_record_hash: str
    plan_content_hash: str


@dataclass(frozen=True)
class TestEvaluationResult:
    test_rows: tuple[dict[str, Any], ...]
    model3_seed_rows: tuple[dict[str, Any], ...]
    model3_aggregate_rows: tuple[dict[str, Any], ...]


@dataclass
class TestEvaluationContext:
    config: Any
    frozen: FrozenPlanReference
    test_view: Any
    evaluator: Any
    point_order: list[tuple[str, str, float, float]]
    point_models: dict[tuple[str, str, float, float], dict[str, Any]]
    point_selections: dict[tuple[str, str, float, float], dict[str, Any]]
    pilot_n_sur: int
    selection_record_hash: str
    event_log: list[dict[str, Any]]
    solve_state: Callable[..., tuple[torch.Tensor, dict[str, Any], str]]
    build_features: Callable[..., tuple[torch.Tensor, str]]
    metric_row: Callable[..., dict[str, float]]


def build_frozen_plan_payload(
    *,
    config: Any,
    view: Any,
    point_order: list[tuple[str, str, float, float]],
    point_models: dict[tuple[str, str, float, float], dict[str, Any]],
    point_selections: dict[tuple[str, str, float, float], dict[str, Any]],
    pilot_n_sur: int,
    bindings: dict[str, Any],
    selection_record_hash: str,
    select_ridge: Callable[..., tuple[Any, float, list[dict[str, Any]]]],
    tensor_manifest: Callable[[Any, str], dict[str, dict[str, Any]]],
    plan_hash: Callable[[dict[str, Any]], str],
    protocol_version: str,
) -> dict[str, Any]:
    """Freeze evaluation-seed maps/readouts without validation or test data."""
    assert config.e2 is not None
    serialized_by_physical: dict[tuple[Any, ...], dict[str, Any]] = {}
    complete_models: dict[str, Any] = {}
    for identity in point_order:
        bundle = point_models[identity]
        physical_key = bundle["physical_key"]
        if physical_key not in serialized_by_physical:
            selected = bundle["models"]["model3"]
            serialized: dict[str, Any] = {}
            for seed in config.e2.model3.evaluation_seeds:
                random_map = RandomFeatureMap.create(
                    bundle["features"].shape[1],
                    selected["width"],
                    activation=config.e2.model3.activation,
                    seed=seed,
                    weight_scale=selected["weight_scale"],
                    bias_scale=selected["bias_scale"],
                    dtype=bundle["features"].dtype,
                    device=bundle["features"].device,
                )
                augmented = random_map(
                    bundle["features"][view.train_indices]
                )
                readout, _, _ = select_ridge(
                    augmented,
                    view.target_train,
                    augmented,
                    view.target_train,
                    (float(selected["zeta"]),),
                    tolerance=0.0,
                    svd_rcond=config.e2.ridge.svd_rcond,
                )
                serialized[str(seed)] = {
                    "A": random_map.A.detach().cpu(),
                    "c": random_map.c.detach().cpu(),
                    "W": readout.W.detach().cpu(),
                    "b": readout.b.detach().cpu(),
                    "zeta": float(selected["zeta"]),
                    "solver": readout.solver,
                    "rank": readout.numerical_rank,
                    "cutoff": readout.singular_value_cutoff,
                    "svd_rcond": readout.svd_rcond,
                    "parameter_count": int(
                        random_map.A.numel()
                        + random_map.c.numel()
                        + readout.W.numel()
                        + readout.b.numel()
                    ),
                    "readout_frobenius_norm": float(
                        torch.linalg.vector_norm(readout.W)
                    ),
                    "shape": {
                        "A": list(random_map.A.shape),
                        "c": list(random_map.c.shape),
                        "W": list(readout.W.shape),
                        "b": list(readout.b.shape),
                    },
                    "dtype": str(readout.W.dtype),
                }
            serialized_by_physical[physical_key] = serialized
        serialized = serialized_by_physical[physical_key]
        model2 = bundle["models"]["model2"]
        complete_models[str(identity)] = {
            "physical_key": list(physical_key[:-1]),
            "physical_identity": {
                "family": identity[0],
                "nu_tilde": identity[2],
                "T_tilde": identity[3],
                "n_sur": pilot_n_sur,
            },
            "axis_alias": list(identity),
            "model1": {
                "kind": "fixed_equispaced_fourier_decoder",
                "J": config.spatial.observation_dim,
                "q": config.spatial.target_output_dim,
                "domain_length": config.domain.length,
                "q_gt_J_policy": "unobservable_coefficients_zero",
                "parameter_count": 0,
            },
            "model2": {
                "W": model2.W.detach().cpu(),
                "b": model2.b.detach().cpu(),
                "zeta": point_selections[identity]["model2"]["zeta"],
                "solver": model2.solver,
                "rank": model2.numerical_rank,
                "cutoff": model2.singular_value_cutoff,
                "svd_rcond": model2.svd_rcond,
                "parameter_count": int(model2.W.numel() + model2.b.numel()),
                "readout_frobenius_norm": float(
                    torch.linalg.vector_norm(model2.W)
                ),
                "shape": {
                    "W": list(model2.W.shape),
                    "b": list(model2.b.shape),
                },
                "dtype": str(model2.W.dtype),
            },
            "model3": {
                "candidate": {
                    key: bundle["models"]["model3"][key]
                    for key in ("width", "weight_scale", "bias_scale", "zeta")
                },
                "activation": config.e2.model3.activation,
                "scaling": "skip_[phi,rho(Aphi+c)/sqrt(M)]",
                "evaluation_seeds": serialized,
            },
        }
    payload = {
        "schema_version": protocol_version,
        "protocol_version": protocol_version,
        "bindings": bindings,
        "selection_record_hash": selection_record_hash,
        "final_pilot_n_sur": pilot_n_sur,
        "models": complete_models,
        "tensor_hashes": tensor_manifest(complete_models, "models"),
    }
    payload["plan_content_hash"] = plan_hash(payload)
    return payload


def evaluate_test(
    *,
    context: TestEvaluationContext,
) -> tuple[TestEvaluationResult, dict[str, Any]]:
    """Evaluate all final test rows from a durable frozen-plan capability."""
    from .e2_points import TestDatasetView

    if not isinstance(context, TestEvaluationContext):
        raise TypeError("test evaluation requires TestEvaluationContext")
    frozen, test_view = context.frozen, context.test_view
    if not isinstance(frozen, FrozenPlanReference):
        raise TypeError("test evaluation requires FrozenPlanReference")
    if not frozen.path.is_file():
        raise ValueError("frozen plan reference no longer exists")
    if not isinstance(test_view, TestDatasetView):
        raise TypeError("test evaluation requires TestDatasetView")
    config, evaluator = context.config, context.evaluator
    assert config.e2 is not None
    rows: list[dict[str, Any]] = []
    model3_seed_rows: list[dict[str, Any]] = []
    model3_aggregate_rows: list[dict[str, Any]] = []
    saved_models: dict[str, Any] = {}
    frozen_hash = frozen.plan_content_hash
    for identity in context.point_order:
        family, axis, nu, time_value = identity
        selections = context.point_selections[identity]
        state, _, state_digest = context.solve_state(
            family, nu, time_value
        )
        if not any(
            item["event"] == "first_test_state_solve"
            for item in context.event_log
        ):
            context.event_log.append(
                {
                    "event": "first_test_state_solve",
                    "frozen_plan_hash": frozen_hash,
                }
            )
        features, _ = context.build_features(state, state_digest)
        features = features.to(test_view.u0_test.device)
        for model_name in ("model1", "model2", "model3"):
            if model_name == "model1":
                predictions = [
                    (None, evaluator.predict(identity, "model1", features))
                ]
            elif model_name == "model2":
                predictions = [
                    (None, evaluator.predict(identity, "model2", features))
                ]
            else:
                predictions = [
                    (
                        seed,
                        evaluator.predict(
                            identity, "model3", features, seed=seed
                        ),
                    )
                    for seed in config.e2.model3.evaluation_seeds
                ]
            metrics_by_seed = []
            for seed, prediction in predictions:
                metrics = context.metric_row(prediction)
                metrics_by_seed.append(metrics)
                if not any(
                    item["event"] == "first_test_metric"
                    for item in context.event_log
                ):
                    context.event_log.append(
                        {
                            "event": "first_test_metric",
                            "frozen_plan_hash": frozen_hash,
                        }
                    )
                if seed is not None:
                    model3_seed_rows.append(
                        {
                            "family": family,
                            "sweep_axis": axis,
                            "nu_tilde": nu,
                            "T_tilde": time_value,
                            "seed": seed,
                            **metrics,
                            "selection_record_hash":
                                context.selection_record_hash,
                            "frozen_plan_hash": frozen_hash,
                        }
                    )
            averaged = {
                key: sum(row[key] for row in metrics_by_seed)
                / len(metrics_by_seed)
                for key in metrics_by_seed[0]
            }
            row = {
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
                **averaged,
                "selection_record_hash": context.selection_record_hash,
                "frozen_plan_hash": frozen_hash,
            }
            projection = real_fourier_synthesis(
                test_view.target_coefficients_test,
                config.spatial.reference_nx,
                domain_length=config.domain.length,
            )
            representation = samplewise_l2_errors(
                projection,
                test_view.reference_test,
                domain_length=config.domain.length,
            )["relative"]
            row["E_repr_q"] = float(representation.mean())
            row["field_error_to_representation_floor_ratio"] = (
                averaged["field_relative_l2_mean"]
                / max(row["E_repr_q"], 1e-15)
            )
            rows.append(row)
            if model_name == "model3":
                values = torch.tensor(
                    [
                        metric["field_relative_l2_mean"]
                        for metric in metrics_by_seed
                    ],
                    dtype=torch.float64,
                )
                count = len(values)
                mean = float(values.mean())
                std = (
                    float(values.std(unbiased=True)) if count >= 2 else None
                )
                critical = (
                    float(student_t.ppf(0.975, df=count - 1))
                    if count >= 2
                    else None
                )
                half = (
                    None
                    if std is None
                    else critical * std / math.sqrt(count)
                )
                model3_aggregate_rows.append(
                    {
                        "family": family,
                        "sweep_axis": axis,
                        "nu_tilde": nu,
                        "T_tilde": time_value,
                        "seed_count": count,
                        "mean": mean,
                        "std": std,
                        "ci95_low": None if half is None else mean - half,
                        "ci95_high": None if half is None else mean + half,
                        "ci_reason": None
                        if half is not None
                        else "fewer than two evaluation seeds",
                        "selection_record_hash":
                            context.selection_record_hash,
                        "frozen_plan_hash": frozen_hash,
                    }
                )
        saved_models[str(identity)] = evaluator.payload["models"][
            str(identity)
        ]
    return (
        TestEvaluationResult(
            test_rows=tuple(rows),
            model3_seed_rows=tuple(model3_seed_rows),
            model3_aggregate_rows=tuple(model3_aggregate_rows),
        ),
        saved_models,
    )


def publish_and_read_back_frozen_plan(
    *,
    freeze_dir: Path,
    selection_record: dict[str, Any],
    selection_record_hash: str,
    frozen_payload: dict[str, Any],
    validate_payload,
) -> tuple[FrozenPlanReference, Any]:
    """Durably publish selection/plan and return only a validated capability."""
    freeze_dir = Path(freeze_dir)
    selection_path = freeze_dir / "selection_record.json"
    atomic_json(selection_path, selection_record)
    persisted = json.loads(selection_path.read_text(encoding="utf-8"))
    if stable_hash(persisted) != selection_record_hash:
        raise ValueError("selection_record.json read-back hash mismatch")

    frozen_path = freeze_dir / "frozen_evaluation_plan.pt"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".frozen.", suffix=".pt", dir=frozen_path.parent
    )
    os.close(fd)
    try:
        torch.save(frozen_payload, temporary)
        os.replace(temporary, frozen_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    evaluator = validate_payload(frozen_path)
    reference = FrozenPlanReference(
        path=frozen_path,
        selection_record_hash=selection_record_hash,
        plan_content_hash=evaluator.plan_hash,
    )
    return reference, evaluator
