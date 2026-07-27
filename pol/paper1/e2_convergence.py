"""Typed, validation-only E2 convergence science and rerun decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch

from .e2_points import SelectionDatasetView
from .metrics import (
    aggregate_errors,
    compare_fields_on_common_grid,
    samplewise_l2_errors,
)
from .model1 import decode_equispaced_point_observation_to_real_fourier
from .random_features import RandomFeatureMap
from .target_representation import real_fourier_synthesis


@dataclass(frozen=True)
class ConvergenceDecision:
    status: Literal["accept", "rerun", "reject"]
    selected_n_sur: int | None
    reason: str | None = None


@dataclass(frozen=True)
class ConvergenceInput:
    """All convergence authority; deliberately contains no test capability."""

    config: Any
    view: SelectionDatasetView
    sample_ids: torch.Tensor
    selection_positions: torch.Tensor
    representatives: dict[str, dict[str, Any]]
    pilot_n_sur: int
    batch_size: int | None
    solve_state: Callable[..., tuple[torch.Tensor, dict[str, Any], str]]
    build_features: Callable[
        [torch.Tensor, str], tuple[torch.Tensor, str]
    ]
    fit_point: Callable[[Any], tuple[dict[str, Any], dict[str, Any], list]]
    build_fit_data: Callable[..., Any]
    select_ridge: Callable[..., tuple[Any, float, list[dict[str, Any]]]]


@dataclass(frozen=True)
class ConvergenceResult:
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def decide_convergence(
    *, pilot_n_sur: int, selected_base: int | None, reruns_remaining: int
) -> ConvergenceDecision:
    if selected_base is None:
        return ConvergenceDecision("reject", None, "no acceptable n_sur")
    if selected_base <= pilot_n_sur:
        return ConvergenceDecision("accept", selected_base)
    if reruns_remaining <= 0:
        return ConvergenceDecision("reject", selected_base, "rerun limit reached")
    return ConvergenceDecision("rerun", selected_base, "selected base exceeds pilot")


def evaluate_convergence(request: ConvergenceInput) -> ConvergenceResult:
    """Evaluate resolution convergence using train/validation data only."""
    if not isinstance(request, ConvergenceInput):
        raise TypeError("convergence requires ConvergenceInput")
    view = request.view
    if not isinstance(view, SelectionDatasetView):
        raise TypeError("convergence requires SelectionDatasetView")
    config = request.config
    assert config.e2 is not None
    conv = config.e2.convergence
    length = config.domain.length
    observation_dim = config.spatial.observation_dim
    candidates = tuple(
        nx for nx in conv.n_sur_candidates if nx >= request.pilot_n_sur
    )
    if len(candidates) < 2:
        return ConvergenceResult(
            (),
            {
                "status": "nonconverged",
                "global_n_sur_base": None,
                "families": {},
                "reason": "no finer n_sur candidate remains",
                "sample_ids": list(conv.sample_ids),
            },
        )
    position_by_id = {
        int(sample_id): position
        for position, sample_id in enumerate(request.sample_ids.tolist())
    }
    conv_positions = torch.tensor(
        [position_by_id[int(i)] for i in conv.sample_ids],
        dtype=torch.long,
        device=view.u0_train_validation.device,
    )
    selection_lookup = {
        int(position): local
        for local, position in enumerate(request.selection_positions.tolist())
    }
    conv_in_selection = torch.tensor(
        [selection_lookup[int(position)] for position in conv_positions.tolist()],
        dtype=torch.long,
        device=view.u0_train_validation.device,
    )
    local_train = view.train_indices
    local_val = view.validation_indices
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"families": {}}
    bases = []
    for family, representative in request.representatives.items():
        states: dict[int, torch.Tensor] = {}
        features: dict[int, torch.Tensor] = {}
        finest = candidates[-1]
        for nx in candidates:
            local_shard = (
                torch.arange(
                    request.selection_positions.numel(),
                    device=local_train.device,
                )
                if nx == finest
                else conv_in_selection
            )
            global_shard = (
                request.selection_positions if nx == finest else conv_positions
            )
            state, _, digest = request.solve_state(
                view.u0_train_validation[local_shard],
                family,
                representative["nu_star"],
                representative["T_star"],
                nx,
                global_shard,
            )
            feature, _ = request.build_features(state, digest)
            states[nx] = state.to(view.u0_train_validation.device)
            features[nx] = feature.to(view.u0_train_validation.device)
        frozen_models, _, _ = request.fit_point(
            request.build_fit_data(
                x_train=features[finest][local_train],
                x_validation=features[finest][local_val],
                y_train=view.target_train,
                y_validation=view.target_validation,
                target_data_validation=view.target_data_validation,
                reference_validation=view.reference_validation,
            )
        )
        frozen2 = frozen_models["model2"]
        selected3 = frozen_models["model3"]
        frozen3 = {}
        for seed in config.e2.model3.evaluation_seeds:
            random_map = RandomFeatureMap.create(
                features[finest].shape[1],
                selected3["width"],
                activation=config.e2.model3.activation,
                seed=seed,
                weight_scale=selected3["weight_scale"],
                bias_scale=selected3["bias_scale"],
                dtype=features[finest].dtype,
                device=features[finest].device,
            )
            augmented = random_map(features[finest][local_train])
            readout, _, _ = request.select_ridge(
                augmented,
                view.target_train,
                augmented,
                view.target_train,
                (float(selected3["zeta"]),),
                tolerance=0.0,
                svd_rcond=config.e2.ridge.svd_rcond,
            )
            frozen3[seed] = (random_map, readout)
        passing = []
        for nx in candidates:
            nx_state = (
                states[nx] if nx != finest else states[nx][conv_in_selection]
            )
            nx_feature = (
                features[nx]
                if nx != finest
                else features[nx][conv_in_selection]
            )
            finest_state = states[finest][conv_in_selection]
            finest_feature = features[finest][conv_in_selection]
            terminal = compare_fields_on_common_grid(
                nx_state,
                finest_state,
                common_nx=finest,
                domain_length=length,
            )["relative_aggregate"]
            feature = aggregate_errors(
                samplewise_l2_errors(
                    nx_feature,
                    finest_feature,
                    domain_length=float(observation_dim),
                )["relative"]
            )
            prediction_sets = []
            named_prediction = {}
            for model_name, pred, pred_ref in (
                (
                    "model1",
                    decode_equispaced_point_observation_to_real_fourier(
                        nx_feature,
                        config.spatial.target_output_dim,
                        domain_length=length,
                    ),
                    decode_equispaced_point_observation_to_real_fourier(
                        finest_feature,
                        config.spatial.target_output_dim,
                        domain_length=length,
                    ),
                ),
                ("model2", frozen2(nx_feature), frozen2(finest_feature)),
            ):
                evidence = aggregate_errors(
                    samplewise_l2_errors(
                        real_fourier_synthesis(
                            pred,
                            config.spatial.reference_nx,
                            domain_length=length,
                        ),
                        real_fourier_synthesis(
                            pred_ref,
                            config.spatial.reference_nx,
                            domain_length=length,
                        ),
                        domain_length=length,
                    )["relative"]
                )
                prediction_sets.append(evidence)
                named_prediction[model_name] = evidence
            model3_seed_evidence = {}
            for seed, (random_map, readout) in frozen3.items():
                evidence = aggregate_errors(
                    samplewise_l2_errors(
                        real_fourier_synthesis(
                            readout(random_map(nx_feature)),
                            config.spatial.reference_nx,
                            domain_length=length,
                        ),
                        real_fourier_synthesis(
                            readout(random_map(finest_feature)),
                            config.spatial.reference_nx,
                            domain_length=length,
                        ),
                        domain_length=length,
                    )["relative"]
                )
                prediction_sets.append(evidence)
                model3_seed_evidence[str(seed)] = evidence
            prediction = {
                "mean": max(item["mean"] for item in prediction_sets),
                "max": max(item["max"] for item in prediction_sets),
            }
            worst_seed = max(
                model3_seed_evidence,
                key=lambda seed: model3_seed_evidence[seed]["max"],
            )
            tol = conv.tolerances
            passed = (
                terminal["mean"] <= tol.terminal_mean
                and terminal["max"] <= tol.terminal_max
                and feature["mean"] <= tol.feature_mean
                and feature["max"] <= tol.feature_max
                and prediction["mean"] <= tol.prediction_mean
                and prediction["max"] <= tol.prediction_max
            )
            rows.append(
                {
                    "family": family,
                    "n_sur": nx,
                    "reference_n_sur": finest,
                    "sample_ids": ",".join(str(i) for i in conv.sample_ids),
                    "sample_membership": "train_or_validation",
                    "terminal_relative_l2_mean": terminal["mean"],
                    "terminal_relative_l2_max": terminal["max"],
                    "feature_relative_l2_mean": feature["mean"],
                    "feature_relative_l2_max": feature["max"],
                    "prediction_relative_l2_mean": prediction["mean"],
                    "prediction_relative_l2_max": prediction["max"],
                    "model1_prediction_mean": named_prediction["model1"]["mean"],
                    "model1_prediction_max": named_prediction["model1"]["max"],
                    "model1_prediction_pass": (
                        named_prediction["model1"]["mean"] <= tol.prediction_mean
                        and named_prediction["model1"]["max"] <= tol.prediction_max
                    ),
                    "model2_prediction_mean": named_prediction["model2"]["mean"],
                    "model2_prediction_max": named_prediction["model2"]["max"],
                    "model2_prediction_pass": (
                        named_prediction["model2"]["mean"] <= tol.prediction_mean
                        and named_prediction["model2"]["max"] <= tol.prediction_max
                    ),
                    "model3_worst_seed": worst_seed,
                    "model3_prediction_mean": max(
                        item["mean"] for item in model3_seed_evidence.values()
                    ),
                    "model3_prediction_max": max(
                        item["max"] for item in model3_seed_evidence.values()
                    ),
                    "model3_prediction_pass": all(
                        item["mean"] <= tol.prediction_mean
                        and item["max"] <= tol.prediction_max
                        for item in model3_seed_evidence.values()
                    ),
                    "model3_evaluation_seed_evidence": __import__("json").dumps(
                        model3_seed_evidence, sort_keys=True
                    ),
                    "model3_aggregate_type":
                        "worst_case_over_evaluation_seeds_and_models",
                    "frozen_readout":
                        "max_of_model1_model2_model3_frozen_at_finest",
                    "status": "pass" if passed else "fail",
                }
            )
            if passed and nx < finest:
                passing.append(nx)
        base = min(passing) if passing else None
        summary["families"][family] = {
            "n_sur_base": base,
            "status": "pass" if base is not None else "fail",
        }
        if base is not None:
            bases.append(base)
    summary["global_n_sur_base"] = (
        max(bases) if len(bases) == len(request.representatives) else None
    )
    summary["status"] = (
        "pass" if summary["global_n_sur_base"] is not None else "fail"
    )
    summary["sample_ids"] = list(conv.sample_ids)
    return ConvergenceResult(tuple(rows), summary)
