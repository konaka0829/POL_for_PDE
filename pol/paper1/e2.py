"""Paper 1 E2 parameter/time selection workflow."""
from __future__ import annotations

import itertools
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from scipy.stats import t as student_t

from .config import Paper1Config, canonical_config_json
from .datasets import Paper1MasterDataset
from .e2_cache import TensorCache, atomic_json, stable_hash, tensor_hash
from .grids import spectral_resample_periodic
from .interfaces import build_surrogate_initial_state, derive_finite_resolution_data
from .metrics import aggregate_errors, compare_fields_on_common_grid, samplewise_l2_errors
from .model1 import decode_equispaced_point_observation_to_real_fourier
from .observations import observe_equispaced_periodic
from .random_features import RandomFeatureMap
from .readouts import AffineReadout, fit_centered_affine_ridge
from .solvers import solve_burgers_final_state, solve_reaction_diffusion_final_state
from .target_representation import real_fourier_synthesis

E2_SCHEMA_VERSION = "paper1-e2-v2"
MODELS = ("model1", "model2", "model3")


@dataclass(frozen=True)
class TrainValidationData:
    """Selection data boundary: it deliberately contains no test tensors or indices."""

    x_train: torch.Tensor
    x_validation: torch.Tensor
    y_train: torch.Tensor
    y_validation: torch.Tensor
    target_data_validation: torch.Tensor
    reference_validation: torch.Tensor


def select_first_with_tolerance(rows: list[dict[str, Any]], metric: str, tolerance: float) -> dict[str, Any]:
    """Validation-only parameter selection preserving config/grid order."""
    if not rows:
        raise ValueError("cannot select from empty validation rows")
    best = min(float(row[metric]) for row in rows)
    return next(row for row in rows if float(row[metric]) <= best + tolerance)


def sample_id_membership(dataset: Paper1MasterDataset) -> dict[int, str]:
    """Map actual sample IDs to their shuffled split membership."""
    result: dict[int, str] = {}
    for name, indices in (("train", dataset.train_indices), ("validation", dataset.val_indices),
                          ("test", dataset.test_indices)):
        for position in indices.tolist():
            sample_id = int(dataset.sample_ids[position])
            if sample_id in result:
                raise ValueError(f"sample ID {sample_id} occurs in multiple splits")
            result[sample_id] = name
    if len(result) != dataset.sample_ids.numel():
        raise ValueError("split membership is not a disjoint full cover")
    return result


def validate_convergence_membership(
    dataset: Paper1MasterDataset, sample_ids: tuple[int, ...],
) -> dict[int, str]:
    """Reject convergence IDs outside the actual train/validation membership."""
    membership = sample_id_membership(dataset)
    for sample_id in sample_ids:
        where = membership.get(int(sample_id))
        if where is None:
            raise ValueError(f"e2.convergence.sample_ids contains unknown sample ID {sample_id}")
        if where == "test":
            raise ValueError(
                f"e2.convergence.sample_ids contains test sample ID {sample_id}; "
                "convergence may use actual train/validation members only")
    return {int(i): membership[int(i)] for i in sample_ids}


def select_ridge(
    x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
    zetas: tuple[float, ...], *, tolerance: float, svd_rcond: float | None,
    validation_reference_master: torch.Tensor | None = None,
    validation_target_data: torch.Tensor | None = None,
    n_ref: int | None = None, n_tar: int | None = None, domain_length: float = 1.0,
) -> tuple[AffineReadout, float, list[dict[str, Any]]]:
    """Fit a normalized-objective ridge path from one compact SVD.

    The filter is ``s/(s**2 + N*zeta)`` because the data loss is normalized
    by ``N``.  At zero ridge the retained singular values use ``1/s``.
    """
    started = time.perf_counter()
    xm, ym = x_train.mean(0), y_train.mean(0)
    xc, yc = x_train - xm, y_train - ym
    U, singular, Vh = torch.linalg.svd(xc, full_matrices=False)
    cutoff = ((torch.finfo(xc.dtype).eps * max(xc.shape)) if svd_rcond is None
              else svd_rcond) * (float(singular.max()) if singular.numel() else 0.0)
    retained = singular > cutoff
    uy = U.mT @ yc
    candidates: list[dict[str, Any]] = []
    for order, zeta in enumerate(zetas):
        if zeta == 0:
            factors = torch.where(retained, singular.reciprocal(), torch.zeros_like(singular))
            solver = "svd_minimum_norm"
        else:
            factors = singular / (singular.square() + x_train.shape[0] * zeta)
            solver = "svd_ridge_path"
        W = (Vh.mT @ (factors[:, None] * uy)).mT
        b = ym - xm @ W.mT
        model = AffineReadout(W=W, b=b, solver=solver, svd_rcond=svd_rcond,
                              singular_value_cutoff=cutoff,
                              numerical_rank=int(retained.sum()))
        prediction = model(x_val)
        mse = float(torch.mean((prediction - y_val) ** 2))
        if validation_reference_master is not None:
            if validation_target_data is None or n_ref is None or n_tar is None:
                raise ValueError("field-metric ridge selection requires both reference grids")
            metrics = _metric_row(prediction, y_val, validation_reference_master,
                                  validation_target_data, n_tar=n_tar, n_ref=n_ref,
                                  L=domain_length)
            score = metrics["field_relative_l2_mean"]
        else:
            metrics, score = {}, mse
        candidates.append({
            "order": order, "zeta": zeta, "validation_coefficient_mse": mse,
            "validation_field_relative_l2_mean": score, **metrics, "solver": solver,
            "rank": int(retained.sum()), "cutoff": cutoff, "svd_rcond": svd_rcond,
            "model": model})
    best = min(row["validation_field_relative_l2_mean"] for row in candidates)
    tied = [row for row in candidates if row["validation_field_relative_l2_mean"] <= best + tolerance]
    selected = max(tied, key=lambda row: row["zeta"])
    elapsed = time.perf_counter() - started
    provenance = [{**{k: v for k, v in row.items() if k != "model"},
                   "ridge_path_runtime_seconds": elapsed} for row in candidates]
    return selected["model"], float(selected["zeta"]), provenance


def _metric_row(
    coefficients: torch.Tensor, reference_coefficients: torch.Tensor, reference_master: torch.Tensor,
    target_data: torch.Tensor, *, n_tar: int, n_ref: int, L: float,
) -> dict[str, float]:
    prediction_ref = real_fourier_synthesis(coefficients, n_ref, domain_length=L)
    prediction_data = real_fourier_synthesis(coefficients, n_tar, domain_length=L)
    field = samplewise_l2_errors(prediction_ref, reference_master, domain_length=L)
    data = samplewise_l2_errors(prediction_data, target_data, domain_length=L)
    coeff_abs = torch.linalg.vector_norm(coefficients - reference_coefficients, dim=-1)
    coeff_den = torch.linalg.vector_norm(reference_coefficients, dim=-1)
    coeff_rel = coeff_abs / torch.clamp(coeff_den, min=1e-14)
    result: dict[str, float] = {"coefficient_mse": float(torch.mean((coefficients - reference_coefficients) ** 2))}
    for prefix, values in (("field_relative_l2", field["relative"]), ("field_absolute_l2", field["absolute"]),
                           ("data_relative_l2", data["relative"]), ("coefficient_relative_l2", coeff_rel)):
        result.update({f"{prefix}_{key}": value for key, value in aggregate_errors(values).items()})
    return result


def _state_key(
    config: Paper1Config, dataset: Paper1MasterDataset, family: str, nu: float, T: float, n_sur: int,
    sample_positions: torch.Tensor | None = None,
) -> dict[str, Any]:
    assert config.e2 is not None
    e2 = config.e2
    if family == "burgers":
        solver = {
            "solver": e2.burgers.solver, "dt": e2.burgers.dt,
            "fine_dt": e2.burgers.fine_dt, "dealias": e2.burgers.dealias,
            "advection_coefficient": e2.burgers.advection_coefficient,
        }
    else:
        solver = {
            "solver": e2.reaction_diffusion.solver, "dt": e2.reaction_diffusion.dt,
            "nonlinear_filter": e2.reaction_diffusion.nonlinear_filter,
            "alpha": e2.reaction_diffusion.alpha, "beta": e2.reaction_diffusion.beta,
        }
    positions = (torch.arange(dataset.sample_ids.numel()) if sample_positions is None
                 else sample_positions.detach().cpu())
    return {
        "dataset_hash": dataset.metadata["dataset_hash"], "split_hash": dataset.metadata["split_hash"],
        "sample_hash": tensor_hash(dataset.sample_ids), "sample_shard_hash": tensor_hash(positions),
        "n_tar": config.spatial.target_data_nx,
        "input_interpolation": "periodic_spectral", "n_sur": n_sur, "family": family,
        "nu_tilde": nu, "T_tilde": T, "solver": solver, "L": config.domain.length,
        "dtype": config.data.dtype, "device": config.data.device,
    }


def _solve_state(
    config: Paper1Config, u0_data: torch.Tensor, dataset: Paper1MasterDataset, cache: TensorCache,
    family: str, nu: float, T: float, n_sur: int, batch_size: int | None = None,
    sample_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    assert config.e2 is not None
    L, e2 = config.domain.length, config.e2
    u0_sur = build_surrogate_initial_state(u0_data, surrogate_internal_nx=n_sur, domain_length=L)
    key = _state_key(config, dataset, family, nu, T, n_sur, sample_positions)
    def compute():
        chunks, metadata = [], None
        size = u0_sur.shape[0] if batch_size is None else batch_size
        for start in range(0, u0_sur.shape[0], size):
            batch = u0_sur[start:start+size]
            if family == "burgers":
                result = solve_burgers_final_state(
                    batch, nu=nu, T=T, dt=e2.burgers.dt, fine_dt=e2.burgers.fine_dt,
                    solver=e2.burgers.solver, dealias=e2.burgers.dealias, domain_length=L)
                metadata = result.metadata.to_dict() | {"family": family, "nu_tilde": nu, "T_tilde": T,
                                                        "advection_coefficient": 1.0}
            else:
                result = solve_reaction_diffusion_final_state(
                    batch, nu=nu, alpha=e2.reaction_diffusion.alpha, beta=e2.reaction_diffusion.beta,
                    T=T, dt=e2.reaction_diffusion.dt, domain_length=L,
                    nonlinear_filter=e2.reaction_diffusion.nonlinear_filter,
                    context=f"family={family},nu={nu},T={T},n_sur={n_sur},samples={start}:{start+len(batch)}")
                metadata = result.metadata | {"family": family, "T_tilde": T}
            chunks.append(result.values)
        assert metadata is not None
        metadata["batch_size"] = int(size)
        return torch.cat(chunks), metadata
    return cache.get_or_compute("states", key, compute)


def _features(
    config: Paper1Config, state: torch.Tensor, state_digest: str, cache: TensorCache,
) -> tuple[torch.Tensor, str]:
    J, L = config.spatial.observation_dim, config.domain.length
    key = {"state_key": state_digest, "J": J, "positions": "x_j=jL/J", "scaling": "sqrt(L/J)"}
    values, _, digest = cache.get_or_compute(
        "features", key, lambda: (observe_equispaced_periodic(state, J, domain_length=L, l2_scale=True), {}))
    return values, digest


def _fit_point(
    config: Paper1Config, data: TrainValidationData,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Fit all models using train/validation arrays only."""
    assert config.e2 is not None
    e2, L, q = config.e2, config.domain.length, config.spatial.target_output_dim
    features = torch.cat((data.x_train, data.x_validation))
    targets = torch.cat((data.y_train, data.y_validation))
    train = torch.arange(data.x_train.shape[0], device=features.device)
    val = torch.arange(data.x_train.shape[0], features.shape[0], device=features.device)
    models: dict[str, Any] = {"model1": None}
    selections: dict[str, Any] = {"model1": {"kind": "fixed_decoder"}}
    provenance: list[dict[str, Any]] = []
    model2, zeta2, ridge_rows = select_ridge(
        features[train], targets[train], features[val], targets[val], e2.ridge.zetas,
        tolerance=e2.ridge.tie_tolerance, svd_rcond=e2.ridge.svd_rcond,
        validation_reference_master=data.reference_validation,
        validation_target_data=data.target_data_validation,
        n_ref=config.spatial.reference_nx, n_tar=config.spatial.target_data_nx,
        domain_length=L)
    models["model2"], selections["model2"] = model2, {"zeta": zeta2, "ridge_candidates": ridge_rows}
    candidates = []
    for order, (width, ws, bs, common_zeta) in enumerate(itertools.product(
            e2.model3.widths, e2.model3.weight_scales, e2.model3.bias_scales,
            e2.ridge.zetas)):
        seed_metrics, seed_models = [], {}
        for seed in e2.model3.selection_seeds:
            random_map = RandomFeatureMap.create(features.shape[1], width, activation=e2.model3.activation,
                                                 seed=seed, weight_scale=ws, bias_scale=bs,
                                                 dtype=features.dtype, device=features.device)
            augmented = random_map(features)
            readout, zeta, _ = select_ridge(
                augmented[train], targets[train], augmented[val], targets[val],
                (common_zeta,), tolerance=0.0, svd_rcond=e2.ridge.svd_rcond,
                validation_reference_master=data.reference_validation,
                validation_target_data=data.target_data_validation,
                n_ref=config.spatial.reference_nx, n_tar=config.spatial.target_data_nx,
                domain_length=L)
            pred = readout(augmented[val])
            field = real_fourier_synthesis(pred, config.spatial.reference_nx, domain_length=L)
            metric = float(samplewise_l2_errors(
                field, data.reference_validation, domain_length=L)["relative"].mean())
            seed_metrics.append(metric); seed_models[seed] = (random_map, readout, zeta)
            provenance.append({"candidate_order": order, "width": width, "weight_scale": ws,
                               "bias_scale": bs, "zeta": common_zeta, "seed": seed, "validation_field_relative_l2_mean": metric,
                               "selected_zeta": zeta})
        candidates.append({"order": order, "width": width, "weight_scale": ws, "bias_scale": bs,
                           "zeta": common_zeta,
                           "validation_field_relative_l2_mean": sum(seed_metrics) / len(seed_metrics),
                           "seed_models": seed_models})
    best_value = min(row["validation_field_relative_l2_mean"] for row in candidates)
    tied = [row for row in candidates if row["validation_field_relative_l2_mean"] <= best_value + e2.parameter_tie_tolerance]
    selected = min(tied, key=lambda row: (row["width"], -row["zeta"],
                                          row["weight_scale"], row["bias_scale"], row["order"]))
    models["model3"] = selected
    selections["model3"] = {k: v for k, v in selected.items() if k != "seed_models"} | {
        "selection_seeds": list(e2.model3.selection_seeds),
        "selected_zetas": {str(seed): value[2] for seed, value in selected["seed_models"].items()}}
    return models, selections, provenance


def run_e2(
    config: Paper1Config, dataset: Paper1MasterDataset, *, cache_dir: Path, resume: bool,
    pilot_n_sur: int | None = None, auto_reruns_remaining: int | None = None,
    batch_size: int | None = None,
    freeze_dir: Path | None = None,
) -> dict[str, Any]:
    """Run E2 with validation-only selection followed by frozen test evaluation."""
    if config.e2 is None:
        raise ValueError("config must contain e2")
    if dataset.y_target_master is None:
        raise ValueError("dataset y_target_master is required")
    convergence_membership = validate_convergence_membership(
        dataset, config.e2.convergence.sample_ids)
    start = time.perf_counter()
    pilot_n_sur = config.spatial.surrogate_internal_nx if pilot_n_sur is None else int(pilot_n_sur)
    auto_reruns_remaining = config.e2.convergence.max_auto_reruns if auto_reruns_remaining is None else auto_reruns_remaining
    dtype, device = config.data.torch_dtype(), torch.device("cpu" if config.data.device == "auto" else config.data.device)
    uref = dataset.u0_master.to(dtype=dtype, device=device)
    yref = dataset.y_target_master.to(dtype=dtype, device=device)
    finite = derive_finite_resolution_data(
        uref, yref, target_data_nx=config.spatial.target_data_nx,
        target_output_dim=config.spatial.target_output_dim, domain_length=config.domain.length,
        sample_ids=dataset.sample_ids.to(device))
    assert finite.target_coefficients is not None
    split = {"train": dataset.train_indices.to(device), "val": dataset.val_indices.to(device), "test": dataset.test_indices.to(device)}
    selection_positions = torch.cat((split["train"], split["val"]))
    n_train = split["train"].numel()
    local_train = torch.arange(n_train, device=device)
    local_val = torch.arange(n_train, selection_positions.numel(), device=device)
    cache = TensorCache(cache_dir, resume=resume)
    validation_rows, point_models, point_selections, model3_validation = [], {}, {}, []
    point_order: list[tuple[str, str, float, float]] = []
    representatives: dict[str, dict[str, Any]] = {}
    failed_runs: list[dict[str, Any]] = []

    def evaluate_validation_point(family: str, axis: str, nu: float, T: float):
        identity = (family, axis, float(nu), float(T))
        if identity in point_models:
            return
        try:
            state, solver_meta, state_digest = _solve_state(
                config, finite.u0_data[selection_positions], dataset, cache, family, nu, T,
                pilot_n_sur, batch_size, selection_positions)
            features, feature_digest = _features(config, state, state_digest, cache)
            features = features.to(device)
            selection_data = TrainValidationData(
                x_train=features[local_train], x_validation=features[local_val],
                y_train=finite.target_coefficients[split["train"]],
                y_validation=finite.target_coefficients[split["val"]],
                target_data_validation=finite.y_target_data[split["val"]],
                reference_validation=yref[split["val"]])
            models, selections, provenance = _fit_point(config, selection_data)
            point_models[identity] = {"models": models, "features": features.to(device), "solver_metadata": solver_meta,
                                      "state_digest": state_digest, "feature_digest": feature_digest}
            point_selections[identity] = selections; model3_validation.extend(
                [{"family": family, "sweep_axis": axis, "nu_tilde": nu, "T_tilde": T, **row} for row in provenance])
            for model_name in MODELS:
                if model_name == "model1":
                    pred = decode_equispaced_point_observation_to_real_fourier(features[local_val], config.spatial.target_output_dim, domain_length=config.domain.length)
                elif model_name == "model2":
                    pred = models["model2"](features[local_val])
                else:
                    seed_predictions = [readout(random_map(features[local_val])) for random_map, readout, _ in models["model3"]["seed_models"].values()]
                    pred = torch.stack(seed_predictions).mean(0)
                metrics = _metric_row(pred, finite.target_coefficients[split["val"]], yref[split["val"]],
                                      finite.y_target_data[split["val"]], n_tar=config.spatial.target_data_nx,
                                      n_ref=config.spatial.reference_nx, L=config.domain.length)
                if model_name == "model3":
                    metrics["validation_ensemble_prediction_field_relative_l2_mean"] = metrics["field_relative_l2_mean"]
                    metrics["field_relative_l2_mean"] = models["model3"]["validation_field_relative_l2_mean"]
                validation_rows.append({"family": family, "sweep_axis": axis, "parameter_value": nu if axis == "nu_tilde" else T,
                                        "fixed_parameter": T if axis == "nu_tilde" else nu, "nu_tilde": nu, "T_tilde": T,
                                        "model": model_name, "validation_field_relative_l2_mean": metrics["field_relative_l2_mean"],
                                        **metrics, "selected_zeta": selections.get(model_name, {}).get("zeta"),
                                        "state_cache_key": state_digest, "feature_cache_key": feature_digest})
            point_order.append(identity)
        except Exception as exc:
            failed_runs.append({"family": family, "sweep_axis": axis, "nu_tilde": nu, "T_tilde": T, "reason": str(exc)})
            raise

    e2 = config.e2
    model_specific: dict[str, Any] = {}
    representative_identities: dict[str, tuple[str, str, float, float]] = {}
    for family, family_cfg in (("burgers", e2.burgers), ("reaction_diffusion", e2.reaction_diffusion)):
        for nu in family_cfg.nu_grid:
            evaluate_validation_point(family, "nu_tilde", nu, family_cfg.initial_T_anchor)
        model_specific[family] = {}
        for model in MODELS:
            nu_rows = [r for r in validation_rows if r["family"] == family
                       and r["sweep_axis"] == "nu_tilde" and r["model"] == model]
            nu_star = float(select_first_with_tolerance(
                nu_rows, e2.selection_metric, e2.parameter_tie_tolerance)["nu_tilde"])
            time_axis = "T_tilde" if model == e2.representative_model else f"T_tilde_{model}"
            for T in family_cfg.T_grid:
                evaluate_validation_point(family, time_axis, nu_star, T)
            time_rows = [r for r in validation_rows if r["family"] == family
                         and r["sweep_axis"] == time_axis and r["model"] == model]
            selected = select_first_with_tolerance(
                time_rows, e2.selection_metric, e2.parameter_tie_tolerance)
            T_star = float(selected["T_tilde"])
            for round_index in range(e2.coordinate_refinement_rounds):
                nu_axis = (f"nu_refinement_{round_index + 1}" if model == e2.representative_model
                           else f"nu_refinement_{round_index + 1}_{model}")
                for nu in family_cfg.nu_grid:
                    evaluate_validation_point(family, nu_axis, nu, T_star)
                nu_star = float(select_first_with_tolerance(
                    [r for r in validation_rows if r["family"] == family
                     and r["sweep_axis"] == nu_axis and r["model"] == model],
                    e2.selection_metric, e2.parameter_tie_tolerance)["nu_tilde"])
                time_axis = (f"T_refinement_{round_index + 1}" if model == e2.representative_model
                             else f"T_refinement_{round_index + 1}_{model}")
                for T in family_cfg.T_grid:
                    evaluate_validation_point(family, time_axis, nu_star, T)
                selected = select_first_with_tolerance(
                    [r for r in validation_rows if r["family"] == family
                     and r["sweep_axis"] == time_axis and r["model"] == model],
                    e2.selection_metric, e2.parameter_tie_tolerance)
                T_star = float(selected["T_tilde"])
            model_specific[family][model] = {
                "nu_tilde": nu_star, "T_tilde": T_star,
                "validation_field_relative_l2_mean": selected[e2.selection_metric],
                "coordinate_path": "independent_validation_only"}
            if model == e2.representative_model:
                representatives[family] = {
                    "nu_star": nu_star, "T_star": T_star,
                    "representative_model": model, "selection_metric": e2.selection_metric}
                representative_identities[family] = (
                    family, time_axis, nu_star, T_star)
    # Freeze every evaluation-seed map/readout using train data and the common
    # selected candidate. No validation or test choice is made here.
    frozen_evaluation: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    frozen_tensor_payload: dict[str, Any] = {}
    for identity in point_order:
        bundle = point_models[identity]
        selected = bundle["models"]["model3"]
        seed_models: dict[int, tuple[RandomFeatureMap, AffineReadout]] = {}
        serialized: dict[str, Any] = {}
        for seed in e2.model3.evaluation_seeds:
            random_map = RandomFeatureMap.create(
                bundle["features"].shape[1], selected["width"],
                activation=e2.model3.activation, seed=seed,
                weight_scale=selected["weight_scale"], bias_scale=selected["bias_scale"],
                dtype=bundle["features"].dtype, device=bundle["features"].device)
            augmented_train = random_map(bundle["features"][local_train])
            readout, _, _ = select_ridge(
                augmented_train, finite.target_coefficients[split["train"]],
                augmented_train, finite.target_coefficients[split["train"]],
                (float(selected["zeta"]),), tolerance=0.0,
                svd_rcond=e2.ridge.svd_rcond)
            seed_models[seed] = (random_map, readout)
            serialized[str(seed)] = {
                "A": random_map.A.detach().cpu(), "c": random_map.c.detach().cpu(),
                "W": readout.W.detach().cpu(), "b": readout.b.detach().cpu(),
                "zeta": float(selected["zeta"])}
        frozen_evaluation[identity] = {"seed_models": seed_models, "serialized": serialized}
        frozen_tensor_payload[str(identity)] = serialized
    model_hashes = {
        identity: stable_hash({
            seed: {name: tensor_hash(value) if isinstance(value, torch.Tensor) else value
                   for name, value in tensors.items()}
            for seed, tensors in models.items()})
        for identity, models in frozen_tensor_payload.items()}
    selection_record = {"schema_version": E2_SCHEMA_VERSION, "representatives": representatives,
                        "model_specific_optima": model_specific,
                        "point_hyperparameters": {str(key): value for key, value in point_selections.items()},
                        "evaluation_model_hashes": model_hashes,
                        "test_data_used": False}
    selection_hash = stable_hash(selection_record)
    shared_hyperparameters = {}
    for family, representative in representatives.items():
        identity = representative_identities[family]
        shared_hyperparameters[family] = point_selections[identity]

    # This durable read-back boundary precedes every test state/feature solve.
    if freeze_dir is not None:
        selection_path = Path(freeze_dir) / "selection_record.json"
        atomic_json(selection_path, selection_record)
        persisted = __import__("json").loads(selection_path.read_text())
        if stable_hash(persisted) != selection_hash:
            raise ValueError("selection_record.json read-back hash mismatch")
        frozen_path = Path(freeze_dir) / "frozen_evaluation_plan.pt"
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".frozen.", suffix=".pt", dir=frozen_path.parent)
        os.close(fd)
        try:
            torch.save({"schema_version": E2_SCHEMA_VERSION,
                        "selection_record_hash": selection_hash,
                        "models": frozen_tensor_payload}, temporary)
            os.replace(temporary, frozen_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        loaded_frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
        if loaded_frozen.get("selection_record_hash") != selection_hash:
            raise ValueError("frozen evaluation plan read-back hash mismatch")

    # Test states/features are first generated below, after durable selection freeze.
    test_rows, model3_test_by_seed, model3_aggregate, saved_models = [], [], [], {}
    test = split["test"]
    for identity in point_order:
        family, axis, nu, T = identity
        bundle, selections = point_models[identity], point_selections[identity]
        selection_features = bundle["features"]
        test_state, _, test_state_digest = _solve_state(
            config, finite.u0_data[test], dataset, cache, family, nu, T,
            pilot_n_sur, batch_size, test)
        features, _ = _features(config, test_state, test_state_digest, cache)
        features = features.to(device)
        for model_name in MODELS:
            seed_metrics = []
            if model_name == "model1":
                predictions = [(None, decode_equispaced_point_observation_to_real_fourier(
                    features, config.spatial.target_output_dim, domain_length=config.domain.length))]
            elif model_name == "model2":
                predictions = [(None, bundle["models"]["model2"](features))]
            else:
                selected = bundle["models"]["model3"]
                predictions = []
                evaluation_models = frozen_evaluation[identity]["serialized"]
                for seed, (random_map, readout) in frozen_evaluation[identity]["seed_models"].items():
                    augmented_test = random_map(features)
                    predictions.append((seed, readout(augmented_test)))
            metrics_by_seed = []
            for seed, pred in predictions:
                metrics = _metric_row(pred, finite.target_coefficients[test], yref[test], finite.y_target_data[test],
                                      n_tar=config.spatial.target_data_nx, n_ref=config.spatial.reference_nx, L=config.domain.length)
                metrics_by_seed.append(metrics)
                if seed is not None:
                    model3_test_by_seed.append({"family": family, "sweep_axis": axis, "nu_tilde": nu, "T_tilde": T,
                                                "seed": seed, **metrics})
            metrics = {key: sum(row[key] for row in metrics_by_seed) / len(metrics_by_seed) for key in metrics_by_seed[0]}
            row = {"family": family, "sweep_axis": axis, "parameter_value": nu if axis == "nu_tilde" else T,
                   "fixed_parameter": T if axis == "nu_tilde" else nu, "nu_tilde": nu, "T_tilde": T,
                   "model": model_name, **metrics, "selection_record_hash": selection_hash}
            projection = real_fourier_synthesis(finite.target_coefficients[test], config.spatial.reference_nx, domain_length=config.domain.length)
            repr_error = samplewise_l2_errors(projection, yref[test], domain_length=config.domain.length)["relative"]
            row["E_repr_q"] = float(repr_error.mean())
            row["field_error_to_representation_floor_ratio"] = metrics["field_relative_l2_mean"] / max(row["E_repr_q"], 1e-15)
            test_rows.append(row)
            if model_name == "model3":
                values = torch.tensor([r["field_relative_l2_mean"] for r in metrics_by_seed], dtype=torch.float64)
                nseed = len(values)
                mean, std = float(values.mean()), float(values.std(unbiased=True)) if nseed >= 2 else None
                critical = float(student_t.ppf(0.975, df=nseed - 1)) if nseed >= 2 else None
                half = None if std is None else critical * std / math.sqrt(nseed)
                model3_aggregate.append({"family": family, "sweep_axis": axis, "nu_tilde": nu, "T_tilde": T,
                                         "seed_count": nseed, "mean": mean, "std": std,
                                         "ci95_low": None if half is None else mean-half,
                                         "ci95_high": None if half is None else mean+half,
                                         "ci_reason": None if half is not None else "fewer than two evaluation seeds"})
        model2 = bundle["models"]["model2"]
        model3 = bundle["models"]["model3"]
        saved_models[str(identity)] = {
            "model2": {"W": model2.W.detach().cpu(), "b": model2.b.detach().cpu(),
                       "selection": selections["model2"]},
            "model3": {
                "candidate": {k: model3[k] for k in ("width", "weight_scale", "bias_scale", "zeta")},
                "selection_seed_models": {
                    str(seed): {"A": random_map.A.detach().cpu(), "c": random_map.c.detach().cpu(),
                                "W": readout.W.detach().cpu(), "b": readout.b.detach().cpu(), "zeta": zeta}
                    for seed, (random_map, readout, zeta) in model3["seed_models"].items()
                },
                "evaluation_seed_models": evaluation_models,
            },
        }

    convergence_rows, convergence_summary = _convergence(
        config, dataset, finite.u0_data, finite.target_coefficients, split, cache, representatives,
        pilot_n_sur=pilot_n_sur, validation_reference_master=yref[split["val"]], batch_size=batch_size)
    selected_base = convergence_summary["global_n_sur_base"]
    if selected_base is not None and selected_base > pilot_n_sur:
        candidates = [nx for nx in config.e2.convergence.n_sur_candidates if nx >= selected_base]
        if auto_reruns_remaining <= 0 or len(candidates) < 2:
            convergence_summary["status"] = "nonconverged"
            convergence_summary["reason"] = "pilot failed and no permitted rerun with a finer confirmation level"
        else:
            rerun = run_e2(config, dataset, cache_dir=cache_dir, resume=True, pilot_n_sur=selected_base,
                           auto_reruns_remaining=auto_reruns_remaining - 1, batch_size=batch_size,
                           freeze_dir=freeze_dir)
            rerun["auto_rerun_history"] = [{"from_n_sur": pilot_n_sur, "to_n_sur": selected_base},
                                           *rerun.get("auto_rerun_history", [])]
            return rerun
    runtime = time.perf_counter() - start
    solver_metadata = []
    for identity in point_order:
        family, axis, nu, T = identity
        bundle = point_models[identity]
        solver_metadata.append({
            "family": family, "sweep_axis": axis, "nu_tilde": nu, "T_tilde": T,
            "n_sur": pilot_n_sur, "state_key": bundle["state_digest"],
            **bundle["solver_metadata"]})
    coordinate_history = [{
        "family": row["family"], "model": row["model"],
        "stage": row["sweep_axis"], "nu_tilde": row["nu_tilde"],
        "T_tilde": row["T_tilde"],
        "validation_field_relative_l2_mean": row["validation_field_relative_l2_mean"],
    } for row in validation_rows]
    return {
        "validation_sweep": validation_rows, "test_sweep": test_rows,
        "model3_validation_by_seed": model3_validation, "model3_test_by_seed": model3_test_by_seed,
        "model3_test_aggregate": model3_aggregate, "model_specific_optima": model_specific,
        "shared_representatives": representatives, "selection_record": selection_record,
        "selection_record_hash": selection_hash, "convergence_results": convergence_rows,
        "convergence_summary": convergence_summary, "failed_runs": failed_runs,
        "selected_models": saved_models,
        "cache": {"hits": cache.hits, "misses": cache.misses, **cache.stats},
        "runtime_seconds": runtime, "finite_data": finite,
        "pilot_n_sur": pilot_n_sur, "auto_rerun_history": [],
        "shared_hyperparameters": shared_hyperparameters,
        "convergence_sample_membership": convergence_membership,
        "solver_metadata": solver_metadata, "coordinate_history": coordinate_history,
        "attempt_history": [{"attempt_index": 0, "input_pilot_n_sur": pilot_n_sur,
                             "selection_record_hash": selection_hash,
                             "convergence": convergence_summary,
                             "cache": cache.stats, "runtime_seconds": runtime,
                             "status": convergence_summary["status"]}],
    }


def _convergence(
    config: Paper1Config, dataset: Paper1MasterDataset, u0_data: torch.Tensor, targets: torch.Tensor,
    split: dict[str, torch.Tensor], cache: TensorCache, representatives: dict[str, dict[str, Any]],
    *, pilot_n_sur: int, validation_reference_master: torch.Tensor,
    batch_size: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert config.e2 is not None
    conv, L, J = config.e2.convergence, config.domain.length, config.spatial.observation_dim
    candidates = tuple(nx for nx in conv.n_sur_candidates if nx >= pilot_n_sur)
    if len(candidates) < 2:
        return [], {"status": "nonconverged", "global_n_sur_base": None, "families": {},
                    "reason": "no finer n_sur candidate remains", "sample_ids": list(conv.sample_ids)}
    position_by_id = {int(sample_id): position for position, sample_id in enumerate(dataset.sample_ids.tolist())}
    ids = torch.tensor([position_by_id[int(i)] for i in conv.sample_ids],
                       dtype=torch.long, device=u0_data.device)
    rows, summary = [], {"families": {}}
    bases = []
    for family, representative in representatives.items():
        states, features = {}, {}
        for nx in candidates:
            state, _, digest = _solve_state(
                config, u0_data, dataset, cache, family, representative["nu_star"],
                representative["T_star"], nx, batch_size)
            feature, _ = _features(config, state, digest, cache)
            states[nx], features[nx] = state.to(u0_data.device), feature.to(u0_data.device)
        finest = candidates[-1]
        frozen_models, _, _ = _fit_point(config, TrainValidationData(
            x_train=features[finest][split["train"]],
            x_validation=features[finest][split["val"]],
            y_train=targets[split["train"]], y_validation=targets[split["val"]],
            target_data_validation=real_fourier_synthesis(
                targets[split["val"]], config.spatial.target_data_nx,
                domain_length=L),
            reference_validation=validation_reference_master))
        frozen2 = frozen_models["model2"]
        frozen3_entry = next(iter(frozen_models["model3"]["seed_models"].values()))
        frozen_map3, frozen_readout3, _ = frozen3_entry
        passing = []
        for nx in candidates:
            terminal = compare_fields_on_common_grid(states[nx][ids], states[finest][ids], common_nx=finest, domain_length=L)["relative_aggregate"]
            feature_err = samplewise_l2_errors(features[nx][ids], features[finest][ids], domain_length=float(J))["relative"]
            feature = aggregate_errors(feature_err)
            prediction_sets = []
            for pred, pred_ref in (
                (decode_equispaced_point_observation_to_real_fourier(features[nx][ids], config.spatial.target_output_dim, domain_length=L),
                 decode_equispaced_point_observation_to_real_fourier(features[finest][ids], config.spatial.target_output_dim, domain_length=L)),
                (frozen2(features[nx][ids]), frozen2(features[finest][ids])),
                (frozen_readout3(frozen_map3(features[nx][ids])), frozen_readout3(frozen_map3(features[finest][ids]))),
            ):
                pred_field = real_fourier_synthesis(pred, config.spatial.reference_nx, domain_length=L)
                pred_ref_field = real_fourier_synthesis(pred_ref, config.spatial.reference_nx, domain_length=L)
                prediction_sets.append(aggregate_errors(samplewise_l2_errors(pred_field, pred_ref_field, domain_length=L)["relative"]))
            prediction = {"mean": max(item["mean"] for item in prediction_sets),
                          "max": max(item["max"] for item in prediction_sets)}
            tol = conv.tolerances
            passed = terminal["mean"] <= tol.terminal_mean and terminal["max"] <= tol.terminal_max and \
                feature["mean"] <= tol.feature_mean and feature["max"] <= tol.feature_max and \
                prediction["mean"] <= tol.prediction_mean and prediction["max"] <= tol.prediction_max
            rows.append({"family": family, "n_sur": nx, "reference_n_sur": finest,
                         "terminal_relative_l2_mean": terminal["mean"], "terminal_relative_l2_max": terminal["max"],
                         "feature_relative_l2_mean": feature["mean"], "feature_relative_l2_max": feature["max"],
                         "prediction_relative_l2_mean": prediction["mean"], "prediction_relative_l2_max": prediction["max"],
                         "frozen_readout": "max_of_model1_model2_model3_frozen_at_finest",
                         "status": "pass" if passed else "fail"})
            if passed and nx < finest:
                passing.append(nx)
        base = min(passing) if passing else None
        summary["families"][family] = {"n_sur_base": base, "status": "pass" if base is not None else "fail"}
        if base is not None: bases.append(base)
    summary["global_n_sur_base"] = max(bases) if len(bases) == len(representatives) else None
    summary["status"] = "pass" if summary["global_n_sur_base"] is not None else "fail"
    summary["sample_ids"] = list(conv.sample_ids)
    return rows, summary
