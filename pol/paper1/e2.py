"""Paper 1 E2 parameter/time selection workflow."""
from __future__ import annotations

import itertools
import json
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
from .e2_cache import (
    TensorCache, atomic_json, canonical_object, stable_hash, tensor_hash)
from .e2_convergence import ConvergenceDecision, decide_convergence
from .e2_evaluation import FrozenPlanReference, TestEvaluationResult
from .e2_points import (
    PointEvaluationResult,
    SelectionDatasetView,
    TestDatasetView,
)
from .e2_selection import (
    SelectionResult,
    assert_validation_only_record,
    build_selection_bindings,
)
from .grids import spectral_resample_periodic
from .interfaces import build_surrogate_initial_state, derive_finite_resolution_data
from .metrics import aggregate_errors, compare_fields_on_common_grid, samplewise_l2_errors
from .model1 import decode_equispaced_point_observation_to_real_fourier
from .observations import observe_equispaced_periodic
from .random_features import RandomFeatureMap
from .readouts import AffineReadout, fit_centered_affine_ridge
from .solvers import solve_burgers_final_state, solve_reaction_diffusion_final_state
from .target_representation import real_fourier_synthesis

from .protocols import E2_SCHEMA_VERSION
MODELS = ("model1", "model2", "model3")


def dry_run_cost_summary(config: Paper1Config) -> dict[str, Any]:
    """Pure upper-bound cost plan; it performs no tensor allocation or solve."""
    if config.e2 is None:
        raise ValueError("config must contain e2")
    e2 = config.e2
    axis_rows = 0
    physical: set[tuple[str, float, float, int]] = set()
    for family, family_cfg in (
            ("burgers", e2.burgers),
            ("reaction_diffusion", e2.reaction_diffusion)):
        # Initial nu stage, one T path per model, then two coordinate axes per
        # model and refinement round.
        axis_rows += len(family_cfg.nu_grid)
        axis_rows += len(MODELS) * len(family_cfg.T_grid)
        axis_rows += (
            e2.coordinate_refinement_rounds * len(MODELS)
            * (len(family_cfg.nu_grid) + len(family_cfg.T_grid)))
        for nu in family_cfg.nu_grid:
            for T in set((*family_cfg.T_grid, family_cfg.initial_T_anchor)):
                physical.add((family, float(nu), float(T),
                              config.spatial.surrogate_internal_nx))
    nphysical = len(physical)
    map_shapes = (len(e2.model3.widths) * len(e2.model3.weight_scales)
                  * len(e2.model3.bias_scales))
    selection_seed_fits = nphysical * map_shapes * len(e2.model3.selection_seeds)
    per_attempt = {
        "selection_state_solves": nphysical,
        "model2_svd_count": nphysical,
        "model3_selection_svd_count": selection_seed_fits,
        "evaluation_seed_fit_count":
            nphysical * len(e2.model3.evaluation_seeds),
        "convergence_evaluation_seed_fit_count":
            2 * len(e2.model3.evaluation_seeds),
    }
    attempts = 1 + e2.convergence.max_auto_reruns
    return {
        "schema_version": "paper1-e2-cost-v1",
        "axis_row_count": axis_rows,
        "unique_physical_point_count_upper_bound": nphysical,
        "expected_surrogate_state_solves": {
            "selection_train_validation_shard": nphysical,
            "convergence_nonfinest_shards_per_family":
                max(0, len(e2.convergence.n_sur_candidates) - 1),
            "convergence_finest_train_validation_shards_per_family": 1,
            "test_shards_after_final_freeze": nphysical,
        },
        "model2_svd_count": nphysical,
        "model3_svd_count_after_path_reuse": selection_seed_fits,
        "model3_svd_count_legacy_single_zeta":
            selection_seed_fits * len(e2.ridge.zetas),
        "evaluation_seed_fit_count":
            nphysical * len(e2.model3.evaluation_seeds),
        "convergence_fit_count_upper_bound":
            2 * (1 + map_shapes * len(e2.model3.selection_seeds)),
        "zeta_count": len(e2.ridge.zetas),
        "max_attempt_count": attempts,
        "per_attempt": per_attempt,
        "worst_case_total": {
            key: value * attempts for key, value in per_attempt.items()},
    }


def canonical_plan_content(value: Any) -> Any:
    """Canonical, finite representation protecting tensors and all metadata."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("frozen plan contains non-finite tensor")
        return {
            "sha256": tensor_hash(tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    if isinstance(value, dict):
        return {
            str(key): canonical_plan_content(value[key])
            for key in sorted(value, key=str)
            if key != "plan_content_hash"
        }
    if isinstance(value, (list, tuple)):
        return [canonical_plan_content(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("frozen plan contains non-finite metadata")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported frozen-plan value: {type(value).__name__}")


def frozen_plan_content_hash(payload: dict[str, Any]) -> str:
    return stable_hash(canonical_plan_content(payload))


def _tensor_manifest(value: Any, prefix: str = "root") -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite frozen tensor at {prefix}")
        records[prefix] = {
            "sha256": tensor_hash(value), "shape": list(value.shape),
            "dtype": str(value.dtype)}
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            records.update(_tensor_manifest(value[key], f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            records.update(_tensor_manifest(item, f"{prefix}[{index}]"))
    return records


def validate_frozen_evaluation_plan(
        path: Path, *, expected_selection_hash: str | None = None,
        expected_bindings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Strict read-back boundary for the immutable test evaluator input."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version", "protocol_version", "bindings",
        "selection_record_hash", "final_pilot_n_sur", "models",
        "tensor_hashes", "plan_content_hash"}
    if set(payload) != required:
        raise ValueError("frozen evaluation plan schema mismatch")
    if payload["schema_version"] != E2_SCHEMA_VERSION:
        raise ValueError("frozen evaluation plan version mismatch")
    if (expected_selection_hash is not None
            and payload["selection_record_hash"] != expected_selection_hash):
        raise ValueError("frozen evaluation plan selection hash mismatch")
    if expected_bindings is not None and payload["bindings"] != expected_bindings:
        raise ValueError("frozen evaluation plan input binding mismatch")
    actual_tensors = _tensor_manifest(payload["models"], "models")
    if actual_tensors != payload["tensor_hashes"]:
        raise ValueError("frozen evaluation plan tensor hash mismatch")
    if frozen_plan_content_hash(payload) != payload["plan_content_hash"]:
        raise ValueError("frozen evaluation plan content hash mismatch")
    return payload


@dataclass(frozen=True)
class FrozenEvaluator:
    """Inference-only evaluator reconstructed exclusively from a disk plan."""

    payload: dict[str, Any]

    @property
    def plan_hash(self) -> str:
        return str(self.payload["plan_content_hash"])

    def predict(
        self, identity: tuple[str, str, float, float], model: str,
        features: torch.Tensor, *, seed: int | None = None,
    ) -> torch.Tensor:
        record = self.payload["models"][str(identity)]
        if model == "model1":
            spec = record["model1"]
            return decode_equispaced_point_observation_to_real_fourier(
                features, int(spec["q"]),
                domain_length=float(spec["domain_length"]))
        if model == "model2":
            spec = record["model2"]
            return features @ spec["W"].to(features).T + spec["b"].to(features)
        if model != "model3" or seed is None:
            raise ValueError("Model 3 prediction requires an evaluation seed")
        spec = record["model3"]
        seed_spec = spec["evaluation_seeds"][str(seed)]
        random_map = RandomFeatureMap(
            A=seed_spec["A"].to(features), c=seed_spec["c"].to(features),
            activation=str(spec["activation"]), seed=int(seed),
            weight_scale=float(spec["candidate"]["weight_scale"]),
            bias_scale=float(spec["candidate"]["bias_scale"]))
        augmented = random_map(features)
        return augmented @ seed_spec["W"].to(features).T + seed_spec["b"].to(features)


def load_frozen_evaluator(
        plan_path: Path, *, expected_selection_hash: str,
        expected_bindings: dict[str, Any],
) -> FrozenEvaluator:
    return FrozenEvaluator(validate_frozen_evaluation_plan(
        plan_path, expected_selection_hash=expected_selection_hash,
        expected_bindings=expected_bindings))


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
    _return_candidate_models: bool = False,
) -> tuple[AffineReadout, float, list[dict[str, Any]]]:
    """Fit a normalized-objective ridge path from one compact SVD.

    The filter is ``s/(s**2 + N*zeta)`` because the data loss is normalized
    by ``N``.  At zero ridge the retained singular values use ``1/s``.
    """
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
    # Selection provenance is deliberately mathematical content only.  Timing,
    # cache and host diagnostics belong in runtime_diagnostics, never in a
    # selection hash.
    provenance = [
        {**{k: v for k, v in row.items() if k != "model"},
         **({"_model": row["model"]} if _return_candidate_models else {})}
        for row in candidates
    ]
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
    finite_input_hash: str | None = None,
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
        "finite_input_hash": finite_input_hash,
        "sample_hash": tensor_hash(dataset.sample_ids),
        "sample_shard_hash": tensor_hash(positions),
        "n_tar": config.spatial.target_data_nx,
        "input_interpolation": "periodic_spectral", "n_sur": n_sur, "family": family,
        "nu_tilde": nu, "T_tilde": T, "solver": solver, "L": config.domain.length,
        "dtype": config.data.dtype,
        "resolved_device": str(
            torch.device("cpu" if config.data.device == "auto"
                         else config.data.device)),
        "solver_backend": "torch.fft",
        "torch_version": torch.__version__,
        "protocol_version": E2_SCHEMA_VERSION,
    }


def _solve_state(
    config: Paper1Config, u0_data: torch.Tensor, dataset: Paper1MasterDataset, cache: TensorCache,
    family: str, nu: float, T: float, n_sur: int, batch_size: int | None = None,
    sample_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    assert config.e2 is not None
    L, e2 = config.domain.length, config.e2
    u0_sur = build_surrogate_initial_state(u0_data, surrogate_internal_nx=n_sur, domain_length=L)
    key = _state_key(
        config,
        dataset,
        family,
        nu,
        T,
        n_sur,
        sample_positions,
        finite_input_hash=tensor_hash(u0_data),
    )
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
    order = 0
    # One random map and one compact SVD per (width, scales, seed); every zeta
    # is evaluated from that same decomposition.
    for width, ws, bs in itertools.product(
            e2.model3.widths, e2.model3.weight_scales, e2.model3.bias_scales):
        by_zeta: dict[float, dict[int, tuple[RandomFeatureMap, AffineReadout, float]]] = {
            float(z): {} for z in e2.ridge.zetas}
        metrics_by_zeta: dict[float, list[float]] = {float(z): [] for z in e2.ridge.zetas}
        for seed in e2.model3.selection_seeds:
            random_map = RandomFeatureMap.create(features.shape[1], width, activation=e2.model3.activation,
                                                 seed=seed, weight_scale=ws, bias_scale=bs,
                                                 dtype=features.dtype, device=features.device)
            augmented = random_map(features)
            _, _, ridge_path = select_ridge(
                augmented[train], targets[train], augmented[val], targets[val],
                e2.ridge.zetas, tolerance=0.0, svd_rcond=e2.ridge.svd_rcond,
                validation_reference_master=data.reference_validation,
                validation_target_data=data.target_data_validation,
                n_ref=config.spatial.reference_nx, n_tar=config.spatial.target_data_nx,
                domain_length=L, _return_candidate_models=True)
            for ridge_row in ridge_path:
                zeta = float(ridge_row["zeta"])
                zeta_index = tuple(float(v) for v in e2.ridge.zetas).index(zeta)
                readout = ridge_row.pop("_model")
                metric = float(ridge_row["validation_field_relative_l2_mean"])
                metrics_by_zeta[zeta].append(metric)
                by_zeta[zeta][seed] = (random_map, readout, zeta)
                provenance.append({
                    "candidate_order": order + zeta_index, "width": width,
                    "weight_scale": ws, "bias_scale": bs, "zeta": zeta,
                    "seed": seed,
                    "formal_selection_seed_field_relative_l2": metric,
                })
        for common_zeta in e2.ridge.zetas:
            zeta = float(common_zeta)
            seed_metrics = metrics_by_zeta[zeta]
            candidates.append({
                "order": order, "width": width, "weight_scale": ws,
                "bias_scale": bs, "zeta": zeta,
                "validation_field_relative_l2_mean":
                    sum(seed_metrics) / len(seed_metrics),
                "seed_models": by_zeta[zeta],
            })
            order += 1
    best_value = min(row["validation_field_relative_l2_mean"] for row in candidates)
    tied = [row for row in candidates if row["validation_field_relative_l2_mean"] <= best_value + e2.parameter_tie_tolerance]
    selected = min(tied, key=lambda row: (row["width"], -row["zeta"],
                                          row["weight_scale"], row["bias_scale"], row["order"]))
    models["model3"] = selected
    selections["model3"] = {k: v for k, v in selected.items() if k != "seed_models"} | {
        "selection_seeds": list(e2.model3.selection_seeds),
        "selected_zetas": {str(seed): value[2] for seed, value in selected["seed_models"].items()}}
    return models, selections, provenance


def _pretest_failure_result(
    *, config: Paper1Config, finite: Any, cache: TensorCache,
    validation_rows: list[dict[str, Any]],
    model3_validation: list[dict[str, Any]],
    model_specific: dict[str, Any], representatives: dict[str, Any],
    selection_record: dict[str, Any], selection_hash: str,
    convergence_rows: list[dict[str, Any]],
    convergence_summary: dict[str, Any],
    failed_runs: list[dict[str, Any]],
    point_order: list[tuple[str, str, float, float]],
    point_models: dict[tuple[str, str, float, float], dict[str, Any]],
    pilot_n_sur: int, runtime: float,
    shared_hyperparameters: dict[str, Any],
    convergence_membership: dict[int, str],
    event_log: list[dict[str, Any]], failure_reason: str,
) -> dict[str, Any]:
    solver_metadata = []
    physical_aliases = []
    for identity in point_order:
        family, axis, nu, T = identity
        bundle = point_models[identity]
        solver_metadata.append({
            "family": family, "sweep_axis": axis, "nu_tilde": nu,
            "T_tilde": T, "n_sur": pilot_n_sur,
            "state_key": bundle["state_digest"], **bundle["solver_metadata"]})
        physical_aliases.append({
            "family": family, "sweep_axis": axis, "nu_tilde": nu,
            "T_tilde": T, "n_sur": pilot_n_sur,
            "physical_point_hash": stable_hash({
                "family": family, "nu_tilde": nu, "T_tilde": T,
                "n_sur": pilot_n_sur,
                "sample_shard_hash": bundle["physical_key"][-1]}),
            "state_cache_key": bundle["state_digest"],
            "feature_cache_key": bundle["feature_digest"]})
    coordinate_history = [{
        "family": row["family"], "model": row["model"],
        "stage": row["sweep_axis"], "nu_tilde": row["nu_tilde"],
        "T_tilde": row["T_tilde"],
        "validation_field_relative_l2_mean":
            row["validation_field_relative_l2_mean"],
    } for row in validation_rows]
    attempt = {
        "attempt_index": 0, "input_pilot_n_sur": pilot_n_sur,
        "selection_record_hash": selection_hash,
        "selected_shared_points": representatives,
        "selected_model_specific_points": model_specific,
        "convergence": convergence_summary, "cache": cache.stats,
        "runtime_seconds": runtime, "status": "nonconverged",
        "test_evaluated": False, "rerun_reason": None,
    }
    return {
        "procedural_status": "nonconverged", "test_evaluated": False,
        "failure_kind": "n_sur_nonconvergence",
        "failure_reason": failure_reason,
        "validation_sweep": validation_rows,
        "test_sweep": [], "model3_validation_by_seed": model3_validation,
        "model3_test_by_seed": [], "model3_test_aggregate": [],
        "model_specific_optima": model_specific,
        "shared_representatives": representatives,
        "selection_record": selection_record,
        "selection_record_hash": selection_hash,
        "frozen_plan_hash": None,
        "convergence_results": convergence_rows,
        "convergence_summary": convergence_summary,
        "failed_runs": [*failed_runs, {
            "failure_kind": "n_sur_nonconvergence",
            "reason": failure_reason,
            "family_failures": [
                {"family": family, **summary}
                for family, summary in
                convergence_summary.get("families", {}).items()
                if summary.get("status") != "pass"]}],
        "selected_models": {}, "cache": {
            "hits": cache.hits, "misses": cache.misses, **cache.stats},
        "runtime_seconds": runtime, "finite_data": finite,
        "pilot_n_sur": pilot_n_sur, "auto_rerun_history": [],
        "shared_hyperparameters": shared_hyperparameters,
        "convergence_sample_membership": convergence_membership,
        "solver_metadata": solver_metadata,
        "coordinate_history": coordinate_history,
        "physical_point_aliases": physical_aliases,
        "attempt_history": [attempt], "event_log": event_log,
    }


def _run_e2_attempt(
    config: Paper1Config, dataset: Paper1MasterDataset, *, cache_dir: Path, resume: bool,
    pilot_n_sur: int | None = None, auto_reruns_remaining: int | None = None,
    batch_size: int | None = None,
    freeze_dir: Path | None = None,
    input_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run E2 with validation-only selection followed by frozen test evaluation."""
    if config.e2 is None:
        raise ValueError("config must contain e2")
    if dataset.y_target_master is None:
        raise ValueError("dataset y_target_master is required")
    convergence_membership = validate_convergence_membership(
        dataset, config.e2.convergence.sample_ids)
    start = time.perf_counter()
    event_log: list[dict[str, Any]] = []
    input_bindings = {} if input_bindings is None else canonical_object(input_bindings)
    pilot_n_sur = config.spatial.surrogate_internal_nx if pilot_n_sur is None else int(pilot_n_sur)
    auto_reruns_remaining = config.e2.convergence.max_auto_reruns if auto_reruns_remaining is None else auto_reruns_remaining
    dtype, device = config.data.torch_dtype(), torch.device("cpu" if config.data.device == "auto" else config.data.device)
    uref = dataset.u0_master.to(dtype=dtype, device=device)
    split_cpu = {
        "train": dataset.train_indices.detach().cpu(),
        "val": dataset.val_indices.detach().cpu(),
        "test": dataset.test_indices.detach().cpu(),
    }
    split = {name: indices.to(device) for name, indices in split_cpu.items()}
    selection_positions_cpu = torch.cat(
        (split_cpu["train"], split_cpu["val"])
    )
    selection_positions = selection_positions_cpu.to(device)
    # Materialize only train/validation targets before the durable freeze
    # boundary.  In particular, no finite representation of a test label
    # exists in selection or convergence scope.
    selection_reference = dataset.y_target_master[
        selection_positions_cpu
    ].to(dtype=dtype, device=device)
    finite = derive_finite_resolution_data(
        uref[selection_positions], selection_reference,
        target_data_nx=config.spatial.target_data_nx,
        target_output_dim=config.spatial.target_output_dim, domain_length=config.domain.length,
        sample_ids=dataset.sample_ids.to(device)[selection_positions])
    assert finite.target_coefficients is not None
    # Selection identity deliberately excludes every full-target/test-target
    # digest.  Full dataset provenance remains in the recipe data manifest,
    # while this binding contains only inputs and train/validation labels.
    selection_bindings = build_selection_bindings(
        input_bindings,
        train_indices=torch.arange(split["train"].numel(), device=device),
        validation_indices=torch.arange(
            split["train"].numel(), selection_positions.numel(), device=device
        ),
        u0_data=finite.u0_data,
        target_coefficients=finite.target_coefficients,
        target_data=finite.y_target_data,
        reference=selection_reference,
    )
    n_train = split["train"].numel()
    local_train = torch.arange(n_train, device=device)
    local_val = torch.arange(n_train, selection_positions.numel(), device=device)
    selection_view = SelectionDatasetView(
        sample_ids=dataset.sample_ids.to(device)[selection_positions],
        train_indices=local_train,
        validation_indices=local_val,
        u0_train_validation=finite.u0_data,
        target_train=finite.target_coefficients[local_train],
        target_validation=finite.target_coefficients[local_val],
        target_data_validation=finite.y_target_data[local_val],
        reference_validation=selection_reference[local_val],
    )
    cache = TensorCache(cache_dir, resume=resume)
    validation_rows, point_models, point_selections, model3_validation = [], {}, {}, []
    physical_models: dict[tuple[str, float, float, int, str], tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = {}
    point_order: list[tuple[str, str, float, float]] = []
    representatives: dict[str, dict[str, Any]] = {}
    failed_runs: list[dict[str, Any]] = []

    def evaluate_validation_point(family: str, axis: str, nu: float, T: float):
        identity = (family, axis, float(nu), float(T))
        if identity in point_models:
            return
        try:
            physical_key = (
                family, float(nu), float(T), pilot_n_sur,
                tensor_hash(selection_positions.detach().cpu()))
            if physical_key not in physical_models:
                state, solver_meta, state_digest = _solve_state(
                    config, finite.u0_data, dataset, cache,
                    family, nu, T, pilot_n_sur, batch_size,
                    selection_positions)
                features, feature_digest = _features(
                    config, state, state_digest, cache)
                features = features.to(device)
                selection_data = TrainValidationData(
                    x_train=features[selection_view.train_indices],
                    x_validation=features[selection_view.validation_indices],
                    y_train=selection_view.target_train,
                    y_validation=selection_view.target_validation,
                    target_data_validation=selection_view.target_data_validation,
                    reference_validation=selection_view.reference_validation)
                models, selections, provenance = _fit_point(
                    config, selection_data)
                bundle = {
                    "models": models, "features": features,
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
                physical_models[physical_key] = (bundle, selections, provenance)
            bundle, selections, provenance = physical_models[physical_key]
            point_models[identity] = bundle
            point_selections[identity] = selections
            model3_validation.extend(
                [{"family": family, "sweep_axis": axis, "nu_tilde": nu,
                  "T_tilde": T, **row} for row in provenance])
            features = bundle["features"]
            models = bundle["models"]
            state_digest = bundle["state_digest"]
            feature_digest = bundle["feature_digest"]
            for model_name in MODELS:
                if model_name == "model1":
                    pred = decode_equispaced_point_observation_to_real_fourier(features[local_val], config.spatial.target_output_dim, domain_length=config.domain.length)
                elif model_name == "model2":
                    pred = models["model2"](features[local_val])
                else:
                    seed_predictions = [readout(random_map(features[local_val])) for random_map, readout, _ in models["model3"]["seed_models"].values()]
                    pred = torch.stack(seed_predictions).mean(0)
                metrics = _metric_row(pred, selection_view.target_validation, selection_view.reference_validation,
                                      selection_view.target_data_validation, n_tar=config.spatial.target_data_nx,
                                      n_ref=config.spatial.reference_nx, L=config.domain.length)
                if model_name == "model3":
                    ensemble_metrics = {
                        f"ensemble_prediction_{key}": value
                        for key, value in metrics.items()}
                    formal_values = [
                        float(samplewise_l2_errors(
                            real_fourier_synthesis(
                                readout(random_map(features[local_val])),
                                config.spatial.reference_nx,
                                domain_length=config.domain.length),
                            selection_view.reference_validation,
                            domain_length=config.domain.length)["relative"].mean())
                        for random_map, readout, _ in
                        models["model3"]["seed_models"].values()]
                    formal_tensor = torch.tensor(
                        formal_values, dtype=torch.float64)
                    metrics = {
                        "formal_seed_metric_mean": float(formal_tensor.mean()),
                        "formal_seed_metric_std": (
                            float(formal_tensor.std(unbiased=True))
                            if len(formal_values) > 1 else 0.0),
                        "formal_seed_metric_max": float(formal_tensor.max()),
                        "formal_seed_count": len(formal_values),
                        **ensemble_metrics,
                    }
                validation_rows.append({"family": family, "sweep_axis": axis, "parameter_value": nu if axis == "nu_tilde" else T,
                                        "fixed_parameter": T if axis == "nu_tilde" else nu, "nu_tilde": nu, "T_tilde": T,
                                        "model": model_name,
                                        "validation_field_relative_l2_mean": (
                                            metrics["formal_seed_metric_mean"]
                                            if model_name == "model3"
                                            else metrics["field_relative_l2_mean"]),
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

    selection_record = {
        "schema_version": E2_SCHEMA_VERSION,
        "protocol_version": E2_SCHEMA_VERSION,
        "bindings": selection_bindings,
        "representatives": representatives,
        "model_specific_optima": model_specific,
        "point_hyperparameters": {
            str(key): value for key, value in point_selections.items()},
        "test_data_used": False,
    }
    assert_validation_only_record(selection_record)
    selection_hash = stable_hash(selection_record)
    selection_result = SelectionResult(
        representatives=representatives,
        model_specific_optima=model_specific,
        selection_record=selection_record,
        selection_record_hash=selection_hash,
    )
    shared_hyperparameters = {}
    for family, representative in representatives.items():
        identity = representative_identities[family]
        shared_hyperparameters[family] = point_selections[identity]

    # Resolution is part of selection.  Convergence uses only configured
    # train/validation members and completes before a test state, feature or
    # label is generated.  A rejected pilot returns immediately into the next
    # attempt, so rejected attempts cannot evaluate test data.
    convergence_rows, convergence_summary = _convergence(
        config, dataset, selection_view, selection_positions,
        cache, representatives, pilot_n_sur=pilot_n_sur,
        batch_size=batch_size)
    selected_base = convergence_summary["global_n_sur_base"]
    decision: ConvergenceDecision = decide_convergence(
        pilot_n_sur=pilot_n_sur,
        selected_base=selected_base,
        reruns_remaining=auto_reruns_remaining,
    )
    event_log.append({
        "event": "convergence_complete", "pilot_n_sur": pilot_n_sur,
        "selected_base": selected_base, "test_evaluated": False})
    if decision.status == "reject" and selected_base is None:
        runtime = time.perf_counter() - start
        reason = str(convergence_summary.get(
            "reason", "no strictly finer-confirmed acceptable n_sur"))
        return _pretest_failure_result(
            config=config, finite=finite, cache=cache,
            validation_rows=validation_rows,
            model3_validation=model3_validation,
            model_specific=model_specific, representatives=representatives,
            selection_record=selection_record, selection_hash=selection_hash,
            convergence_rows=convergence_rows,
            convergence_summary=convergence_summary,
            failed_runs=failed_runs, point_order=point_order,
            point_models=point_models, pilot_n_sur=pilot_n_sur,
            runtime=runtime, shared_hyperparameters=shared_hyperparameters,
            convergence_membership=convergence_membership,
            event_log=event_log, failure_reason=reason)
    if decision.status in {"rerun", "reject"} and selected_base is not None and selected_base > pilot_n_sur:
        candidates = [
            nx for nx in config.e2.convergence.n_sur_candidates
            if nx >= selected_base]
        if auto_reruns_remaining <= 0 or len(candidates) < 2:
            convergence_summary["status"] = "nonconverged"
            convergence_summary["reason"] = (
                "pilot failed and no permitted rerun with a finer confirmation level")
            runtime = time.perf_counter() - start
            return _pretest_failure_result(
                config=config, finite=finite, cache=cache,
                validation_rows=validation_rows,
                model3_validation=model3_validation,
                model_specific=model_specific, representatives=representatives,
                selection_record=selection_record,
                selection_hash=selection_hash,
                convergence_rows=convergence_rows,
                convergence_summary=convergence_summary,
                failed_runs=failed_runs, point_order=point_order,
                point_models=point_models, pilot_n_sur=pilot_n_sur,
                runtime=runtime,
                shared_hyperparameters=shared_hyperparameters,
                convergence_membership=convergence_membership,
                event_log=event_log,
                failure_reason=convergence_summary["reason"])
        else:
            attempt_runtime = time.perf_counter() - start
            rejected = {
                "attempt_index": 0, "input_pilot_n_sur": pilot_n_sur,
                "selected_next_pilot": selected_base,
                "convergence": convergence_summary, "cache": cache.stats,
                "runtime_seconds": attempt_runtime, "status": "rerun",
                "rerun_reason": "selected base exceeds pilot",
                "test_evaluated": False,
            }
            return {
                "_rerun_request": True,
                "next_pilot_n_sur": selected_base,
                "rejected_attempt": rejected,
                "runtime_seconds": attempt_runtime,
            }

    # Freeze every evaluation-seed map/readout using train data and the common
    # selected candidate. No validation or test choice is made here.
    frozen_evaluation: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    frozen_tensor_payload: dict[str, Any] = {}
    frozen_by_physical: dict[tuple[Any, ...], dict[str, Any]] = {}
    for identity in point_order:
        bundle = point_models[identity]
        physical_key = bundle["physical_key"]
        if physical_key in frozen_by_physical:
            frozen_evaluation[identity] = frozen_by_physical[physical_key]
            frozen_tensor_payload[str(identity)] = frozen_evaluation[identity]["serialized"]
            continue
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
                augmented_train, selection_view.target_train,
                augmented_train, selection_view.target_train,
                (float(selected["zeta"]),), tolerance=0.0,
                svd_rcond=e2.ridge.svd_rcond)
            seed_models[seed] = (random_map, readout)
            serialized[str(seed)] = {
                "A": random_map.A.detach().cpu(), "c": random_map.c.detach().cpu(),
                "W": readout.W.detach().cpu(), "b": readout.b.detach().cpu(),
                "zeta": float(selected["zeta"]),
                "solver": readout.solver, "rank": readout.numerical_rank,
                "cutoff": readout.singular_value_cutoff,
                "svd_rcond": readout.svd_rcond,
                "parameter_count": int(
                    random_map.A.numel() + random_map.c.numel()
                    + readout.W.numel() + readout.b.numel()),
                "readout_frobenius_norm": float(
                    torch.linalg.vector_norm(readout.W)),
                "shape": {
                    "A": list(random_map.A.shape), "c": list(random_map.c.shape),
                    "W": list(readout.W.shape), "b": list(readout.b.shape)},
                "dtype": str(readout.W.dtype)}
        frozen_evaluation[identity] = {"seed_models": seed_models, "serialized": serialized}
        frozen_by_physical[physical_key] = frozen_evaluation[identity]
        frozen_tensor_payload[str(identity)] = serialized
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
            complete_models = {}
            for identity in point_order:
                bundle = point_models[identity]
                model2 = bundle["models"]["model2"]
                complete_models[str(identity)] = {
                    "physical_key": list(bundle["physical_key"][:-1]),
                    "physical_identity": {
                        "family": identity[0], "nu_tilde": identity[2],
                        "T_tilde": identity[3], "n_sur": pilot_n_sur},
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
                        "parameter_count": int(
                            model2.W.numel() + model2.b.numel()),
                        "readout_frobenius_norm": float(
                            torch.linalg.vector_norm(model2.W)),
                        "shape": {
                            "W": list(model2.W.shape),
                            "b": list(model2.b.shape)},
                        "dtype": str(model2.W.dtype),
                    },
                    "model3": {
                        "candidate": {
                            key: bundle["models"]["model3"][key]
                            for key in ("width", "weight_scale",
                                        "bias_scale", "zeta")},
                        "activation": e2.model3.activation,
                        "scaling": "skip_[phi,rho(Aphi+c)/sqrt(M)]",
                        "evaluation_seeds": frozen_tensor_payload[str(identity)],
                    },
                }
            tensor_hashes = _tensor_manifest(complete_models, "models")
            frozen_payload = {
                "schema_version": E2_SCHEMA_VERSION,
                "protocol_version": E2_SCHEMA_VERSION,
                "bindings": selection_bindings,
                "selection_record_hash": selection_hash,
                "final_pilot_n_sur": pilot_n_sur,
                "models": complete_models,
                "tensor_hashes": tensor_hashes,
            }
            frozen_payload["plan_content_hash"] = frozen_plan_content_hash(
                frozen_payload)
            torch.save(frozen_payload, temporary)
            os.replace(temporary, frozen_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        evaluator = load_frozen_evaluator(
            frozen_path, expected_selection_hash=selection_hash,
            expected_bindings=selection_bindings)
        frozen_plan_hash = evaluator.plan_hash
        frozen_reference = FrozenPlanReference(
            path=frozen_path,
            selection_record_hash=selection_result.selection_record_hash,
            plan_content_hash=frozen_plan_hash,
        )
        event_log.append({
            "event": "freeze_read_back", "selection_record_hash": selection_hash,
            "frozen_plan_hash": frozen_plan_hash,
            "test_evaluated": False})
    else:
        raise ValueError("freeze_dir is required for disk-only test evaluation")

    # Open exactly the test slice below, after durable selection freeze.  A
    # full target tensor/finite representation never exists in this scope.
    test_reference = dataset.y_target_master[split_cpu["test"]].to(
        dtype=dtype, device=device
    )
    test_inputs = uref[split["test"]]
    finite_test = derive_finite_resolution_data(
        test_inputs,
        test_reference,
        target_data_nx=config.spatial.target_data_nx,
        target_output_dim=config.spatial.target_output_dim,
        domain_length=config.domain.length,
        sample_ids=dataset.sample_ids[split_cpu["test"]].to(device),
    )
    assert finite_test.target_coefficients is not None
    test_view = TestDatasetView(
        sample_ids=dataset.sample_ids[split_cpu["test"]].to(device),
        indices=split["test"],
        u0_test=finite_test.u0_data,
        target_coefficients_test=finite_test.target_coefficients,
        target_data_test=finite_test.y_target_data,
        reference_test=test_reference,
    )
    test_rows, model3_test_by_seed, model3_aggregate, saved_models = [], [], [], {}
    test = test_view.indices
    for identity in point_order:
        family, axis, nu, T = identity
        bundle, selections = point_models[identity], point_selections[identity]
        selection_features = bundle["features"]
        test_state, _, test_state_digest = _solve_state(
            config, test_view.u0_test, dataset, cache, family, nu, T,
            pilot_n_sur, batch_size, test)
        if not any(item["event"] == "first_test_state_solve" for item in event_log):
            event_log.append({
                "event": "first_test_state_solve",
                "frozen_plan_hash": frozen_plan_hash})
        features, _ = _features(config, test_state, test_state_digest, cache)
        features = features.to(device)
        for model_name in MODELS:
            seed_metrics = []
            if model_name == "model1":
                predictions = [(None, evaluator.predict(
                    identity, "model1", features))]
            elif model_name == "model2":
                predictions = [(None, evaluator.predict(
                    identity, "model2", features))]
            else:
                predictions = []
                for seed in e2.model3.evaluation_seeds:
                    predictions.append((seed, evaluator.predict(
                        identity, "model3", features, seed=seed)))
            metrics_by_seed = []
            for seed, pred in predictions:
                metrics = _metric_row(pred, test_view.target_coefficients_test,
                                      test_view.reference_test, test_view.target_data_test,
                                      n_tar=config.spatial.target_data_nx, n_ref=config.spatial.reference_nx, L=config.domain.length)
                metrics_by_seed.append(metrics)
                if not any(item["event"] == "first_test_metric" for item in event_log):
                    event_log.append({
                        "event": "first_test_metric",
                        "frozen_plan_hash": frozen_plan_hash})
                if seed is not None:
                    model3_test_by_seed.append({"family": family, "sweep_axis": axis, "nu_tilde": nu, "T_tilde": T,
                                                "seed": seed, **metrics,
                                                "selection_record_hash": selection_hash,
                                                "frozen_plan_hash": frozen_plan_hash})
            metrics = {key: sum(row[key] for row in metrics_by_seed) / len(metrics_by_seed) for key in metrics_by_seed[0]}
            row = {"family": family, "sweep_axis": axis, "parameter_value": nu if axis == "nu_tilde" else T,
                   "fixed_parameter": T if axis == "nu_tilde" else nu, "nu_tilde": nu, "T_tilde": T,
                   "model": model_name, **metrics,
                   "selection_record_hash": selection_hash,
                   "frozen_plan_hash": frozen_plan_hash}
            projection = real_fourier_synthesis(test_view.target_coefficients_test, config.spatial.reference_nx, domain_length=config.domain.length)
            repr_error = samplewise_l2_errors(projection, test_view.reference_test, domain_length=config.domain.length)["relative"]
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
                                         "ci_reason": None if half is not None else "fewer than two evaluation seeds",
                                         "selection_record_hash": selection_hash,
                                         "frozen_plan_hash": frozen_plan_hash})
        saved_models[str(identity)] = evaluator.payload["models"][str(identity)]

    runtime = time.perf_counter() - start
    test_result = TestEvaluationResult(
        test_rows=tuple(test_rows),
        model3_seed_rows=tuple(model3_test_by_seed),
        model3_aggregate_rows=tuple(model3_aggregate),
    )
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
    physical_aliases = [{
        "family": identity[0], "sweep_axis": identity[1],
        "nu_tilde": identity[2], "T_tilde": identity[3],
        "n_sur": pilot_n_sur,
        "physical_point_hash": stable_hash({
            "family": identity[0], "nu_tilde": identity[2],
            "T_tilde": identity[3], "n_sur": pilot_n_sur,
            "sample_shard_hash": point_models[identity]["physical_key"][-1],
        }),
        "state_cache_key": point_models[identity]["state_digest"],
        "feature_cache_key": point_models[identity]["feature_digest"],
    } for identity in point_order]
    return {
        "procedural_status": "pass", "test_evaluated": True,
        "failure_kind": None, "failure_reason": None,
        "validation_sweep": validation_rows, "test_sweep": list(test_result.test_rows),
        "model3_validation_by_seed": model3_validation, "model3_test_by_seed": list(test_result.model3_seed_rows),
        "model3_test_aggregate": list(test_result.model3_aggregate_rows), "model_specific_optima": selection_result.model_specific_optima,
        "shared_representatives": selection_result.representatives, "selection_record": selection_result.selection_record,
        "selection_record_hash": selection_result.selection_record_hash, "convergence_results": convergence_rows,
        "frozen_plan_hash": frozen_reference.plan_content_hash,
        "convergence_summary": convergence_summary, "failed_runs": failed_runs,
        "selected_models": saved_models,
        "cache": {"hits": cache.hits, "misses": cache.misses, **cache.stats},
        # Retained for internal compatibility; this is intentionally the
        # train/validation finite view, never a full target materialization.
        "runtime_seconds": runtime, "finite_data": finite,
        "pilot_n_sur": pilot_n_sur, "auto_rerun_history": [],
        "shared_hyperparameters": shared_hyperparameters,
        "convergence_sample_membership": convergence_membership,
        "solver_metadata": solver_metadata, "coordinate_history": coordinate_history,
        "physical_point_aliases": physical_aliases,
        "event_log": event_log,
        "attempt_history": [{"attempt_index": 0, "input_pilot_n_sur": pilot_n_sur,
                             "selection_record_hash": selection_hash,
                             "frozen_plan_hash": frozen_plan_hash,
                             "selected_shared_points": representatives,
                             "selected_model_specific_points": model_specific,
                             "convergence": convergence_summary,
                             "cache": cache.stats, "runtime_seconds": runtime,
                             "status": convergence_summary["status"],
                             "test_evaluated": True}],
    }


def run_e2(
    config: Paper1Config,
    dataset: Paper1MasterDataset,
    *,
    cache_dir: Path,
    resume: bool,
    pilot_n_sur: int | None = None,
    auto_reruns_remaining: int | None = None,
    batch_size: int | None = None,
    freeze_dir: Path | None = None,
    input_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Orchestrate explicit E2 attempts; rejected attempts never reach test."""
    if config.e2 is None:
        raise ValueError("config must contain e2")
    current_pilot = (
        config.spatial.surrogate_internal_nx
        if pilot_n_sur is None
        else int(pilot_n_sur)
    )
    remaining = (
        config.e2.convergence.max_auto_reruns
        if auto_reruns_remaining is None
        else int(auto_reruns_remaining)
    )
    rejected: list[dict[str, Any]] = []
    rerun_history: list[dict[str, Any]] = []
    rejected_runtime = 0.0
    while True:
        result = _run_e2_attempt(
            config,
            dataset,
            cache_dir=cache_dir,
            resume=resume or bool(rejected),
            pilot_n_sur=current_pilot,
            auto_reruns_remaining=remaining,
            batch_size=batch_size,
            freeze_dir=freeze_dir,
            input_bindings=input_bindings,
        )
        if not result.pop("_rerun_request", False):
            attempts = [*rejected, *result.get("attempt_history", [])]
            for index, attempt in enumerate(attempts):
                attempt["attempt_index"] = index
            result["attempt_history"] = attempts
            result["auto_rerun_history"] = [
                *rerun_history,
                *result.get("auto_rerun_history", []),
            ]
            result["runtime_seconds"] = (
                rejected_runtime + float(result["runtime_seconds"])
            )
            return result
        attempt = result["rejected_attempt"]
        next_pilot = int(result["next_pilot_n_sur"])
        rejected.append(attempt)
        rejected_runtime += float(result["runtime_seconds"])
        rerun_history.append(
            {
                "from_n_sur": current_pilot,
                "to_n_sur": next_pilot,
                "reason": attempt["rerun_reason"],
            }
        )
        current_pilot = next_pilot
        remaining -= 1


def _convergence(
    config: Paper1Config,
    dataset: Paper1MasterDataset,
    view: SelectionDatasetView,
    selection_positions: torch.Tensor,
    cache: TensorCache,
    representatives: dict[str, dict[str, Any]],
    *,
    pilot_n_sur: int,
    batch_size: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert config.e2 is not None
    conv, L, J = config.e2.convergence, config.domain.length, config.spatial.observation_dim
    candidates = tuple(nx for nx in conv.n_sur_candidates if nx >= pilot_n_sur)
    if len(candidates) < 2:
        return [], {"status": "nonconverged", "global_n_sur_base": None, "families": {},
                    "reason": "no finer n_sur candidate remains", "sample_ids": list(conv.sample_ids)}
    position_by_id = {int(sample_id): position for position, sample_id in enumerate(dataset.sample_ids.tolist())}
    conv_positions = torch.tensor(
        [position_by_id[int(i)] for i in conv.sample_ids],
        dtype=torch.long, device=view.u0_train_validation.device)
    selection_lookup = {
        int(position): local for local, position in enumerate(selection_positions.tolist())}
    conv_in_selection = torch.tensor(
        [selection_lookup[int(position)] for position in conv_positions.tolist()],
        dtype=torch.long, device=view.u0_train_validation.device)
    local_train = view.train_indices
    local_val = view.validation_indices
    rows, summary = [], {"families": {}}
    bases = []
    for family, representative in representatives.items():
        states, features = {}, {}
        finest = candidates[-1]
        for nx in candidates:
            local_shard = (
                torch.arange(selection_positions.numel(), device=local_train.device)
                if nx == finest
                else conv_in_selection
            )
            global_shard = selection_positions if nx == finest else conv_positions
            state, _, digest = _solve_state(
                config, view.u0_train_validation[local_shard], dataset, cache, family,
                representative["nu_star"], representative["T_star"], nx,
                batch_size, global_shard)
            feature, _ = _features(config, state, digest, cache)
            states[nx] = state.to(view.u0_train_validation.device)
            features[nx] = feature.to(view.u0_train_validation.device)
        frozen_models, _, _ = _fit_point(config, TrainValidationData(
            x_train=features[finest][local_train],
            x_validation=features[finest][local_val],
            y_train=view.target_train,
            y_validation=view.target_validation,
            target_data_validation=view.target_data_validation,
            reference_validation=view.reference_validation))
        frozen2 = frozen_models["model2"]
        selected3 = frozen_models["model3"]
        frozen3 = {}
        for seed in config.e2.model3.evaluation_seeds:
            random_map = RandomFeatureMap.create(
                features[finest].shape[1], selected3["width"],
                activation=config.e2.model3.activation, seed=seed,
                weight_scale=selected3["weight_scale"],
                bias_scale=selected3["bias_scale"],
                dtype=features[finest].dtype,
                device=features[finest].device)
            augmented = random_map(features[finest][local_train])
            readout, _, _ = select_ridge(
                augmented, view.target_train, augmented,
                view.target_train, (float(selected3["zeta"]),),
                tolerance=0.0, svd_rcond=config.e2.ridge.svd_rcond)
            frozen3[seed] = (random_map, readout)
        passing = []
        for nx in candidates:
            nx_state = states[nx] if nx != finest else states[nx][conv_in_selection]
            nx_feature = features[nx] if nx != finest else features[nx][conv_in_selection]
            finest_state = states[finest][conv_in_selection]
            finest_feature = features[finest][conv_in_selection]
            terminal = compare_fields_on_common_grid(
                nx_state, finest_state, common_nx=finest,
                domain_length=L)["relative_aggregate"]
            feature_err = samplewise_l2_errors(
                nx_feature, finest_feature, domain_length=float(J))["relative"]
            feature = aggregate_errors(feature_err)
            prediction_sets = []
            named_prediction = {}
            for model_name, pred, pred_ref in (
                ("model1", decode_equispaced_point_observation_to_real_fourier(nx_feature, config.spatial.target_output_dim, domain_length=L),
                 decode_equispaced_point_observation_to_real_fourier(finest_feature, config.spatial.target_output_dim, domain_length=L)),
                ("model2", frozen2(nx_feature), frozen2(finest_feature)),
            ):
                pred_field = real_fourier_synthesis(pred, config.spatial.reference_nx, domain_length=L)
                pred_ref_field = real_fourier_synthesis(pred_ref, config.spatial.reference_nx, domain_length=L)
                evidence = aggregate_errors(samplewise_l2_errors(
                    pred_field, pred_ref_field,
                    domain_length=L)["relative"])
                prediction_sets.append(evidence)
                named_prediction[model_name] = evidence
            model3_seed_evidence = {}
            for seed, (random_map, readout) in frozen3.items():
                pred = readout(random_map(nx_feature))
                pred_ref = readout(random_map(finest_feature))
                pred_field = real_fourier_synthesis(
                    pred, config.spatial.reference_nx, domain_length=L)
                pred_ref_field = real_fourier_synthesis(
                    pred_ref, config.spatial.reference_nx, domain_length=L)
                evidence = aggregate_errors(samplewise_l2_errors(
                    pred_field, pred_ref_field,
                    domain_length=L)["relative"])
                prediction_sets.append(evidence)
                model3_seed_evidence[str(seed)] = evidence
            prediction = {"mean": max(item["mean"] for item in prediction_sets),
                          "max": max(item["max"] for item in prediction_sets)}
            worst_seed = max(
                model3_seed_evidence,
                key=lambda seed: model3_seed_evidence[seed]["max"])
            tol = conv.tolerances
            passed = terminal["mean"] <= tol.terminal_mean and terminal["max"] <= tol.terminal_max and \
                feature["mean"] <= tol.feature_mean and feature["max"] <= tol.feature_max and \
                prediction["mean"] <= tol.prediction_mean and prediction["max"] <= tol.prediction_max
            rows.append({"family": family, "n_sur": nx, "reference_n_sur": finest,
                         "sample_ids": ",".join(str(i) for i in conv.sample_ids),
                         "sample_membership": "train_or_validation",
                         "terminal_relative_l2_mean": terminal["mean"], "terminal_relative_l2_max": terminal["max"],
                         "feature_relative_l2_mean": feature["mean"], "feature_relative_l2_max": feature["max"],
                         "prediction_relative_l2_mean": prediction["mean"], "prediction_relative_l2_max": prediction["max"],
                         "model1_prediction_mean": named_prediction["model1"]["mean"],
                         "model1_prediction_max": named_prediction["model1"]["max"],
                         "model1_prediction_pass": (
                             named_prediction["model1"]["mean"] <= tol.prediction_mean
                             and named_prediction["model1"]["max"] <= tol.prediction_max),
                         "model2_prediction_mean": named_prediction["model2"]["mean"],
                         "model2_prediction_max": named_prediction["model2"]["max"],
                         "model2_prediction_pass": (
                             named_prediction["model2"]["mean"] <= tol.prediction_mean
                             and named_prediction["model2"]["max"] <= tol.prediction_max),
                         "model3_worst_seed": worst_seed,
                         "model3_prediction_mean": max(
                             item["mean"] for item in model3_seed_evidence.values()),
                         "model3_prediction_max": max(
                             item["max"] for item in model3_seed_evidence.values()),
                         "model3_prediction_pass": all(
                             item["mean"] <= tol.prediction_mean
                             and item["max"] <= tol.prediction_max
                             for item in model3_seed_evidence.values()),
                         "model3_evaluation_seed_evidence":
                             __import__("json").dumps(model3_seed_evidence, sort_keys=True),
                         "model3_aggregate_type": "worst_case_over_evaluation_seeds_and_models",
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
