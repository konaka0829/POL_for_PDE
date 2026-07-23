"""Paper 1 E2 parameter/time selection workflow."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from scipy.stats import t as student_t

from .config import Paper1Config, canonical_config_json
from .datasets import Paper1MasterDataset
from .grids import spectral_resample_periodic
from .interfaces import build_surrogate_initial_state, derive_finite_resolution_data
from .metrics import aggregate_errors, compare_fields_on_common_grid, samplewise_l2_errors
from .model1 import decode_equispaced_point_observation_to_real_fourier
from .observations import observe_equispaced_periodic
from .random_features import RandomFeatureMap
from .readouts import AffineReadout, fit_centered_affine_ridge
from .solvers import solve_burgers_final_state, solve_reaction_diffusion_final_state
from .target_representation import real_fourier_synthesis

E2_SCHEMA_VERSION = "paper1-e2-v1"
MODELS = ("model1", "model2", "model3")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.numpy().tobytes())
    return h.hexdigest()


class TensorCache:
    """Content-addressed tensor cache with read-back hash verification."""

    def __init__(self, root: Path, *, resume: bool):
        self.root, self.resume = root, resume
        self.hits = self.misses = 0

    def get_or_compute(self, kind: str, key: dict[str, Any], compute) -> tuple[torch.Tensor, dict[str, Any], str]:
        digest = stable_hash({"schema": E2_SCHEMA_VERSION, "kind": kind, **key})
        directory = self.root / kind
        path, meta_path = directory / f"{digest}.pt", directory / f"{digest}.json"
        if path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta["key"] == key and hashlib.sha256(path.read_bytes()).hexdigest() == meta["file_sha256"]:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    if tensor_hash(payload["values"]) == meta["tensor_hash"]:
                        self.hits += 1
                        return payload["values"], payload["solver_metadata"], digest
            except Exception:
                pass
            if self.resume:
                raise ValueError(f"resume cache integrity check failed: {path}")
        values, solver_metadata = compute()
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"values": values.detach().cpu(), "solver_metadata": solver_metadata}, path)
        meta = {"key": key, "tensor_hash": tensor_hash(values), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self.misses += 1
        return values.detach().cpu(), solver_metadata, digest


def select_first_with_tolerance(rows: list[dict[str, Any]], metric: str, tolerance: float) -> dict[str, Any]:
    """Validation-only parameter selection preserving config/grid order."""
    if not rows:
        raise ValueError("cannot select from empty validation rows")
    best = min(float(row[metric]) for row in rows)
    return next(row for row in rows if float(row[metric]) <= best + tolerance)


def select_ridge(
    x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
    zetas: tuple[float, ...], *, tolerance: float, svd_rcond: float | None,
) -> tuple[AffineReadout, float, list[dict[str, Any]]]:
    candidates = []
    for order, zeta in enumerate(zetas):
        model = fit_centered_affine_ridge(x_train, y_train, zeta, svd_rcond=svd_rcond)
        mse = float(torch.mean((model(x_val) - y_val) ** 2))
        candidates.append({"order": order, "zeta": zeta, "validation_coefficient_mse": mse, "model": model})
    best = min(row["validation_coefficient_mse"] for row in candidates)
    tied = [row for row in candidates if row["validation_coefficient_mse"] <= best + tolerance]
    selected = max(tied, key=lambda row: row["zeta"])
    provenance = [{k: v for k, v in row.items() if k != "model"} for row in candidates]
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
) -> dict[str, Any]:
    assert config.e2 is not None
    e2 = config.e2
    solver = asdict(e2.burgers) if family == "burgers" else asdict(e2.reaction_diffusion)
    return {
        "dataset_hash": dataset.metadata["dataset_hash"], "split_hash": dataset.metadata["split_hash"],
        "sample_hash": tensor_hash(dataset.sample_ids), "n_tar": config.spatial.target_data_nx,
        "input_interpolation": "periodic_spectral", "n_sur": n_sur, "family": family,
        "nu_tilde": nu, "T_tilde": T, "solver": solver, "L": config.domain.length,
        "dtype": config.data.dtype, "device": config.data.device,
    }


def _solve_state(
    config: Paper1Config, u0_data: torch.Tensor, dataset: Paper1MasterDataset, cache: TensorCache,
    family: str, nu: float, T: float, n_sur: int, batch_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    assert config.e2 is not None
    L, e2 = config.domain.length, config.e2
    u0_sur = build_surrogate_initial_state(u0_data, surrogate_internal_nx=n_sur, domain_length=L)
    key = _state_key(config, dataset, family, nu, T, n_sur)
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
    config: Paper1Config, features: torch.Tensor, targets: torch.Tensor, split: dict[str, torch.Tensor],
    validation_reference_master: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Fit all models using train/validation arrays only."""
    assert config.e2 is not None
    e2, L, q = config.e2, config.domain.length, config.spatial.target_output_dim
    train, val = split["train"], split["val"]
    models: dict[str, Any] = {"model1": None}
    selections: dict[str, Any] = {"model1": {"kind": "fixed_decoder"}}
    provenance: list[dict[str, Any]] = []
    model2, zeta2, ridge_rows = select_ridge(
        features[train], targets[train], features[val], targets[val], e2.ridge.zetas,
        tolerance=e2.ridge.tie_tolerance, svd_rcond=e2.ridge.svd_rcond)
    models["model2"], selections["model2"] = model2, {"zeta": zeta2, "ridge_candidates": ridge_rows}
    candidates = []
    for order, (width, ws, bs) in enumerate(itertools.product(
            e2.model3.widths, e2.model3.weight_scales, e2.model3.bias_scales)):
        seed_metrics, seed_models = [], {}
        for seed in e2.model3.selection_seeds:
            random_map = RandomFeatureMap.create(features.shape[1], width, activation=e2.model3.activation,
                                                 seed=seed, weight_scale=ws, bias_scale=bs,
                                                 dtype=features.dtype, device=features.device)
            augmented = random_map(features)
            readout, zeta, _ = select_ridge(
                augmented[train], targets[train], augmented[val], targets[val], e2.ridge.zetas,
                tolerance=e2.ridge.tie_tolerance, svd_rcond=e2.ridge.svd_rcond)
            pred = readout(augmented[val])
            field = real_fourier_synthesis(pred, config.spatial.reference_nx, domain_length=L)
            metric = float(samplewise_l2_errors(field, validation_reference_master, domain_length=L)["relative"].mean())
            seed_metrics.append(metric); seed_models[seed] = (random_map, readout, zeta)
            provenance.append({"candidate_order": order, "width": width, "weight_scale": ws,
                               "bias_scale": bs, "seed": seed, "validation_field_relative_l2_mean": metric,
                               "selected_zeta": zeta})
        candidates.append({"order": order, "width": width, "weight_scale": ws, "bias_scale": bs,
                           "validation_field_relative_l2_mean": sum(seed_metrics) / len(seed_metrics),
                           "seed_models": seed_models})
    best_value = min(row["validation_field_relative_l2_mean"] for row in candidates)
    tied = [row for row in candidates if row["validation_field_relative_l2_mean"] <= best_value + e2.parameter_tie_tolerance]
    selected = min(tied, key=lambda row: (row["width"], -max(m[2] for m in row["seed_models"].values()),
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
) -> dict[str, Any]:
    """Run E2 with validation-only selection followed by frozen test evaluation."""
    if config.e2 is None:
        raise ValueError("config must contain e2")
    if dataset.y_target_master is None:
        raise ValueError("dataset y_target_master is required")
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
                config, finite.u0_data, dataset, cache, family, nu, T, pilot_n_sur, batch_size)
            features, feature_digest = _features(config, state, state_digest, cache)
            models, selections, provenance = _fit_point(
                config, features.to(device), finite.target_coefficients, split, yref[split["val"]])
            point_models[identity] = {"models": models, "features": features.to(device), "solver_metadata": solver_meta,
                                      "state_digest": state_digest, "feature_digest": feature_digest}
            point_selections[identity] = selections; model3_validation.extend(
                [{"family": family, "sweep_axis": axis, "nu_tilde": nu, "T_tilde": T, **row} for row in provenance])
            for model_name in MODELS:
                if model_name == "model1":
                    pred = decode_equispaced_point_observation_to_real_fourier(features[split["val"]], config.spatial.target_output_dim, domain_length=config.domain.length)
                elif model_name == "model2":
                    pred = models["model2"](features[split["val"]])
                else:
                    seed_predictions = [readout(random_map(features[split["val"]])) for random_map, readout, _ in models["model3"]["seed_models"].values()]
                    pred = torch.stack(seed_predictions).mean(0)
                metrics = _metric_row(pred, finite.target_coefficients[split["val"]], yref[split["val"]],
                                      finite.y_target_data[split["val"]], n_tar=config.spatial.target_data_nx,
                                      n_ref=config.spatial.reference_nx, L=config.domain.length)
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
    for family, family_cfg in (("burgers", e2.burgers), ("reaction_diffusion", e2.reaction_diffusion)):
        for nu in family_cfg.nu_grid:
            evaluate_validation_point(family, "nu_tilde", nu, family_cfg.initial_T_anchor)
        nu_rows = [row for row in validation_rows if row["family"] == family and row["sweep_axis"] == "nu_tilde" and row["model"] == e2.representative_model]
        nu_selected = select_first_with_tolerance(nu_rows, e2.selection_metric, e2.parameter_tie_tolerance)
        nu_star = float(nu_selected["nu_tilde"])
        for T in family_cfg.T_grid:
            evaluate_validation_point(family, "T_tilde", nu_star, T)
        time_rows = [row for row in validation_rows if row["family"] == family and row["sweep_axis"] == "T_tilde" and row["model"] == e2.representative_model]
        time_selected = select_first_with_tolerance(time_rows, e2.selection_metric, e2.parameter_tie_tolerance)
        T_star = float(time_selected["T_tilde"])
        for round_index in range(e2.coordinate_refinement_rounds):
            nu_axis = f"nu_refinement_{round_index + 1}"
            for nu in family_cfg.nu_grid:
                evaluate_validation_point(family, nu_axis, nu, T_star)
            refine_nu_rows = [row for row in validation_rows if row["family"] == family and row["sweep_axis"] == nu_axis
                              and row["model"] == e2.representative_model]
            nu_star = float(select_first_with_tolerance(
                refine_nu_rows, e2.selection_metric, e2.parameter_tie_tolerance)["nu_tilde"])
            time_axis = f"T_refinement_{round_index + 1}"
            for T in family_cfg.T_grid:
                evaluate_validation_point(family, time_axis, nu_star, T)
            refine_time_rows = [row for row in validation_rows if row["family"] == family and row["sweep_axis"] == time_axis
                                and row["model"] == e2.representative_model]
            T_star = float(select_first_with_tolerance(
                refine_time_rows, e2.selection_metric, e2.parameter_tie_tolerance)["T_tilde"])
        representatives[family] = {"nu_star": nu_star, "T_star": T_star, "representative_model": e2.representative_model,
                                   "selection_metric": e2.selection_metric}

    model_specific: dict[str, Any] = {}
    for family in representatives:
        model_specific[family] = {}
        for model in MODELS:
            rows = [row for row in validation_rows if row["family"] == family and row["model"] == model]
            selected = select_first_with_tolerance(rows, e2.selection_metric, e2.parameter_tie_tolerance)
            model_specific[family][model] = {"nu_tilde": selected["nu_tilde"], "T_tilde": selected["T_tilde"],
                                             "validation_field_relative_l2_mean": selected[e2.selection_metric]}
    selection_record = {"schema_version": E2_SCHEMA_VERSION, "representatives": representatives,
                        "model_specific_optima": model_specific,
                        "point_hyperparameters": {str(key): value for key, value in point_selections.items()},
                        "test_data_used": False}
    selection_hash = stable_hash(selection_record)
    shared_hyperparameters = {}
    for family, representative in representatives.items():
        final_axis = "T_tilde" if e2.coordinate_refinement_rounds == 0 else f"T_refinement_{e2.coordinate_refinement_rounds}"
        identity = (family, final_axis, representative["nu_star"], representative["T_star"])
        shared_hyperparameters[family] = point_selections[identity]

    # Test data is first accessed below, after the selection record is frozen.
    test_rows, model3_test_by_seed, model3_aggregate, saved_models = [], [], [], {}
    test = split["test"]
    for identity in point_order:
        family, axis, nu, T = identity
        bundle, selections = point_models[identity], point_selections[identity]
        features = bundle["features"]
        for model_name in MODELS:
            seed_metrics = []
            if model_name == "model1":
                predictions = [(None, decode_equispaced_point_observation_to_real_fourier(
                    features[test], config.spatial.target_output_dim, domain_length=config.domain.length))]
            elif model_name == "model2":
                predictions = [(None, bundle["models"]["model2"](features[test]))]
            else:
                selected = bundle["models"]["model3"]
                predictions = []
                for seed in e2.model3.evaluation_seeds:
                    random_map = RandomFeatureMap.create(features.shape[1], selected["width"], activation=e2.model3.activation,
                                                         seed=seed, weight_scale=selected["weight_scale"],
                                                         bias_scale=selected["bias_scale"], dtype=features.dtype, device=features.device)
                    augmented = random_map(features)
                    readout, zeta, _ = select_ridge(augmented[split["train"]], finite.target_coefficients[split["train"]],
                                                   augmented[split["val"]], finite.target_coefficients[split["val"]],
                                                   e2.ridge.zetas, tolerance=e2.ridge.tie_tolerance, svd_rcond=e2.ridge.svd_rcond)
                    predictions.append((seed, readout(augmented[test])))
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
                "candidate": {k: model3[k] for k in ("width", "weight_scale", "bias_scale")},
                "selection_seed_models": {
                    str(seed): {"A": random_map.A.detach().cpu(), "c": random_map.c.detach().cpu(),
                                "W": readout.W.detach().cpu(), "b": readout.b.detach().cpu(), "zeta": zeta}
                    for seed, (random_map, readout, zeta) in model3["seed_models"].items()
                },
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
                           auto_reruns_remaining=auto_reruns_remaining - 1, batch_size=batch_size)
            rerun["auto_rerun_history"] = [{"from_n_sur": pilot_n_sur, "to_n_sur": selected_base},
                                           *rerun.get("auto_rerun_history", [])]
            return rerun
    runtime = time.perf_counter() - start
    return {
        "validation_sweep": validation_rows, "test_sweep": test_rows,
        "model3_validation_by_seed": model3_validation, "model3_test_by_seed": model3_test_by_seed,
        "model3_test_aggregate": model3_aggregate, "model_specific_optima": model_specific,
        "shared_representatives": representatives, "selection_record": selection_record,
        "selection_record_hash": selection_hash, "convergence_results": convergence_rows,
        "convergence_summary": convergence_summary, "failed_runs": failed_runs,
        "selected_models": saved_models, "cache": {"hits": cache.hits, "misses": cache.misses},
        "runtime_seconds": runtime, "finite_data": finite,
        "pilot_n_sur": pilot_n_sur, "auto_rerun_history": [],
        "shared_hyperparameters": shared_hyperparameters,
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
    ids = torch.tensor(conv.sample_ids, dtype=torch.long, device=u0_data.device)
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
        frozen_models, _, _ = _fit_point(
            config, features[finest], targets, split, validation_reference_master)
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
            if passed: passing.append(nx)
        base = min(passing) if passing else None
        summary["families"][family] = {"n_sur_base": base, "status": "pass" if base is not None else "fail"}
        if base is not None: bases.append(base)
    summary["global_n_sur_base"] = max(bases) if len(bases) == len(representatives) else None
    summary["status"] = "pass" if summary["global_n_sur_base"] is not None else "fail"
    summary["sample_ids"] = list(conv.sample_ids)
    return rows, summary
