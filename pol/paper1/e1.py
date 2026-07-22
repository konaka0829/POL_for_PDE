from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .config import Paper1Config, config_from_dict
from .datasets import tensor_hash
from .e0 import E0_SCHEMA_VERSION, MASTER_SCHEMA_VERSION, load_master_initial_conditions
from .grids import spectral_resample_periodic
from .heat import heat_multiplier_vector, heat_regime, solve_heat_exact
from .initial_conditions import resolve_device
from .readouts import fit_centered_affine_ridge, l2_analysis_matrix, l2_synthesis_matrix
from .schemas import stable_hash_json
from .target_representation import real_fourier_analysis, real_fourier_synthesis

E1_SCHEMA_VERSION = "paper1-e1-v2"
E0_REQUIRED = (
    "e0_summary.json", "resampling_checks.json", "input_interface_checks.json",
    "model1_identity.json", "reference_convergence.json",
    "master_initial_conditions.pt", "master_manifest.json", "resolved_config.json",
    "accepted_production_config.json",
)
E0_REQUIRED_CHECKS = {
    "resampling", "fourier_projector", "reference_spatial_convergence",
    "reference_temporal_convergence", "reference_joint_convergence",
    "finite_data_interface", "no_high_frequency_leak",
    "target_coefficient_consistency", "model1_full_observation_identity",
    "model1_bandlimited_reduced_j", "model1_aliasing_counterexample",
}


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    return {"relative_path": str(path.relative_to(root)) if root else str(path), "byte_size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def validate_e0_prerequisite(e0_dir: str | Path, config: Paper1Config) -> tuple[Paper1Config, Any, dict[str, Any]]:
    root = Path(e0_dir)
    missing = [name for name in E0_REQUIRED if not (root / name).is_file()]
    if missing:
        raise ValueError("E0 prerequisite missing artifact: " + missing[0])
    artifacts = [file_record(root / name, root) for name in E0_REQUIRED]
    if (root / "environment.json").is_file(): artifacts.append(file_record(root / "environment.json", root))
    summary = json.loads((root / "e0_summary.json").read_text())
    if summary.get("schema_version") != E0_SCHEMA_VERSION or summary.get("status") != "pass":
        raise ValueError("E0 prerequisite is not a passing known-schema E0 run")
    required = summary.get("required_checks")
    if not isinstance(required, dict):
        raise ValueError("E0 prerequisite required_checks is missing")
    missing_checks = sorted(E0_REQUIRED_CHECKS - set(required))
    if missing_checks:
        raise ValueError("E0 prerequisite missing required check: " + missing_checks[0])
    failed_checks = sorted(name for name in E0_REQUIRED_CHECKS if required[name] != "pass")
    if failed_checks:
        raise ValueError("E0 prerequisite required check is not pass: " + failed_checks[0])
    documents = {
        name: json.loads((root / name).read_text())
        for name in ("resampling_checks.json", "input_interface_checks.json", "model1_identity.json", "reference_convergence.json")
    }
    for name, document in documents.items():
        if document.get("schema_version") != E0_SCHEMA_VERSION:
            raise ValueError(f"E0 prerequisite {name} schema_version mismatch")
    resampling = documents["resampling_checks.json"]
    if resampling.get("status") != "pass" or resampling.get("fourier_projector", {}).get("status") != "pass":
        raise ValueError("E0 resampling/projector status mismatch")
    interfaces = documents["input_interface_checks.json"]
    if interfaces.get("status") != "pass" or any(
        interfaces.get(name, {}).get("status") != "pass"
        for name in ("finite_data_interface", "no_high_frequency_leak", "target_coefficient_consistency")
    ):
        raise ValueError("E0 input interface nested status mismatch")
    model1 = documents["model1_identity.json"]
    if model1.get("status") != "pass" or any(
        model1.get(name, {}).get("status") != "pass"
        for name in ("full_observation", "bandlimited_reduced_j", "aliasing_counterexample")
    ):
        raise ValueError("E0 Model 1 nested status mismatch")
    convergence = documents["reference_convergence.json"]
    joint_row = convergence.get("joint_row")
    if convergence.get("joint_status") != "pass" or not isinstance(joint_row, dict) or joint_row.get("status") != "pass":
        raise ValueError("E0 reference convergence joint status mismatch")
    if summary.get("selected_reference", {}).get("joint_status") != "pass":
        raise ValueError("E0 summary selected reference joint status mismatch")
    selected = summary.get("selected_reference", {})
    reference_nx = selected.get("reference_nx")
    if not isinstance(reference_nx, int) or reference_nx < config.spatial.target_data_nx:
        raise ValueError("E0 selected reference_nx is missing or below target_data_nx")
    manifest = json.loads((root / "master_manifest.json").read_text())
    if manifest.get("schema_version") != MASTER_SCHEMA_VERSION:
        raise ValueError("unknown E0 master archive schema")
    payload = torch.load(root / "master_initial_conditions.pt", map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("E0 master payload must be a dictionary")
    meta = payload.get("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError("E0 master payload metadata must be a dictionary")
    actual_hash = tensor_hash(payload.get("values")) if isinstance(payload.get("values"), torch.Tensor) else None
    if actual_hash != meta.get("tensor_hash"):
        raise ValueError("actual tensor hash does not match payload metadata")
    if actual_hash != manifest.get("tensor_hash"):
        raise ValueError("actual tensor hash does not match master manifest")
    metadata_keys = (
        "domain_length", "seed", "maximum_nx", "dtype", "grf_gamma",
        "grf_tau", "grf_sigma", "grf_mean", "tensor_hash",
    )
    if any(meta.get(key) != manifest.get(key) for key in metadata_keys):
        raise ValueError("master manifest/payload metadata mismatch")
    ids = payload.get("sample_ids")
    if not isinstance(ids, torch.Tensor) or ids.tolist() != manifest.get("sample_ids"):
        raise ValueError("master manifest/payload sample IDs mismatch")
    if manifest.get("sample_count") != ids.numel():
        raise ValueError("master sample count mismatch")
    e0_cfg = json.loads((root / "resolved_config.json").read_text())
    for path in (("domain","length"),("data","total_samples"),("data","n_train"),("data","n_val"),("data","n_test"),("data","seed"),("data","dtype"),("data","ic_type"),("data","grf_gamma"),("data","grf_tau"),("data","grf_sigma"),("data","grf_mean"),("data","preprocessing")):
        e0_value = e0_cfg
        e1_value = config.to_dict()
        for key in path:
            e0_value = e0_value[key]
            e1_value = e1_value[key]
        if e0_value != e1_value:
            raise ValueError("E0/E1 config mismatch: " + ".".join(path))
    maximum_nx = manifest.get("maximum_nx")
    if not isinstance(maximum_nx, int):
        raise ValueError("master maximum_nx is missing or invalid")
    if reference_nx > maximum_nx or reference_nx < config.spatial.target_data_nx:
        raise ValueError("invalid selected reference resolution")
    accepted_raw = json.loads((root / "accepted_production_config.json").read_text())
    accepted = config_from_dict(accepted_raw)
    if accepted.e0 is not None or accepted.e1 is not None:
        raise ValueError("accepted production config must not contain e0/e1")
    if summary.get("accepted_production_config") != "accepted_production_config.json":
        raise ValueError("E0 summary accepted production config filename mismatch")
    selected_summary = summary["selected_reference"]
    if not (
        reference_nx == joint_row.get("candidate_nx") == accepted.spatial.reference_nx
    ):
        raise ValueError("E0 selected reference resolution artifacts disagree")
    selected_pairs = (
        ("solver", "solver"), ("requested_dt", "dt"),
        ("requested_fine_dt", "fine_dt"),
    )
    for artifact_key, target_key in selected_pairs:
        values = (
            selected_summary.get(artifact_key), joint_row.get(artifact_key),
            getattr(accepted.target, target_key),
        )
        if values[0] != values[1] or values[1] != values[2]:
            raise ValueError(f"E0 selected temporal setting mismatch: {artifact_key}")
    if selected_summary.get("effective_inner_step") != joint_row.get("effective_inner_step"):
        raise ValueError("E0 effective inner step mismatch")
    for path in (("domain", "length"),) + tuple(("data", key) for key in (
        "total_samples", "n_train", "n_val", "n_test", "seed", "dtype", "ic_type",
        "grf_gamma", "grf_tau", "grf_sigma", "grf_mean", "preprocessing",
    )):
        resolved_value, accepted_value = e0_cfg, accepted_raw
        for key in path:
            resolved_value, accepted_value = resolved_value[key], accepted_value[key]
        if resolved_value != accepted_value:
            raise ValueError("accepted production config mismatch: " + ".".join(path))
    calibration_ids = e0_cfg["e0"]["calibration_sample_ids"]
    if joint_row.get("master_hash") != actual_hash or joint_row.get("sample_ids") != calibration_ids:
        raise ValueError("E0 joint convergence master/sample IDs mismatch")
    matching_joint_rows = [row for row in convergence.get("rows", []) if row == joint_row]
    if len(matching_joint_rows) != 1:
        raise ValueError("E0 joint row is not uniquely represented in convergence rows")
    effective = replace(config, spatial=replace(config.spatial, reference_nx=reference_nx))
    master = load_master_initial_conditions(root / "master_initial_conditions.pt", effective)
    prerequisite = {
        "schema_version": "paper1-e1-prerequisite-v2", "status": "pass", "e0_dir": str(root),
        "user_requested_reference_nx": config.spatial.reference_nx, "e0_selected_reference_nx": reference_nx, "effective_reference_nx": reference_nx, "master_maximum_nx": maximum_nx,
        "required_e0_checks": {k: required[k] for k in sorted(E0_REQUIRED_CHECKS)},
        "cross_checks": {"actual_vs_payload_tensor_hash":{"status":"pass"},"actual_vs_manifest_tensor_hash":{"status":"pass"},"manifest_vs_payload_metadata":{"status":"pass"}},
        "artifacts": artifacts, "master_tensor_hash": actual_hash,
        "master_file_sha256": file_record(root / "master_initial_conditions.pt", root)["sha256"],
        "master_manifest_file_sha256": file_record(root / "master_manifest.json", root)["sha256"],
    }
    return effective, master, prerequisite


def _stats(v: torch.Tensor) -> dict[str, float]:
    v = v.detach().cpu()
    return {"mean": float(v.mean()), "median": float(v.median()), "max": float(v.max())}


def _relative(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - truth, dim=-1) / torch.clamp(torch.linalg.vector_norm(truth, dim=-1), min=torch.finfo(truth.dtype).eps)


def build_surrogate_from_finite_target(u_tar: torch.Tensor, n_sur: int, domain_length: float) -> torch.Tensor:
    """The audited finite boundary: this API cannot receive reference fields."""
    return spectral_resample_periodic(u_tar, n_sur, domain_length=domain_length)


def finite_input_path_check(n_ref: int, n_tar: int, n_sur: int, domain_length: float, *, dtype=torch.float64, device="cpu") -> dict[str, Any]:
    if n_ref < 2 * n_tar:
        n_ref = 2 * n_tar
    x=torch.arange(n_ref,dtype=dtype,device=device)*domain_length/n_ref
    low=.4+torch.cos(2*torch.pi*2*x/domain_length)
    high_k=n_tar//2+1
    pair=torch.stack((low,low+.3*torch.cos(2*torch.pi*high_k*x/domain_length)))
    tar=spectral_resample_periodic(pair,n_tar,domain_length=domain_length)
    sur=build_surrogate_from_finite_target(tar,n_sur,domain_length)
    error=float(torch.max(torch.abs(sur[0]-sur[1])))
    tolerance=100*torch.finfo(dtype).eps
    return {"status":"pass" if error <= tolerance else "fail","max_abs_error":error,"tolerance":tolerance,"high_wavenumber":high_k}


def _split_indices(config: Paper1Config, device: torch.device) -> dict[str, torch.Tensor]:
    """Reproduce the E0/data split convention without consulting test labels."""
    generator = torch.Generator(device="cpu").manual_seed(config.data.seed)
    permutation = torch.randperm(config.data.total_samples, generator=generator)
    n_train = config.data.n_train
    n_val = config.data.n_val
    return {
        "train": permutation[:n_train].to(device),
        "val": permutation[n_train : n_train + n_val].to(device),
        "test": permutation[n_train + n_val :].to(device),
    }


def heat_algebraic_error(config: Paper1Config, device: torch.device) -> float:
    """Measure exact heat propagation for constant, cosine, and sine modes."""
    dtype = config.data.torch_dtype()
    length = config.domain.length
    nx = config.spatial.target_data_nx
    x = torch.arange(nx, dtype=dtype, device=device) * length / nx
    kappa = 2.0 * torch.pi / length
    fields = torch.stack((torch.ones_like(x), torch.cos(kappa * x), torch.sin(kappa * x)))
    attenuation = math.exp(-config.target.nu * config.target.T * float(kappa**2))
    expected = fields.clone()
    expected[1:] *= attenuation
    actual = solve_heat_exact(
        fields,
        nu=config.target.nu,
        T=config.target.T,
        domain_length=length,
    )
    return float(torch.max(torch.abs(actual - expected)))


def select_ridge_readout(
    train_features: torch.Tensor,
    train_truth: torch.Tensor,
    validation_features: torch.Tensor,
    validation_truth: torch.Tensor,
    *,
    zetas: tuple[float, ...],
    tie_tolerance: float,
    svd_rcond: float | None,
) -> tuple[float, Any, list[dict[str, Any]]]:
    """Fit/select using only train and validation arrays; no test API exists."""
    candidates = []
    rows = []
    for zeta in zetas:
        model = fit_centered_affine_ridge(
            train_features, train_truth, zeta, svd_rcond=svd_rcond,
        )
        train_mse = float(torch.mean((model(train_features) - train_truth) ** 2))
        validation_mse = float(torch.mean((model(validation_features) - validation_truth) ** 2))
        rows.append({
            "zeta": zeta, "train_coefficient_mse": train_mse,
            "validation_coefficient_mse": validation_mse, "selected": False,
        })
        candidates.append((validation_mse, float(zeta), model))
    best_value = min(candidate[0] for candidate in candidates)
    eligible = [candidate for candidate in candidates if candidate[0] <= best_value + tie_tolerance]
    _, selected_zeta, selected_model = max(eligible, key=lambda candidate: candidate[1])
    for row in rows:
        row["selected"] = float(row["zeta"]) == selected_zeta
    return selected_zeta, selected_model, rows


def run_e1(config: Paper1Config, master: Any) -> dict[str, Any]:
    assert config.e1 is not None
    e1, L, dtype = config.e1, config.domain.length, config.data.torch_dtype()
    device = resolve_device(config.data.device)
    u_ref = spectral_resample_periodic(master.values_master.to(device=device, dtype=dtype), config.spatial.reference_nx, domain_length=L)
    u_tar = spectral_resample_periodic(u_ref, config.spatial.target_data_nx, domain_length=L)
    # This boundary receives only finite n_tar values; discarded reference modes cannot leak downstream.
    u_sur0 = build_surrogate_from_finite_target(u_tar, config.spatial.surrogate_internal_nx, L)
    y_ref = solve_heat_exact(u_ref, nu=config.target.nu, T=config.target.T, domain_length=L)
    y_tar = spectral_resample_periodic(y_ref, config.spatial.target_data_nx, domain_length=L)
    ntr = config.data.n_train
    splits = _split_indices(config, device)
    ridge_rows: list[dict[str, Any]] = []; selected_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []; mode_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []; models: dict[str, Any] = {}
    max_coeff_diff = 0.0
    max_fourier_coordinate_error = 0.0
    max_ideal_coordinate_error = 0.0
    noise_zero_diff = 0.0
    for case in e1.surrogate_cases:
        regime, delta = heat_regime(target_nu=config.target.nu, target_T=config.target.T, surrogate_nu=case.nu, surrogate_T=case.T)
        r_sur = solve_heat_exact(u_sur0, nu=case.nu, T=case.T, domain_length=L)
        observed = spectral_resample_periodic(r_sur, config.spatial.observation_dim, domain_length=L)
        features = observed * math.sqrt(L / config.spatial.observation_dim)
        for q in e1.output_dims:
            truth = real_fourier_analysis(y_tar, q, domain_length=L)
            ref_coeff = real_fourier_analysis(y_ref, q, domain_length=L)
            max_coeff_diff = max(max_coeff_diff, float(torch.max(torch.abs(truth - ref_coeff))))
            selected_zeta, model, selection_rows = select_ridge_readout(
                features[splits["train"]], truth[splits["train"]],
                features[splits["val"]], truth[splits["val"]],
                zetas=e1.ridge_zetas, tie_tolerance=e1.ridge_tie_tolerance,
                svd_rcond=e1.ridge_svd_rcond,
            )
            ridge_rows.extend({"case_name": case.name, "regime": regime, "q": q, **row} for row in selection_rows)
            models[f"{case.name}/q{q}"] = {
                "W": model.W.detach().cpu(), "b": model.b.detach().cpu(),
                "zeta": selected_zeta, "regime": regime, "delta": delta,
                "solver": model.solver, "svd_rcond": model.svd_rcond,
                "singular_value_cutoff": model.singular_value_cutoff,
                "solver_numerical_rank": model.numerical_rank,
            }
            D = l2_analysis_matrix(q, config.spatial.observation_dim, domain_length=L, dtype=dtype, device=device)
            S = l2_synthesis_matrix(q, config.spatial.observation_dim, domain_length=L, dtype=dtype, device=device)
            mult = heat_multiplier_vector(q, target_nu=config.target.nu, target_T=config.target.T, surrogate_nu=case.nu, surrogate_T=case.T, domain_length=L, dtype=dtype, device=device)
            ideal = torch.diag(mult) @ D
            effective = model.W @ S
            identity = torch.eye(q, dtype=dtype, device=device)
            max_fourier_coordinate_error = max(
                max_fourier_coordinate_error,
                float(torch.max(torch.abs(D @ S - identity))),
            )
            max_ideal_coordinate_error = max(
                max_ideal_coordinate_error,
                float(torch.max(torch.abs(ideal @ S - torch.diag(mult)))),
            )
            diag = torch.diagonal(effective); diag_err = torch.abs(diag - mult)
            rel_diag = diag_err / torch.clamp(torch.abs(mult), min=torch.finfo(dtype).eps)
            off = effective - torch.diag(diag)
            feature_modes = features[splits["train"]] @ D.T
            variances = torch.var(feature_modes, dim=0, unbiased=False)
            identifiable = variances > e1.identifiable_variance_floor
            identifiable_nonconstant = identifiable.clone()
            identifiable_nonconstant[0] = False
            identifiable_count = int(identifiable_nonconstant.sum())
            nonconstant_count = q - 1
            identifiable_fraction = identifiable_count / nonconstant_count
            if identifiable_count:
                identifiable_errors = rel_diag[identifiable_nonconstant]
                mean_identifiable_diag_error = float(identifiable_errors.mean())
                max_identifiable_diag_error = float(identifiable_errors.max())
                off_denominator = torch.linalg.vector_norm(mult[identifiable_nonconstant])
                identifiable_offdiag = (
                    float(torch.linalg.matrix_norm(off[:, identifiable_nonconstant]) / off_denominator)
                    if float(off_denominator) > 0.0 else None
                )
            else:
                mean_identifiable_diag_error = None
                max_identifiable_diag_error = None
                identifiable_offdiag = None
            for i in range(q):
                k = 0 if i == 0 else (i + 1) // 2
                comp = "constant" if i == 0 else ("cos" if i % 2 else "sin")
                mode_rows.append({"case_name": case.name, "regime": regime, "q": q, "coefficient_index": i, "wavenumber": k, "component": comp, "training_variance": float(variances[i]), "identifiable": bool(identifiable[i]), "theoretical_multiplier": float(mult[i]), "learned_effective_diagonal": float(diag[i]), "absolute_error": float(diag_err[i]), "relative_error": float(rel_diag[i])})
            test_pred = model(features[splits["test"]]); test_truth = truth[splits["test"]]
            pred_ref = real_fourier_synthesis(test_pred, config.spatial.reference_nx, domain_length=L)
            truth_ref = y_ref[splits["test"]]
            field_rel = _relative(pred_ref, truth_ref) * math.sqrt(L / config.spatial.reference_nx) / math.sqrt(L / config.spatial.reference_nx)
            data_pred = real_fourier_synthesis(test_pred, config.spatial.target_data_nx, domain_length=L)
            data_rel = _relative(data_pred, y_tar[splits["test"]])
            floor_ref = real_fourier_synthesis(test_truth, config.spatial.reference_nx, domain_length=L)
            floor_rel = _relative(floor_ref, truth_ref)
            coeff_rel = _relative(test_pred, test_truth)
            sv = torch.linalg.svdvals(model.W); svi = torch.linalg.svdvals(ideal)
            cov_eigs = torch.linalg.eigvalsh((features[splits["train"]] - features[splits["train"]].mean(0)).T @ (features[splits["train"]] - features[splits["train"]].mean(0)) / ntr)
            threshold = torch.finfo(dtype).eps * max(1.0, float(cov_eigs.max())) * max(features[splits["train"]].shape)
            pos = cov_eigs[cov_eigs > threshold]; rank=int(pos.numel()); full=rank==features.shape[1]
            cond_inf = selected_zeta == 0 and not full
            cond = None if cond_inf else float((cov_eigs.max()+selected_zeta)/(torch.clamp(cov_eigs.min(),min=0)+selected_zeta))
            pseudo = float(pos.max()/pos.min()) if pos.numel() else None
            readout_rows.append({
                "case_name": case.name, "regime": regime, "delta_nuT": delta, "q": q,
                "selected_zeta": selected_zeta, "max_theoretical_multiplier": float(mult.max()),
                "off_diagonal_frobenius_norm": float(torch.linalg.matrix_norm(off)),
                "diagonal_absolute_error_all": float(diag_err.mean()),
                "diagonal_relative_error_all": float(rel_diag.mean()),
                "identifiable_nonconstant_count": identifiable_count,
                "nonconstant_count": nonconstant_count,
                "identifiable_nonconstant_fraction": identifiable_fraction,
                "mean_identifiable_diagonal_relative_error": mean_identifiable_diag_error,
                "max_identifiable_diagonal_relative_error": max_identifiable_diag_error,
                "identifiable_off_diagonal_relative_norm": identifiable_offdiag,
                "learned_frobenius_norm": float(torch.linalg.matrix_norm(model.W)),
                "learned_operator_norm": float(sv[0]),
                "ideal_frobenius_norm": float(torch.linalg.matrix_norm(ideal)),
                "ideal_operator_norm": float(svi[0]),
                "difference_frobenius_norm": float(torch.linalg.matrix_norm(model.W - ideal)),
                "difference_operator_norm": float(torch.linalg.svdvals(model.W - ideal)[0]),
                "effective_response_matrix": json.dumps(effective.detach().cpu().tolist()),
                "theoretical_multiplier_vector": json.dumps(mult.detach().cpu().tolist()),
                "learned_effective_diagonal": json.dumps(diag.detach().cpu().tolist()),
                "feature_covariance_eigenvalues": json.dumps(cov_eigs.detach().cpu().tolist()),
                "numerical_rank": rank, "feature_dimension": features.shape[1],
                "regularized_condition_number": cond,
                "regularized_condition_number_is_infinite": cond_inf,
                "nonzero_spectrum_pseudo_condition_number": pseudo,
                "effective_dimension": float(torch.sum(cov_eigs / (cov_eigs + selected_zeta))) if selected_zeta > 0 else float(rank),
            })
            field_stats, floor_stats = _stats(field_rel), _stats(floor_rel)
            floor_ratio = field_stats["mean"] / max(floor_stats["mean"], torch.finfo(dtype).eps)
            selected_rows.append({
                "case_name": case.name, "regime": regime, "delta_nuT": delta, "q": q,
                "selected_zeta": selected_zeta,
                "clean_test_coefficient_mse": float(torch.mean((test_pred-test_truth)**2)),
                "coefficient_relative_l2_mean": _stats(coeff_rel)["mean"],
                "field_error_to_representation_floor_ratio": floor_ratio,
                **{f"full_reference_field_relative_l2_{k}": v for k, v in field_stats.items()},
                **{f"target_data_relative_l2_{k}": v for k, v in _stats(data_rel).items()},
                **{f"output_representation_floor_{k}": v for k, v in floor_stats.items()},
            })
            test_features = features[splits["test"]]
            gen = torch.Generator(device=device).manual_seed(e1.noise_seed + 1000003 * list(e1.surrogate_cases).index(case) + q)
            for repeat in range(e1.noise_repeats):
                gaussian = torch.randn(test_features.shape, generator=gen, dtype=dtype, device=device)
                for level in e1.noise_levels:
                    sigma = level * torch.linalg.vector_norm(test_features, dim=-1, keepdim=True) / math.sqrt(config.spatial.observation_dim)
                    noisy_pred = model(test_features + sigma * gaussian)
                    perturb = torch.linalg.vector_norm(noisy_pred - test_pred, dim=-1)
                    theoretical = math.sqrt(float(torch.mean(sigma.squeeze(-1) ** 2)) * float(torch.linalg.matrix_norm(model.W) ** 2))
                    noisy_field = real_fourier_synthesis(noisy_pred, config.spatial.reference_nx, domain_length=L)
                    row = {"case_name": case.name, "regime": regime, "q": q, "noise_level": level, "repeat": repeat, "output_perturbation_rms": float(torch.sqrt(torch.mean(perturb**2))), "theoretical_output_perturbation_rms": theoretical, "field_relative_l2_mean": _stats(_relative(noisy_field, truth_ref))["mean"]}
                    noise_rows.append(row)
                    if level == 0: noise_zero_diff = max(noise_zero_diff, float(torch.max(torch.abs(noisy_pred-test_pred))))
    noise_summary = []
    for key in sorted({(r["case_name"], r["regime"], r["q"], r["noise_level"]) for r in noise_rows}):
        rows = [r for r in noise_rows if (r["case_name"],r["regime"],r["q"],r["noise_level"]) == key]
        noise_summary.append({"case_name":key[0],"regime":key[1],"q":key[2],"noise_level":key[3],"repeats":len(rows),"output_perturbation_rms_mean":sum(r["output_perturbation_rms"] for r in rows)/len(rows),"theoretical_output_perturbation_rms":sum(r["theoretical_output_perturbation_rms"] for r in rows)/len(rows),"field_relative_l2_mean":sum(r["field_relative_l2_mean"] for r in rows)/len(rows),"theory_definition":"sqrt(mean_i sigma_i^2 * ||W||_F^2)"})
    tol = e1.algebraic_tolerances.float32_atol if dtype == torch.float32 else e1.algebraic_tolerances.float64_atol
    finite_path = finite_input_path_check(
        config.spatial.reference_nx,
        config.spatial.target_data_nx,
        config.spatial.surrogate_internal_nx,
        L,
        dtype=dtype,
        device=device,
    )
    split_ids = {name: master.sample_ids.index_select(0, indices.cpu()).tolist() for name, indices in splits.items()}
    data_manifest = {
        "schema_version": E1_SCHEMA_VERSION,
        "finite_input_path": "n_ref -> spectral low-pass n_tar -> trigonometric interpolation n_sur",
        "finite_input_path_runtime_check": finite_path,
        "reference_to_target_max_coefficient_error": max_coeff_diff,
        "sample_ids": master.sample_ids.tolist(),
        "train_ids": split_ids["train"],
        "validation_ids": split_ids["val"],
        "test_ids": split_ids["test"],
        "split_hash": stable_hash_json(split_ids),
        "n_ref": config.spatial.reference_nx,
        "n_tar": config.spatial.target_data_nx,
        "n_sur": config.spatial.surrogate_internal_nx,
        "J": config.spatial.observation_dim,
        "q_values": list(e1.output_dims),
        "fourier_order": "constant,cos1,sin1,...",
        "observation_scaling": "sqrt(L/J)",
        "actual_solver": "spectral_exact",
        "feature_preprocessing": "l2_scaling_only; train-mean centering inside affine ridge; no z-score",
        "ridge_selection_metric": e1.selection_metric,
        "ridge_tie_break": e1.ridge_tie_break,
        "zero_ridge_solver": e1.zero_ridge_solver,
        "ridge_svd_rcond": e1.ridge_svd_rcond,
        "ridge_svd_default_cutoff_rule": "eps(dtype) * max(N,J) * sigma_max",
    }
    return {
        "ridge_selection": ridge_rows,
        "selected_results": selected_rows,
        "readout_diagnostics": readout_rows,
        "mode_comparison": mode_rows,
        "noise_results": noise_rows,
        "noise_summary": noise_summary,
        "models": models,
        "data_manifest": data_manifest,
        "checks": {
            "target_coefficients_agree_with_reference": max_coeff_diff <= tol,
            "noise_zero_matches_clean": noise_zero_diff <= tol,
            "finite_input_path_verified": finite_path["status"] == "pass",
            "heat_solver_algebraic_error": heat_algebraic_error(config, device),
            "real_fourier_coordinate_error": max_fourier_coordinate_error,
            "ideal_readout_coordinate_error": max_ideal_coordinate_error,
        },
    }
