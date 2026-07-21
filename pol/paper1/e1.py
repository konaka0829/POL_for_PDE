from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .config import Paper1Config
from .e0 import E0_SCHEMA_VERSION, MASTER_SCHEMA_VERSION, load_master_initial_conditions
from .grids import spectral_resample_periodic
from .heat import heat_multiplier_vector, heat_regime, solve_heat_exact
from .readouts import fit_centered_affine_ridge, l2_analysis_matrix, l2_synthesis_matrix
from .target_representation import real_fourier_analysis, real_fourier_synthesis

E1_SCHEMA_VERSION = "paper1-e1-v1"
E0_REQUIRED = ("e0_summary.json", "resampling_checks.json", "input_interface_checks.json", "model1_identity.json", "master_initial_conditions.pt", "master_manifest.json")


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    return {"relative_path": str(path.relative_to(root)) if root else str(path), "byte_size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def validate_e0_prerequisite(e0_dir: str | Path, config: Paper1Config) -> tuple[Paper1Config, Any, dict[str, Any]]:
    root = Path(e0_dir)
    missing = [name for name in E0_REQUIRED if not (root / name).is_file()]
    if missing:
        raise ValueError("E0 prerequisite missing artifact: " + missing[0])
    artifacts = [file_record(root / name) for name in E0_REQUIRED]
    summary = json.loads((root / "e0_summary.json").read_text())
    if summary.get("schema_version") != E0_SCHEMA_VERSION or summary.get("status") != "pass":
        raise ValueError("E0 prerequisite is not a passing known-schema E0 run")
    required = summary.get("required_checks")
    if not isinstance(required, dict) or not required or any(v != "pass" and (not isinstance(v, dict) or v.get("status") != "pass") for v in required.values()):
        raise ValueError("E0 prerequisite required checks are missing or not all pass")
    for name in ("resampling_checks.json", "input_interface_checks.json", "model1_identity.json"):
        doc = json.loads((root / name).read_text())
        if doc.get("status") != "pass":
            raise ValueError(f"E0 prerequisite {name} is not pass")
    selected = summary.get("selected_reference", {})
    reference_nx = selected.get("reference_nx")
    if not isinstance(reference_nx, int) or reference_nx < config.spatial.target_data_nx:
        raise ValueError("E0 selected reference_nx is missing or below target_data_nx")
    manifest = json.loads((root / "master_manifest.json").read_text())
    if manifest.get("schema_version") != MASTER_SCHEMA_VERSION:
        raise ValueError("unknown E0 master archive schema")
    effective = replace(config, spatial=replace(config.spatial, reference_nx=reference_nx))
    master = load_master_initial_conditions(root / "master_initial_conditions.pt", effective)
    prerequisite = {
        "schema_version": E1_SCHEMA_VERSION, "status": "pass", "e0_dir": str(root),
        "user_requested_reference_nx": config.spatial.reference_nx, "effective_reference_nx": reference_nx,
        "artifacts": artifacts, "master_tensor_hash": manifest["tensor_hash"],
    }
    return effective, master, prerequisite


def _stats(v: torch.Tensor) -> dict[str, float]:
    v = v.detach().cpu()
    return {"mean": float(v.mean()), "median": float(v.median()), "max": float(v.max())}


def _relative(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - truth, dim=-1) / torch.clamp(torch.linalg.vector_norm(truth, dim=-1), min=torch.finfo(truth.dtype).eps)


def run_e1(config: Paper1Config, master: Any) -> dict[str, Any]:
    assert config.e1 is not None
    e1, L, dtype = config.e1, config.domain.length, config.data.torch_dtype()
    device = torch.device("cuda" if config.data.device == "auto" and torch.cuda.is_available() else config.data.device)
    u_ref = spectral_resample_periodic(master.values_master.to(device=device, dtype=dtype), config.spatial.reference_nx, domain_length=L)
    u_tar = spectral_resample_periodic(u_ref, config.spatial.target_data_nx, domain_length=L)
    # This boundary receives only finite n_tar values; discarded reference modes cannot leak downstream.
    u_sur0 = spectral_resample_periodic(u_tar, config.spatial.surrogate_internal_nx, domain_length=L)
    y_ref = solve_heat_exact(u_ref, nu=config.target.nu, T=config.target.T, domain_length=L)
    y_tar = spectral_resample_periodic(y_ref, config.spatial.target_data_nx, domain_length=L)
    ntr, nv = config.data.n_train, config.data.n_val
    splits = {"train": slice(0, ntr), "val": slice(ntr, ntr + nv), "test": slice(ntr + nv, config.data.total_samples)}
    ridge_rows: list[dict[str, Any]] = []; selected_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []; mode_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []; models: dict[str, Any] = {}
    max_coeff_diff = 0.0; noise_zero_diff = 0.0
    for case in e1.surrogate_cases:
        regime, delta = heat_regime(target_nu=config.target.nu, target_T=config.target.T, surrogate_nu=case.nu, surrogate_T=case.T)
        r_sur = solve_heat_exact(u_sur0, nu=case.nu, T=case.T, domain_length=L)
        observed = spectral_resample_periodic(r_sur, config.spatial.observation_dim, domain_length=L)
        features = observed * math.sqrt(L / config.spatial.observation_dim)
        for q in e1.output_dims:
            truth = real_fourier_analysis(y_tar, q, domain_length=L)
            ref_coeff = real_fourier_analysis(y_ref, q, domain_length=L)
            max_coeff_diff = max(max_coeff_diff, float(torch.max(torch.abs(truth - ref_coeff))))
            candidates = []
            for zeta in e1.ridge_zetas:
                model = fit_centered_affine_ridge(features[splits["train"]], truth[splits["train"]], zeta)
                mse = {}
                for split, sl in splits.items():
                    mse[split] = float(torch.mean((model(features[sl]) - truth[sl]) ** 2))
                ridge_rows.append({"case_name": case.name, "regime": regime, "q": q, "zeta": zeta, "train_coefficient_mse": mse["train"], "validation_coefficient_mse": mse["val"], "test_coefficient_mse_diagnostic": mse["test"]})
                candidates.append((mse["val"], float(zeta), model))
            best_value = min(x[0] for x in candidates)
            eligible = [x for x in candidates if x[0] <= best_value + e1.ridge_tie_tolerance]
            _, selected_zeta, model = min(eligible, key=lambda x: x[1])
            models[f"{case.name}/q{q}"] = {"W": model.W.detach().cpu(), "b": model.b.detach().cpu(), "zeta": selected_zeta, "regime": regime, "delta": delta}
            D = l2_analysis_matrix(q, config.spatial.observation_dim, domain_length=L, dtype=dtype, device=device)
            S = l2_synthesis_matrix(q, config.spatial.observation_dim, domain_length=L, dtype=dtype, device=device)
            mult = heat_multiplier_vector(q, target_nu=config.target.nu, target_T=config.target.T, surrogate_nu=case.nu, surrogate_T=case.T, domain_length=L, dtype=dtype, device=device)
            ideal = torch.diag(mult) @ D
            effective = model.W @ S
            diag = torch.diagonal(effective); diag_err = torch.abs(diag - mult)
            rel_diag = diag_err / torch.clamp(torch.abs(mult), min=torch.finfo(dtype).eps)
            off = effective - torch.diag(diag)
            feature_modes = features[splits["train"]] @ D.T
            variances = torch.var(feature_modes, dim=0, unbiased=False)
            identifiable = variances > e1.identifiable_variance_floor
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
            pos = cov_eigs[cov_eigs > torch.finfo(dtype).eps * max(1.0, float(cov_eigs.max()))]
            readout_rows.append({"case_name": case.name, "regime": regime, "delta_nuT": delta, "q": q, "selected_zeta": selected_zeta, "max_theoretical_multiplier": float(mult.max()), "off_diagonal_frobenius_norm": float(torch.linalg.matrix_norm(off)), "diagonal_absolute_error_all": float(diag_err.mean()), "diagonal_relative_error_all": float(rel_diag.mean()), "diagonal_absolute_error_identifiable": float(diag_err[identifiable].mean()) if bool(identifiable.any()) else 0.0, "diagonal_relative_error_identifiable": float(rel_diag[identifiable].mean()) if bool(identifiable.any()) else 0.0, "learned_frobenius_norm": float(torch.linalg.matrix_norm(model.W)), "learned_operator_norm": float(sv[0]), "ideal_frobenius_norm": float(torch.linalg.matrix_norm(ideal)), "ideal_operator_norm": float(svi[0]), "difference_frobenius_norm": float(torch.linalg.matrix_norm(model.W - ideal)), "difference_operator_norm": float(torch.linalg.svdvals(model.W - ideal)[0]), "effective_response_matrix": json.dumps(effective.detach().cpu().tolist()), "theoretical_multiplier_vector": json.dumps(mult.detach().cpu().tolist()), "learned_effective_diagonal": json.dumps(diag.detach().cpu().tolist()), "feature_covariance_eigenvalues": json.dumps(cov_eigs.detach().cpu().tolist()), "numerical_rank": int(pos.numel()), "condition_number": float(pos.max() / pos.min()) if pos.numel() else 0.0, "effective_dimension": float(torch.sum(cov_eigs / (cov_eigs + selected_zeta))) if selected_zeta > 0 else float(pos.numel())})
            selected_rows.append({"case_name": case.name, "regime": regime, "delta_nuT": delta, "q": q, "selected_zeta": selected_zeta, "clean_test_coefficient_mse": float(torch.mean((test_pred-test_truth)**2)), "coefficient_relative_l2_mean": _stats(coeff_rel)["mean"], **{f"full_reference_field_relative_l2_{k}": v for k,v in _stats(field_rel).items()}, **{f"target_data_relative_l2_{k}": v for k,v in _stats(data_rel).items()}, **{f"output_representation_floor_{k}": v for k,v in _stats(floor_rel).items()}})
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
    return {"ridge_selection":ridge_rows,"selected_results":selected_rows,"readout_diagnostics":readout_rows,"mode_comparison":mode_rows,"noise_results":noise_rows,"noise_summary":noise_summary,"models":models,"data_manifest":{"finite_input_path":"n_ref -> spectral low-pass n_tar -> trigonometric interpolation n_sur","reference_to_target_max_coefficient_error":max_coeff_diff,"sample_ids":master.sample_ids.tolist()},"checks":{"target_coefficients_agree_with_reference":max_coeff_diff <= tol,"noise_zero_matches_clean":noise_zero_diff <= tol}}
