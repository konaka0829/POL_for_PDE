from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .config import Paper1Config
from .datasets import _tensor_hash
from .grids import periodic_grid, spectral_resample_periodic
from .initial_conditions import MasterInitialConditions, build_master_grf_initial_conditions
from .interfaces import build_surrogate_initial_state, derive_finite_resolution_data
from .metrics import aggregate_errors, compare_fields_on_common_grid, samplewise_l2_errors
from .model1 import decode_equispaced_point_observation_to_real_fourier
from .observations import observe_equispaced_periodic
from .solvers import solve_burgers_final_state
from .target_representation import real_fourier_analysis, real_fourier_synthesis

E0_SCHEMA_VERSION = "paper1-e0-v1"
MASTER_SCHEMA_VERSION = "paper1-master-initial-conditions-v1"


def _close(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> tuple[bool, float]:
    error = float(torch.max(torch.abs(a - b)).detach().cpu())
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol)), error


def run_algebraic_checks(config: Paper1Config) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run E0-a resampling and real-Fourier projector checks."""
    assert config.e0 is not None
    tol = config.e0.algebraic_tolerances
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
        atol = tol.float32_atol if dtype == torch.float32 else tol.float64_atol
        rtol = tol.float32_rtol if dtype == torch.float32 else tol.float64_rtol
        passed, error = _close(actual, expected, atol=atol, rtol=rtol)
        checks[name] = {"status": "pass" if passed else "fail", "max_abs_error": error, "atol": atol, "rtol": rtol}

    L = config.domain.length
    for dtype in (torch.float32, torch.float64):
        suffix = str(dtype).removeprefix("torch.")
        for n_in, n_out in ((15, 16), (16, 15), (16, 24), (15, 25)):
            x = periodic_grid(n_in, L, dtype=dtype)
            for kind, values, expected_fn in (
                ("constant", torch.full_like(x, 1.25), lambda y: torch.full_like(y, 1.25)),
                ("cosine", torch.cos(2 * torch.pi * 3 * x / L), lambda y: torch.cos(2 * torch.pi * 3 * y / L)),
                ("sine", torch.sin(2 * torch.pi * 2 * x / L), lambda y: torch.sin(2 * torch.pi * 2 * y / L)),
            ):
                y = periodic_grid(n_out, L, dtype=dtype)
                record(f"{kind}_{n_in}_to_{n_out}_{suffix}", spectral_resample_periodic(values, n_out, domain_length=L), expected_fn(y), dtype)
        original = torch.randn(2, 16, generator=torch.Generator().manual_seed(17), dtype=dtype)
        cloned = spectral_resample_periodic(original, 16, domain_length=L)
        record(f"identity_batch_{suffix}", cloned, original, dtype)
        checks[f"identity_non_alias_{suffix}"] = {"status": "pass" if cloned.data_ptr() != original.data_ptr() else "fail"}
        x16 = periodic_grid(16, L, dtype=dtype)
        nyq = torch.cos(2 * torch.pi * 8 * x16 / L)
        y32 = periodic_grid(32, L, dtype=dtype)
        record(f"nyquist_split_{suffix}", spectral_resample_periodic(nyq, 32, domain_length=L), torch.cos(2 * torch.pi * 8 * y32 / L), dtype)
        x32 = periodic_grid(32, L, dtype=dtype)
        ordinary = torch.cos(2 * torch.pi * 8 * x32 / L)
        record(f"nyquist_merge_{suffix}", spectral_resample_periodic(ordinary, 16, domain_length=L), nyq, dtype)
        low = 0.4 + 0.7 * torch.cos(2 * torch.pi * 3 * x16 / L) - 0.2 * torch.sin(2 * torch.pi * 2 * x16 / L)
        up = spectral_resample_periodic(low, 32, domain_length=L)
        coeff = torch.fft.fft(up, norm="forward")
        high_energy = torch.max(torch.abs(coeff[9:24]))
        checks[f"upsampling_zero_padding_{suffix}"] = {"status": "pass" if float(high_energy) <= (tol.float32_atol if dtype == torch.float32 else tol.float64_atol) else "fail", "max_high_coefficient": float(high_energy)}
    x64 = periodic_grid(64, L, dtype=torch.float64)
    high = torch.cos(2 * torch.pi * 17 * x64 / L)
    naive = high[::4]
    x16 = periodic_grid(16, L, dtype=torch.float64)
    alias = torch.cos(2 * torch.pi * x16 / L)
    filtered = spectral_resample_periodic(high, 16, domain_length=L)
    naive_ok = torch.allclose(naive, alias, atol=tol.float64_atol, rtol=tol.float64_rtol)
    filtered_ok = float(torch.max(torch.abs(filtered))) <= tol.float64_atol
    checks["alias_prevention_k17_64_to_16"] = {"status": "pass" if naive_ok and filtered_ok else "fail", "naive_aliases_to_k1": bool(naive_ok), "spectral_max_abs": float(torch.max(torch.abs(filtered)))}
    resampling = {"schema_version": E0_SCHEMA_VERSION, "checks": checks, "status": "pass" if all(v["status"] == "pass" for v in checks.values()) else "fail"}

    q, nx = 9, 32
    gen = torch.Generator().manual_seed(23)
    c = torch.randn(4, q, generator=gen, dtype=torch.float64)
    synthesized = real_fourier_synthesis(c, nx, domain_length=L)
    pq = real_fourier_analysis(synthesized, q, domain_length=L)
    projected = real_fourier_synthesis(real_fourier_analysis(torch.randn(4, nx, generator=gen, dtype=torch.float64), q, domain_length=L), nx, domain_length=L)
    projected_twice = real_fourier_synthesis(real_fourier_analysis(projected, q, domain_length=L), nx, domain_length=L)
    energy_field = (L / nx) * torch.sum(synthesized**2, dim=-1)
    energy_coeff = torch.sum(c**2, dim=-1)
    projector_checks = {}
    for name, a, b in (("Pq_Pqstar_identity", pq, c), ("projector_idempotence", projected_twice, projected), ("parseval_isometry", energy_field, energy_coeff)):
        passed, error = _close(a, b, atol=tol.float64_atol, rtol=tol.float64_rtol)
        projector_checks[name] = {"status": "pass" if passed else "fail", "max_abs_error": error}
    projector = {"schema_version": E0_SCHEMA_VERSION, "coefficient_order": "constant,cos1,sin1,cos2,sin2,...", "checks": projector_checks, "status": "pass" if all(v["status"] == "pass" for v in projector_checks.values()) else "fail"}
    return resampling, projector


def save_master_initial_conditions(master: MasterInitialConditions, path: Path, manifest_path: Path, config: Paper1Config) -> dict[str, Any]:
    tensor_hash = _tensor_hash(master.values_master)
    payload = {"schema_version": MASTER_SCHEMA_VERSION, "sample_ids": master.sample_ids.detach().cpu(), "values": master.values_master.detach().cpu(), "metadata": {"domain_length": master.domain_length, "seed": master.seed, "maximum_nx": master.master_nx, "dtype": str(master.values_master.dtype).removeprefix("torch."), "grf_gamma": config.data.grf_gamma, "grf_tau": config.data.grf_tau, "grf_sigma": config.data.grf_sigma, "grf_mean": config.data.grf_mean, "tensor_hash": tensor_hash}}
    torch.save(payload, path)
    manifest = {**payload["metadata"], "schema_version": MASTER_SCHEMA_VERSION, "sample_ids": master.sample_ids.detach().cpu().tolist(), "sample_count": int(master.sample_ids.numel())}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_master_initial_conditions(path: str | Path, config: Paper1Config) -> MasterInitialConditions:
    """Load and validate an E0 master archive for production reuse."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != MASTER_SCHEMA_VERSION:
        raise ValueError("unsupported master initial-condition schema_version")
    values, ids, meta = payload["values"], payload["sample_ids"], payload["metadata"]
    checks = {
        "sample count": ids.numel() == config.data.total_samples,
        "sample IDs": torch.equal(ids, torch.arange(config.data.total_samples)),
        "domain length": meta.get("domain_length") == config.domain.length,
        "seed": meta.get("seed") == config.data.seed,
        "dtype": meta.get("dtype") == config.data.dtype,
        "maximum nx": int(meta.get("maximum_nx", 0)) >= config.spatial.reference_nx,
        "tensor hash": meta.get("tensor_hash") == _tensor_hash(values),
        "GRF parameters": all(meta.get(k) == getattr(config.data, k) for k in ("grf_gamma", "grf_tau", "grf_sigma", "grf_mean")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("master initial-condition archive mismatch: " + ", ".join(failed))
    return MasterInitialConditions(ids, values, torch.fft.rfft(values, dim=-1, norm="forward"), values.shape[-1], float(meta["domain_length"]), int(meta["seed"]))


def _low_mode_error(candidate: torch.Tensor, reference: torch.Tensor, q: int, L: float) -> dict[str, float]:
    c = real_fourier_analysis(candidate, q, domain_length=L)
    r = real_fourier_analysis(reference, q, domain_length=L)
    numerator = torch.linalg.vector_norm(c - r, dim=-1)
    denominator = torch.linalg.vector_norm(r, dim=-1)
    rel = numerator / torch.clamp(denominator, min=1e-14)
    return aggregate_errors(rel)


def run_reference_convergence(config: Paper1Config, master: MasterInitialConditions) -> dict[str, Any]:
    """Run E0-b spatial/temporal convergence from one shared master archive."""
    assert config.e0 is not None
    e0, L = config.e0, config.domain.length
    ids = torch.tensor(e0.calibration_sample_ids, dtype=torch.long, device=master.values_master.device)
    finest_time = e0.time_candidates[-1]
    spatial_results: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for nx in e0.reference_nx_candidates:
        u0 = spectral_resample_periodic(master.values_master.index_select(0, ids), nx, domain_length=L)
        solved = solve_burgers_final_state(u0, nu=config.target.nu, T=config.target.T, dt=finest_time.dt, fine_dt=finest_time.fine_dt, solver=config.target.solver, dealias=config.target.dealias, domain_length=L)
        spatial_results[nx] = (solved.values, solved.metadata.to_dict())
    finest_nx = e0.reference_nx_candidates[-1]
    spatial_ref = spatial_results[finest_nx][0]
    for nx in e0.reference_nx_candidates:
        values, meta = spatial_results[nx]
        comparison = compare_fields_on_common_grid(values, spatial_ref, common_nx=finest_nx, domain_length=L)
        up = spectral_resample_periodic(values, finest_nx, domain_length=L)
        low = _low_mode_error(up, spatial_ref, e0.q_reference_check, L)
        rows.append({"kind": "spatial", "candidate_nx": nx, **meta, "absolute_l2": comparison["absolute_aggregate"], "relative_l2": comparison["relative_aggregate"], "low_mode_relative_l2": low, "master_hash": _tensor_hash(master.values_master), "sample_ids": list(e0.calibration_sample_ids)})
    temporal_results = []
    u0_finest = spectral_resample_periodic(master.values_master.index_select(0, ids), finest_nx, domain_length=L)
    for candidate in e0.time_candidates:
        solved = solve_burgers_final_state(u0_finest, nu=config.target.nu, T=config.target.T, dt=candidate.dt, fine_dt=candidate.fine_dt, solver=config.target.solver, dealias=config.target.dealias, domain_length=L)
        temporal_results.append(solved)
    temporal_ref = temporal_results[-1].values
    for candidate, solved in zip(e0.time_candidates, temporal_results):
        errors = samplewise_l2_errors(solved.values, temporal_ref, domain_length=L)
        rows.append({"kind": "temporal", "candidate_nx": finest_nx, **solved.metadata.to_dict(), "absolute_l2": aggregate_errors(errors["absolute"]), "relative_l2": aggregate_errors(errors["relative"]), "low_mode_relative_l2": _low_mode_error(solved.values, temporal_ref, e0.q_reference_check, L), "master_hash": _tensor_hash(master.values_master), "sample_ids": list(e0.calibration_sample_ids)})
    tol = e0.reference_tolerances
    def passes(row: dict[str, Any]) -> bool:
        return row["relative_l2"]["mean"] <= tol.mean_relative_l2 and row["relative_l2"]["max"] <= tol.max_relative_l2 and row["low_mode_relative_l2"]["mean"] <= tol.low_mode_relative_l2
    spatial_rows = [r for r in rows if r["kind"] == "spatial"]
    temporal_rows = [r for r in rows if r["kind"] == "temporal"]
    spatial_pair_pass = passes(spatial_rows[-2])
    temporal_pair_pass = passes(temporal_rows[-2])
    selected_spatial = next((r for r in spatial_rows if passes(r)), None) if spatial_pair_pass else None
    selected_temporal = next((r for r in temporal_rows if passes(r)), None) if temporal_pair_pass else None
    return {"schema_version": E0_SCHEMA_VERSION, "selection_policy": e0.selection_policy, "tolerances": asdict(tol), "rows": rows, "spatial_status": "pass" if selected_spatial is not None else "fail", "temporal_status": "pass" if selected_temporal is not None else "fail", "selected_spatial": selected_spatial, "selected_temporal": selected_temporal}


def run_interface_checks(config: Paper1Config, master: MasterInitialConditions, reference_state: torch.Tensor) -> dict[str, Any]:
    """Run E0-c finite-data and no-high-frequency-leak checks."""
    assert config.e0 is not None
    identity, L = config.e0.model1_identity, config.domain.length
    uref = master.values_master[:2]
    yref = reference_state[:2]
    finite = derive_finite_resolution_data(uref, yref, target_data_nx=identity.target_data_nx, target_output_dim=identity.target_output_dim, domain_length=L, sample_ids=master.sample_ids[:2])
    surrogate = build_surrogate_initial_state(finite.u0_data, surrogate_internal_nx=identity.surrogate_internal_nx, domain_length=L)
    nref, ntar = uref.shape[-1], identity.target_data_nx
    x = periodic_grid(nref, L, dtype=uref.dtype, device=uref.device)
    low = 0.4 + torch.cos(2 * torch.pi * 2 * x / L)
    high_k = ntar // 2 + 1
    pair = torch.stack([low, low + 0.3 * torch.cos(2 * torch.pi * high_k * x / L)])
    dummy = torch.zeros_like(pair)
    pair_data = derive_finite_resolution_data(pair, dummy, target_data_nx=ntar, target_output_dim=None, domain_length=L)
    pair_sur = build_surrogate_initial_state(pair_data.u0_data, surrogate_internal_nx=identity.surrogate_internal_nx, domain_length=L)
    atol = config.e0.algebraic_tolerances.float64_atol if uref.dtype == torch.float64 else config.e0.algebraic_tolerances.float32_atol
    leak_pass = torch.allclose(pair_data.u0_data[0], pair_data.u0_data[1], atol=atol, rtol=0) and torch.allclose(pair_sur[0], pair_sur[1], atol=atol, rtol=0)
    ref_coeff = real_fourier_analysis(yref, identity.target_output_dim, domain_length=L)
    coeff_pass = torch.allclose(finite.target_coefficients, ref_coeff, atol=atol, rtol=10 * atol)
    return {"schema_version": E0_SCHEMA_VERSION, "status": "pass" if leak_pass and coeff_pass else "fail", "finite_data_interface": {"status": "pass", "u0_shape": list(finite.u0_data.shape), "surrogate_shape": list(surrogate.shape)}, "no_high_frequency_leak": {"status": "pass" if leak_pass else "fail", "high_mode": high_k, "max_finite_difference": float(torch.max(torch.abs(pair_data.u0_data[0] - pair_data.u0_data[1]))), "max_surrogate_difference": float(torch.max(torch.abs(pair_sur[0] - pair_sur[1])))}, "target_coefficient_consistency": {"status": "pass" if coeff_pass else "fail", "max_abs_error": float(torch.max(torch.abs(finite.target_coefficients - ref_coeff)))}}


def run_model1_checks(config: Paper1Config, master: MasterInitialConditions) -> dict[str, Any]:
    """Run E0-d matched full-observation and reduced-J checks."""
    assert config.e0 is not None
    spec, L = config.e0.model1_identity, config.domain.length
    uref = master.values_master.index_select(0, torch.tensor(config.e0.calibration_sample_ids, device=master.values_master.device))
    finite_u = spectral_resample_periodic(uref, spec.target_data_nx, domain_length=L)
    z0 = build_surrogate_initial_state(finite_u, surrogate_internal_nx=spec.surrogate_internal_nx, domain_length=L)
    kwargs = dict(nu=config.target.nu, T=config.target.T, dt=config.target.dt, fine_dt=config.target.fine_dt, solver=config.target.solver, dealias=config.target.dealias, domain_length=L)
    target = solve_burgers_final_state(z0, **kwargs)
    surrogate = solve_burgers_final_state(z0, **kwargs)
    direct = real_fourier_analysis(target.values, spec.target_output_dim, domain_length=L)
    features = observe_equispaced_periodic(surrogate.values, spec.observation_dim, domain_length=L, l2_scale=True)
    decoded = decode_equispaced_point_observation_to_real_fourier(features, spec.target_output_dim, domain_length=L)
    target_proj = real_fourier_synthesis(direct, spec.surrogate_internal_nx, domain_length=L)
    decoded_proj = real_fourier_synthesis(decoded, spec.surrogate_internal_nx, domain_length=L)
    atol = config.e0.algebraic_tolerances.float64_atol if z0.dtype == torch.float64 else config.e0.algebraic_tolerances.float32_atol
    full_checks = {"terminal_array": torch.allclose(target.values, surrogate.values, atol=atol, rtol=atol), "direct_vs_decoder_coefficients": torch.allclose(direct, decoded, atol=atol, rtol=atol), "projected_fields": torch.allclose(target_proj, decoded_proj, atol=atol, rtol=atol)}
    J, q = 16, 9
    x = periodic_grid(spec.surrogate_internal_nx, L, dtype=z0.dtype, device=z0.device)
    bandlimited = 0.3 + 0.7 * torch.cos(2 * torch.pi * 3 * x / L) - 0.2 * torch.sin(2 * torch.pi * 2 * x / L)
    f = observe_equispaced_periodic(bandlimited.unsqueeze(0), J, domain_length=L, l2_scale=True)
    reduced = decode_equispaced_point_observation_to_real_fourier(f, q, domain_length=L)
    truth = real_fourier_analysis(bandlimited.unsqueeze(0), q, domain_length=L)
    reduced_pass = torch.allclose(reduced, truth, atol=atol, rtol=atol)
    high = bandlimited + 0.4 * torch.cos(2 * torch.pi * (J + 2) * x / L)
    aliased = decode_equispaced_point_observation_to_real_fourier(observe_equispaced_periodic(high.unsqueeze(0), J, domain_length=L, l2_scale=True), q, domain_length=L)
    counterexample = not torch.allclose(aliased, truth, atol=atol, rtol=atol)
    full_pass = all(full_checks.values())
    return {"schema_version": E0_SCHEMA_VERSION, "status": "pass" if full_pass and reduced_pass and counterexample else "fail", "solver_metadata": target.metadata.to_dict(), "full_observation": {"status": "pass" if full_pass else "fail", "checks": full_checks, "terminal_max_abs_error": float(torch.max(torch.abs(target.values - surrogate.values))), "coefficient_max_abs_error": float(torch.max(torch.abs(direct - decoded))), "projection_max_abs_error": float(torch.max(torch.abs(target_proj - decoded_proj)))}, "bandlimited_reduced_j": {"status": "pass" if reduced_pass else "fail", "J": J, "q": q, "max_abs_error": float(torch.max(torch.abs(reduced - truth)))}, "aliasing_counterexample": {"status": "pass" if counterexample else "fail", "expected_non_identity": True, "max_abs_difference": float(torch.max(torch.abs(aliased - truth)))}}
