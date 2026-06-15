from __future__ import print_function

import sys

if sys.version_info < (3, 10):
    sys.stderr.write(
        "model123_burgers_1d.py requires Python 3.10+.\n"
        "You are running Python %s.\n"
        "Use `python3 model123_burgers_1d.py ...` instead.\n" % sys.version.split()[0]
    )
    raise SystemExit(1)

import argparse
import csv
import json
import os
from pathlib import Path
from dataclasses import asdict
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pol.cli import add_data_mode_args, add_split_args, validate_data_mode_args
from pol.io_mat import MatReader
from pol.metadata import (
    file_sha256,
    get_command_line,
    get_git_info,
    get_runtime_info,
    normalize_dataset_metadata,
    stable_json_hash,
    to_jsonable,
    validate_dataset_metadata,
)
from pol.model123_1d import Model1Predictor1D, Model2Regressor1D, Model3Regressor1D, Model123Config
from pol.model123_1d.metrics import (
    dataset_abs_l2h_error,
    dataset_abs_l2h_rmse,
    dataset_rel_l2h_aggregate,
    dataset_rel_l2h_mean,
    per_sample_abs_l2h_error,
    per_sample_rel_l2h_error,
)
from pol.model123_1d.error_decomposition import (
    ErrorDecompositionConfig,
    compute_time_scaled_defect_for_dataset,
)
from pol.plotting import plot_1d_prediction, plot_error_histogram, save_figure_all_formats


def parse_args():
    parser = argparse.ArgumentParser(description="Model 1 / 2 / 3 runner for 1D Burgers target")
    parser.add_argument("--config", default="")
    parser.add_argument("--model", choices=("model1", "model2", "model3"), required=True)
    parser.add_argument(
        "--reservoir",
        choices=("burgers", "reaction_diffusion", "ks", "static", "heat", "advection"),
        default="burgers",
    )

    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_model123.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(parser, default_train_split=0.75, default_seed=0)
    parser.add_argument("--ntrain", type=int, default=384)
    parser.add_argument("--nval", type=int, default=0)
    parser.add_argument("--ntest", type=int, default=128)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--data-dtype", choices=("preserve", "float32", "float64"), default="float32")
    parser.add_argument("--sim-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--target-nu", type=float, default=None)
    parser.add_argument("--Ttilde", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--feature-times", type=str, default="")
    parser.add_argument("--K", type=int, default=1)

    parser.add_argument("--obs", choices=("full", "points", "fourier", "proj"), default="full")
    parser.add_argument("--J", type=int, default=128)
    parser.add_argument("--sensor-mode", choices=("equispaced", "random"), default="equispaced")
    parser.add_argument("--sensor-seed", type=int, default=0)

    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--input-shift", type=float, default=0.0)

    parser.add_argument("--ridge-zeta", type=float, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=None)
    parser.add_argument(
        "--ridge-convention",
        default="normalized_empirical_l2h_unweighted_frobenius",
        choices=("normalized_empirical_l2h_unweighted_frobenius", "legacy_unnormalized_gram"),
    )
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--standardize-features", type=int, choices=(0, 1), default=0)
    parser.add_argument("--feature-std-eps", type=float, default=1e-6)

    parser.add_argument("--elm-h", type=int, default=1024)
    parser.add_argument("--elm-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)

    parser.add_argument("--rd-nu", type=float, default=1e-3)
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    parser.add_argument("--res-burgers-nu", type=float, default=0.05)
    parser.add_argument("--res-burgers-b", type=float, default=1.0)
    parser.add_argument("--heat-nu", type=float, default=1e-2)
    parser.add_argument("--advection-c", type=float, default=1.0)
    parser.add_argument("--ks-dealias", action="store_true")
    parser.add_argument("--ks-b", type=float, default=1.0)
    parser.add_argument("--ks-eta", type=float, default=1.0)
    parser.add_argument("--ks-kappa", type=float, default=1.0)
    parser.add_argument("--burgers-scheme", choices=("semi_implicit", "split_step", "etdrk4"), default="split_step")
    parser.add_argument("--burgers-fine-dt", type=float, default=1e-4)
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-metadata-mismatch", action="store_true")
    parser.add_argument("--require-complete-metadata", action="store_true")
    parser.add_argument("--compute-time-scaled-defect", action="store_true")
    parser.add_argument(
        "--defect-target-nu",
        type=float,
        default=None,
        help="target Burgers viscosity used in the defect computation. If omitted, read `nu` from the dataset metadata when available; otherwise warn and fall back to 0.05.",
    )
    parser.add_argument("--defect-time-quadrature", choices=("trapezoid", "left"), default="trapezoid")
    parser.add_argument("--defect-beta-mode", choices=("zero", "fixed"), default="zero")
    parser.add_argument("--defect-beta-fixed", type=float, default=0.0)
    parser.add_argument("--defect-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--out-dir", type=str, default="outputs/model123_burgers_1d")
    parser.add_argument(
        "--save-model",
        nargs="?",
        const="model.pt",
        default="",
        help="Optional save path. If omitted, writes out-dir/model.pt",
    )
    args = parser.parse_args()
    _apply_config_defaults(parser, args)
    if not hasattr(args, "_config_applied"):
        args._config_applied = {}
    for name in [
        "expected_ic_type",
        "expected_solver",
        "expected_time_integrator",
        "expected_burgers_scheme",
        "expected_dealias",
        "expected_equation",
        "expected_domain_length",
    ]:
        if not hasattr(args, name):
            setattr(args, name, None)
    args.data_seed = args.seed if args.data_seed is None else args.data_seed
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.ridge_zeta is None and args.ridge_lambda is None:
        args.ridge_zeta = 1e-4
        args.ridge_lambda = 1e-4
    elif args.ridge_zeta is None:
        args.ridge_zeta = float(args.ridge_lambda)
    elif args.ridge_lambda is None:
        args.ridge_lambda = float(args.ridge_zeta)
    validate_data_mode_args(args, parser)
    if args.Ttilde <= 0.0:
        args.Ttilde = args.T
    if args.T <= 0.0 or args.Ttilde <= 0.0 or args.dt <= 0.0:
        parser.error("--T, --Ttilde, and --dt must be positive")
    if args.ridge_zeta < 0.0:
        parser.error("--ridge-zeta must be non-negative")
    if args.feature_std_eps <= 0.0:
        parser.error("--feature-std-eps must be positive")
    if args.sub <= 0 or args.batch_size <= 0 or args.ntrain <= 0 or args.nval < 0 or args.ntest <= 0:
        parser.error("--sub, --batch-size, --ntrain, and --ntest must be positive; --nval must be nonnegative")
    return args


def _set_if_default(parser, args, name, value):
    if value is None or not hasattr(args, name):
        return
    if getattr(args, name) == parser.get_default(name):
        setattr(args, name, value)
        args._config_applied[name] = value


def _apply_config_defaults(parser, args):
    if not getattr(args, "config", ""):
        return
    args._config_applied = {}
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    target = cfg.get("target", {})
    domain = cfg.get("domain", {})
    data = cfg.get("data", {})
    readout = cfg.get("readout", {})
    _set_if_default(parser, args, "T", target.get("T"))
    _set_if_default(parser, args, "dt", target.get("dt"))
    _set_if_default(parser, args, "target_nu", target.get("nu", target.get("target_nu")))
    _set_if_default(parser, args, "burgers_scheme", target.get("time_integrator", target.get("burgers_scheme")))
    if target.get("dealias") is not None and getattr(args, "burgers_dealias", None) == parser.get_default("burgers_dealias"):
        args.burgers_dealias = int(bool(target.get("dealias")))
    _set_if_default(parser, args, "ntrain", data.get("ntrain"))
    _set_if_default(parser, args, "nval", data.get("nval"))
    _set_if_default(parser, args, "ntest", data.get("ntest"))
    _set_if_default(parser, args, "data_seed", data.get("data_seed"))
    _set_if_default(parser, args, "split_seed", data.get("split_seed"))
    _set_if_default(parser, args, "data_dtype", data.get("data_dtype"))
    _set_if_default(parser, args, "sim_dtype", data.get("sim_dtype"))
    _set_if_default(parser, args, "ridge_convention", readout.get("ridge_convention"))
    _set_if_default(parser, args, "ridge_dtype", readout.get("ridge_dtype"))
    args.expected_ic_type = data.get("ic_type")
    args.expected_solver = target.get("solver")
    args.expected_time_integrator = target.get("time_integrator")
    args.expected_burgers_scheme = target.get("burgers_scheme")
    args.expected_dealias = target.get("dealias")
    args.expected_equation = target.get("equation", "burgers")
    if domain.get("length") is not None:
        args.expected_domain_length = domain.get("length")


def ridge_dtype_from_name(name):
    return torch.float32 if name == "float32" else torch.float64


def _extract_scalar_meta(reader, field):
    if field not in reader.data:
        return None
    return reader.read_scalar_meta(field)


def _validate_shapes(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
):
    if x_train.shape[1] != y_train.shape[1]:
        raise ValueError("Train a/u resolution mismatch: %s vs %s" % (x_train.shape[1], y_train.shape[1]))
    if x_test.shape[1] != y_test.shape[1]:
        raise ValueError("Test a/u resolution mismatch: %s vs %s" % (x_test.shape[1], y_test.shape[1]))
    if x_val is not None and x_val.shape[1] != y_val.shape[1]:
        raise ValueError("Val a/u resolution mismatch: %s vs %s" % (x_val.shape[1], y_val.shape[1]))
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train/test input resolution mismatch: %s vs %s" % (x_train.shape[1], x_test.shape[1]))
    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError("Train/test output resolution mismatch: %s vs %s" % (y_train.shape[1], y_test.shape[1]))
    if x_val is not None and x_train.shape[1] != x_val.shape[1]:
        raise ValueError("Train/val input resolution mismatch: %s vs %s" % (x_train.shape[1], x_val.shape[1]))


def _hash_indices(indices, split_name: str = "") -> str:
    arr = np.asarray(indices, dtype=np.int64)
    h = stable_json_hash({"split": split_name, "indices": arr.tolist()})
    return h


def _validate_target_time(args, train_reader, test_reader=None):
    train_T = _extract_scalar_meta(train_reader, "T")
    test_T = _extract_scalar_meta(test_reader, "T") if test_reader is not None else train_T
    available = [val for val in (train_T, test_T) if val is not None]
    if not available:
        warnings.warn(
            "Dataset does not provide T metadata; --T cannot be validated against the target file.",
            RuntimeWarning,
        )
        return
    if train_T is not None and test_T is not None and not np.isclose(train_T, test_T):
        raise ValueError("Train/test target-time metadata mismatch: %s vs %s" % (train_T, test_T))
    data_T = available[0]
    if not np.isclose(data_T, args.T):
        raise ValueError(
            "Requested --T=%s but dataset target time is T=%s. "
            "The CLI target time must match the dataset target."
            % (args.T, data_T)
        )


def _meta_nu_from_path(path):
    if not path:
        return None
    try:
        meta = _load_dataset_file(path).get("metadata", {})
        normalized = normalize_dataset_metadata(meta)
        value = normalized.get("target_nu")
        return float(value) if value is not None else None
    except Exception:
        return None


def _load_dataset_file(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".mat":
        with MatReader(path) as reader:
            return {
                "kind": "raw",
                "a": reader.read_field("a"),
                "u": reader.read_field("u"),
                "metadata": {
                    field: _extract_scalar_meta(reader, field)
                    for field in [
                        "T",
                        "dt",
                        "nu",
                        "target_nu",
                        "nx",
                        "ic_type",
                        "domain_length",
                        "solver",
                        "time_integrator",
                        "burgers_scheme",
                        "dealias",
                        "equation",
                        "target_equation",
                    ]
                    if field in reader.data
                },
            }
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("metadata", payload.get("config", {})) if isinstance(payload, dict) else {}
        if "a" in payload and "u" in payload:
            return {"kind": "raw", "a": payload["a"], "u": payload["u"], "metadata": meta}
        if "u0_train" in payload and "y_train" in payload:
            return {
                "kind": "split",
                "x_train": payload["u0_train"],
                "y_train": payload["y_train"],
                "x_val": payload.get("u0_val", payload["u0_train"][:0]),
                "y_val": payload.get("y_val", payload["y_train"][:0]),
                "x_test": payload["u0_test"],
                "y_test": payload["y_test"],
                "metadata": meta,
            }
        raise ValueError(f"Unsupported .pt dataset payload keys: {sorted(payload.keys())}")
    raise ValueError(f"Unsupported dataset extension for {path}; expected .mat or .pt")


def _expected_metadata(args, nx_after_sub: int | None = None) -> dict:
    return {
        "T": args.T,
        "dt": args.dt,
        "nx": nx_after_sub,
        "target_nu": getattr(args, "target_nu", None),
        "ic_type": getattr(args, "expected_ic_type", None),
        "domain_length": getattr(args, "expected_domain_length", 1.0),
        "solver": getattr(args, "expected_solver", None),
        "time_integrator": getattr(args, "expected_time_integrator", None),
        "burgers_scheme": getattr(args, "expected_burgers_scheme", None),
        "dealias": getattr(args, "expected_dealias", None),
        "target_equation": getattr(args, "expected_equation", None),
    }


def _validate_dataset_metadata(args, metadata, nx_after_sub: int | None = None):
    strict = not bool(args.allow_metadata_mismatch)
    try:
        result = validate_dataset_metadata(
            raw_metadata=metadata,
            expected=_expected_metadata(args, nx_after_sub=nx_after_sub),
            strict=strict,
            require_complete_metadata=bool(getattr(args, "require_complete_metadata", False)),
        )
    except ValueError:
        raise
    for message in result["warnings"]:
        warnings.warn(message, RuntimeWarning)
    if not strict and not result["ok"]:
        warnings.warn("Dataset metadata mismatch allowed by --allow-metadata-mismatch.", RuntimeWarning)
    return result


def _cast_data_tensor(tensor: torch.Tensor, data_dtype: str) -> torch.Tensor:
    if data_dtype == "preserve":
        return tensor
    return tensor.to(dtype=torch.float32 if data_dtype == "float32" else torch.float64)


def _dataset_target_nu_from_args(args):
    paths = []
    if getattr(args, "data_mode", "single_split") == "single_split":
        paths.append(getattr(args, "data_file", None))
    else:
        paths.extend([getattr(args, "train_file", None), getattr(args, "test_file", None)])
    values = []
    for path in paths:
        value = _meta_nu_from_path(path)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    first = values[0]
    for value in values[1:]:
        if not np.isclose(first, value):
            raise ValueError("Dataset target_nu metadata mismatch across files: %s vs %s" % (first, value))
    return first


def resolve_defect_target_nu(args):
    if getattr(args, "defect_target_nu", None) is not None:
        return {"target_nu": float(args.defect_target_nu), "target_nu_source": "defect_target_nu", "target_nu_warning": None}
    if getattr(args, "target_nu", None) is not None:
        return {"target_nu": float(args.target_nu), "target_nu_source": "args.target_nu", "target_nu_warning": None}
    dataset_nu = _dataset_target_nu_from_args(args)
    if dataset_nu is not None:
        return {"target_nu": float(dataset_nu), "target_nu_source": "dataset_metadata", "target_nu_warning": None}
    warning = "Dataset/config did not provide target_nu for defect diagnostics; using fallback target_nu=0.05."
    warnings.warn(warning, RuntimeWarning)
    return {"target_nu": 0.05, "target_nu_source": "fallback", "target_nu_warning": warning}


def load_data(args):
    split_meta = {
        "ntrain": int(args.ntrain),
        "nval": int(args.nval),
        "ntest": int(args.ntest),
        "data_seed": int(args.data_seed),
        "split_seed": int(args.split_seed),
        "split_policy": "deterministic_permutation" if args.shuffle or args.nval > 0 else "legacy_train_split",
    }
    if args.data_mode == "single_split":
        loaded = _load_dataset_file(args.data_file)
        dataset_meta = loaded["metadata"]
        if loaded["kind"] == "split":
            raw_nx = int(loaded["x_train"].shape[-1])
            x_train_full = loaded["x_train"][:, :: args.sub]
            y_train_full = loaded["y_train"][:, :: args.sub]
            x_val_full = loaded["x_val"][:, :: args.sub]
            y_val_full = loaded["y_val"][:, :: args.sub]
            x_test_full = loaded["x_test"][:, :: args.sub]
            y_test_full = loaded["y_test"][:, :: args.sub]
            if args.ntrain > x_train_full.shape[0] or args.nval > x_val_full.shape[0] or args.ntest > x_test_full.shape[0]:
                raise ValueError(
                    "Not enough split dataset samples for ntrain=%s, nval=%s, ntest=%s; available=%s/%s/%s"
                    % (args.ntrain, args.nval, args.ntest, x_train_full.shape[0], x_val_full.shape[0], x_test_full.shape[0])
                )
            x_train = x_train_full[: args.ntrain]
            y_train = y_train_full[: args.ntrain]
            x_val = x_val_full[: args.nval]
            y_val = y_val_full[: args.nval]
            x_test = x_test_full[: args.ntest]
            y_test = y_test_full[: args.ntest]
            train_idx = np.arange(args.ntrain)
            val_idx = np.arange(args.nval)
            test_idx = np.arange(args.ntest)
            split_meta["split_policy"] = "pre_split_dataset"
            effective_nx = int(x_train.shape[1])
            split_meta.update({"dataset_nx": raw_nx, "raw_nx": raw_nx, "effective_nx": effective_nx, "sub": int(args.sub)})
            metadata_validation = _validate_dataset_metadata(args, dataset_meta, nx_after_sub=raw_nx)
            _validate_shapes(x_train, y_train, x_val, y_val, x_test, y_test)
            split_meta.update(
                {
                    "train_indices_hash": _hash_indices(train_idx, "train"),
                    "val_indices_hash": _hash_indices(val_idx, "val"),
                    "test_indices_hash": _hash_indices(test_idx, "test"),
                }
            )
            s = int(x_train.shape[1])
            return (
                _cast_data_tensor(x_train.reshape(args.ntrain, s), args.data_dtype),
                _cast_data_tensor(y_train.reshape(args.ntrain, s), args.data_dtype),
                _cast_data_tensor(x_val.reshape(args.nval, s), args.data_dtype),
                _cast_data_tensor(y_val.reshape(args.nval, s), args.data_dtype),
                _cast_data_tensor(x_test.reshape(args.ntest, s), args.data_dtype),
                _cast_data_tensor(y_test.reshape(args.ntest, s), args.data_dtype),
                split_meta,
                dataset_meta,
                metadata_validation,
            )

        raw_nx = int(loaded["a"].shape[-1])
        x_data = loaded["a"][:, :: args.sub]
        y_data = loaded["u"][:, :: args.sub]
        effective_nx = int(x_data.shape[1])
        split_meta.update({"dataset_nx": raw_nx, "raw_nx": raw_nx, "effective_nx": effective_nx, "sub": int(args.sub)})
        metadata_validation = _validate_dataset_metadata(args, dataset_meta, nx_after_sub=raw_nx)
        total = x_data.shape[0]
        indices = np.arange(total)
        if args.shuffle or args.nval > 0:
            rng = np.random.default_rng(args.split_seed)
            rng.shuffle(indices)
            if args.ntrain + args.nval + args.ntest > total:
                raise ValueError(
                    "Not enough samples for ntrain=%s, nval=%s, ntest=%s, total=%s"
                    % (args.ntrain, args.nval, args.ntest, total)
                )
            train_idx = indices[: args.ntrain]
            val_idx = indices[args.ntrain : args.ntrain + args.nval]
            test_idx = indices[args.ntrain + args.nval : args.ntrain + args.nval + args.ntest]
        else:
            split_idx = int(total * args.train_split)
            train_pool = indices[:split_idx]
            test_pool = indices[split_idx:]
            if args.ntrain > len(train_pool) or args.ntest > len(test_pool):
                raise ValueError("Not enough samples for ntrain=%s, ntest=%s, total=%s" % (args.ntrain, args.ntest, total))
            train_idx = train_pool[: args.ntrain]
            val_idx = np.asarray([], dtype=np.int64)
            test_idx = test_pool[: args.ntest]
        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_val = x_data[val_idx] if args.nval > 0 else x_data[train_idx[:0]]
        y_val = y_data[val_idx] if args.nval > 0 else y_data[train_idx[:0]]
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]
    else:
        with MatReader(args.train_file) as train_reader, MatReader(args.test_file) as test_reader:
            _validate_target_time(args, train_reader, test_reader)
            train_a = train_reader.read_field("a")
            train_u = train_reader.read_field("u")
            test_a = test_reader.read_field("a")
            test_u = test_reader.read_field("u")
            raw_nx = int(train_a.shape[-1])
            x_train = train_a[: args.ntrain, :: args.sub]
            y_train = train_u[: args.ntrain, :: args.sub]
            x_val = train_a[args.ntrain : args.ntrain + args.nval, :: args.sub]
            y_val = train_u[args.ntrain : args.ntrain + args.nval, :: args.sub]
            x_test = test_a[: args.ntest, :: args.sub]
            y_test = test_u[: args.ntest, :: args.sub]
            dataset_meta = {
                "T": _extract_scalar_meta(train_reader, "T"),
                "dt": _extract_scalar_meta(train_reader, "dt"),
                "nu": _extract_scalar_meta(train_reader, "nu"),
                "nx": _extract_scalar_meta(train_reader, "nx"),
            }
            effective_nx = int(x_train.shape[1])
            split_meta.update({"dataset_nx": raw_nx, "raw_nx": raw_nx, "effective_nx": effective_nx, "sub": int(args.sub)})
            metadata_validation = _validate_dataset_metadata(args, dataset_meta, nx_after_sub=raw_nx)
        train_idx = np.arange(args.ntrain)
        val_idx = np.arange(args.ntrain, args.ntrain + args.nval)
        test_idx = np.arange(args.ntest)
    _validate_shapes(x_train, y_train, x_val, y_val, x_test, y_test)
    split_meta.update(
        {
            "train_indices_hash": _hash_indices(train_idx, "train"),
            "val_indices_hash": _hash_indices(val_idx, "val"),
            "test_indices_hash": _hash_indices(test_idx, "test"),
        }
    )
    s = int(x_train.shape[1])
    return (
        _cast_data_tensor(x_train.reshape(args.ntrain, s), args.data_dtype),
        _cast_data_tensor(y_train.reshape(args.ntrain, s), args.data_dtype),
        _cast_data_tensor(x_val.reshape(args.nval, s), args.data_dtype),
        _cast_data_tensor(y_val.reshape(args.nval, s), args.data_dtype),
        _cast_data_tensor(x_test.reshape(args.ntest, s), args.data_dtype),
        _cast_data_tensor(y_test.reshape(args.ntest, s), args.data_dtype),
        split_meta,
        dataset_meta,
        metadata_validation,
    )


def build_model_config(args):
    domain_length = float(getattr(args, "expected_domain_length", None) or 1.0)
    return Model123Config(
        reservoir=args.reservoir,
        Ttilde=args.Ttilde,
        dt=args.dt,
        K=args.K,
        feature_times=args.feature_times,
        obs=args.obs,
        J=args.J,
        sensor_mode=args.sensor_mode,
        sensor_seed=args.sensor_seed,
        input_scale=args.input_scale,
        input_shift=args.input_shift,
        ridge_lambda=args.ridge_lambda,
        ridge_zeta=args.ridge_zeta,
        ridge_convention=args.ridge_convention,
        ridge_dtype=ridge_dtype_from_name(args.ridge_dtype),
        standardize_features=bool(args.standardize_features),
        feature_std_eps=args.feature_std_eps,
        elm_hidden_dim=args.elm_h,
        elm_activation=args.elm_activation,
        elm_seed=args.elm_seed,
        elm_weight_scale=args.elm_weight_scale,
        elm_bias_scale=args.elm_bias_scale,
        rd_nu=args.rd_nu,
        rd_alpha=args.rd_alpha,
        rd_beta=args.rd_beta,
        res_burgers_nu=args.res_burgers_nu,
        res_burgers_b=args.res_burgers_b,
        heat_nu=args.heat_nu,
        advection_c=args.advection_c,
        ks_dealias=args.ks_dealias,
        ks_b=args.ks_b,
        ks_eta=args.ks_eta,
        ks_kappa=args.ks_kappa,
        burgers_scheme=args.burgers_scheme,
        burgers_fine_dt=args.burgers_fine_dt,
        burgers_dealias=bool(args.burgers_dealias),
        device=args.device,
        dtype=torch.float32 if args.sim_dtype == "float32" else torch.float64,
        domain_length=domain_length,
    )


@torch.no_grad()
def evaluate_model(model, loader, progress_label=None):
    preds = []
    ys = []
    xs = []
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    start_time = time.perf_counter()
    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        pred = model.predict(xb).cpu()
        preds.append(pred)
        ys.append(yb)
        xs.append(xb)
        if progress_label is not None:
            if total_batches is None:
                print("[%s] batch %d" % (progress_label, batch_idx), flush=True)
            else:
                print("[%s] batch %d/%d" % (progress_label, batch_idx, total_batches), flush=True)
    if progress_label is not None:
        elapsed = time.perf_counter() - start_time
        print("[%s] done in %.2fs" % (progress_label, elapsed), flush=True)
    preds_all = torch.cat(preds)
    ys_all = torch.cat(ys)
    xs_all = torch.cat(xs)
    abs_l2h = dataset_abs_l2h_rmse(preds_all, ys_all)
    rel_l2h_mean = dataset_rel_l2h_mean(preds_all, ys_all)
    rel_l2h_agg = dataset_rel_l2h_aggregate(preds_all, ys_all)
    return abs_l2h, rel_l2h_mean, rel_l2h_agg, preds_all, ys_all, xs_all


def make_progress_fn(label):
    last_emit = {"batch": 0, "time": time.perf_counter()}

    def progress_fn(batch_idx, total_batches):
        now = time.perf_counter()
        should_emit = (
            batch_idx == 1
            or total_batches is None
            or batch_idx == total_batches
            or batch_idx - last_emit["batch"] >= 5
            or (now - last_emit["time"]) >= 5.0
        )
        if not should_emit:
            return
        if total_batches is None:
            print("[%s] batch %d" % (label, batch_idx), flush=True)
        else:
            print("[%s] batch %d/%d" % (label, batch_idx, total_batches), flush=True)
        last_emit["batch"] = batch_idx
        last_emit["time"] = now

    return progress_fn


def _average_ranks(values):
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=float)
    idx = 0
    while idx < arr.shape[0]:
        end = idx + 1
        while end < arr.shape[0] and arr[order[end]] == arr[order[idx]]:
            end += 1
        avg_rank = 0.5 * (idx + end - 1) + 1.0
        ranks[order[idx:end]] = avg_rank
        idx = end
    return ranks


def pearson_corr_or_none(x, y):
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.size != y_arr.size:
        return None
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
        return None
    x_centered = x_arr - float(np.mean(x_arr))
    y_centered = y_arr - float(np.mean(y_arr))
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom <= 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denom)


def spearman_corr_or_none(x, y):
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.size != y_arr.size:
        return None
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
        return None
    if float(np.max(x_arr) - np.min(x_arr)) == 0.0 or float(np.max(y_arr) - np.min(y_arr)) == 0.0:
        return None
    return pearson_corr_or_none(_average_ranks(x_arr), _average_ranks(y_arr))


def _write_dict_rows_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_error_vs_defect(rows, out_path_no_ext):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(
        [float(row["delta_scale_pathwise_abs_l2h"]) for row in rows],
        [float(row["model_error_abs_l2h"]) for row in rows],
        s=22,
        alpha=0.8,
    )
    ax.set_xlabel("delta_scale_pathwise_abs_l2h")
    ax.set_ylabel("model_error_abs_l2h")
    ax.set_title("Error vs integrated generator defect")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)


def compute_and_save_defect_outputs(args, s, x_test_all, y_test_all, per_sample_abs, per_sample_rel):
    target_nu_resolution = resolve_defect_target_nu(args)
    target_nu = float(target_nu_resolution["target_nu"])
    defect_cfg = ErrorDecompositionConfig(
        num_samples=args.ntest,
        nx=s,
        batch_size=args.batch_size,
        target_nu=target_nu,
        T=args.T,
        Ttilde_values=[args.Ttilde],
        dt=args.dt,
        fine_dt=args.burgers_fine_dt,
        reservoir=args.reservoir,
        rd_nu=args.rd_nu,
        rd_alpha=args.rd_alpha,
        rd_beta=args.rd_beta,
        res_burgers_nu=args.res_burgers_nu,
        res_burgers_b=args.res_burgers_b,
        burgers_scheme=args.burgers_scheme,
        burgers_dealias=bool(args.burgers_dealias),
        ks_b=args.ks_b,
        ks_eta=args.ks_eta,
        ks_kappa=args.ks_kappa,
        ks_dealias=args.ks_dealias,
        input_scale=args.input_scale,
        input_shift=args.input_shift,
        dtype=args.defect_dtype,
        device=args.device,
        beta_mode=args.defect_beta_mode,
        beta_fixed=args.defect_beta_fixed,
        time_quadrature=args.defect_time_quadrature,
    )
    result = compute_time_scaled_defect_for_dataset(
        u0=x_test_all,
        target_T=y_test_all,
        cfg=defect_cfg,
        Ttilde=args.Ttilde,
    )
    per_sample_abs_list = [float(v) for v in per_sample_abs.tolist()]
    per_sample_rel_list = [float(v) for v in per_sample_rel.tolist()]
    joined_rows = []
    for idx, defect_row in enumerate(result["rows"]):
        joined_rows.append(
            {
                "model": args.model,
                "sample_index": int(defect_row["sample_index"]),
                "model_error_abs_l2h": per_sample_abs_list[idx],
                "model_error_rel_l2": per_sample_rel_list[idx],
                "D1_model1_abs_l2h": float(defect_row["D1_model1_abs_l2h"]),
                "delta_scale_pathwise_abs_l2h": float(defect_row["delta_scale_pathwise_abs_l2h"]),
                "Delta_scale_abs_l2h": float(defect_row["Delta_scale_abs_l2h"]),
                "T": float(defect_row["T"]),
                "Ttilde": float(defect_row["Ttilde"]),
                "alpha": float(defect_row["alpha"]),
                "beta_mode": defect_row["beta_mode"],
                "beta_value": float(defect_row["beta_value"]),
                "beta_empirical": float(defect_row["beta_empirical"]),
                "c_beta_T": float(defect_row["c_beta_T"]),
            }
        )

    delta_values = [row["delta_scale_pathwise_abs_l2h"] for row in joined_rows]
    error_values = [row["model_error_abs_l2h"] for row in joined_rows]
    metrics = {
        **result["summary"],
        "target_nu": float(target_nu),
        "target_nu_source": target_nu_resolution["target_nu_source"],
        "target_nu_warning": target_nu_resolution.get("target_nu_warning"),
        "model": args.model,
        "corr_error_delta_scale_pearson": pearson_corr_or_none(error_values, delta_values),
        "corr_error_delta_scale_spearman": spearman_corr_or_none(error_values, delta_values),
        "applies_directly_to_model1_bound": bool(args.model == "model1"),
        "defect_interpretation": (
            "model1_time_scaled_integrated_generator_defect"
            if args.model == "model1"
            else "underlying_surrogate_pde_time_scaled_integrated_generator_defect"
        ),
    }
    if args.model == "model1":
        metrics["max_abs_difference_model1_D1"] = float(
            np.max(
                np.abs(
                    np.asarray(error_values, dtype=float)
                    - np.asarray([row["D1_model1_abs_l2h"] for row in joined_rows], dtype=float)
                )
            )
        )

    _write_dict_rows_csv(os.path.join(args.out_dir, "time_scaled_defect_per_sample.csv"), joined_rows)
    with open(os.path.join(args.out_dir, "time_scaled_defect_per_sample.json"), "w", encoding="utf-8") as f:
        json.dump(joined_rows, f, indent=2)
    with open(os.path.join(args.out_dir, "time_scaled_defect_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _plot_error_vs_defect(joined_rows, os.path.join(args.out_dir, "error_vs_defect_scatter"))
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stage_start = time.perf_counter()
    x_train, y_train, x_val, y_val, x_test, y_test, split_meta, dataset_meta, metadata_validation = load_data(args)
    print("[%s] data loaded in %.2fs" % (args.model, time.perf_counter() - stage_start), flush=True)
    s = int(x_train.shape[1])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    ) if args.nval > 0 else None
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_cfg = build_model_config(args)
    if args.model == "model1":
        model = Model1Predictor1D(s=s, config=model_cfg)
        ridge_state = None
    elif args.model == "model2":
        model = Model2Regressor1D(s=s, config=model_cfg)
        ridge_state = model.fit(
            train_loader,
            progress_fn=make_progress_fn("%s train" % args.model),
            progress_label="%s train" % args.model,
        )
    else:
        model = Model3Regressor1D(s=s, config=model_cfg)
        ridge_state = model.fit(
            train_loader,
            progress_fn=make_progress_fn("%s train" % args.model),
            progress_label="%s train" % args.model,
        )

    train_eval_label = "%s eval-train" % args.model if args.model in {"model2", "model3"} else None
    test_eval_label = "%s eval-test" % args.model if args.model in {"model2", "model3"} else None
    train_abs, train_rel_mean, train_rel_agg, _, _, _ = evaluate_model(model, eval_train_loader, progress_label=train_eval_label)
    if val_loader is not None:
        val_eval_label = "%s eval-val" % args.model if args.model in {"model2", "model3"} else None
        val_abs, val_rel_mean, val_rel_agg, _, _, _ = evaluate_model(model, val_loader, progress_label=val_eval_label)
    else:
        val_abs = None
        val_rel_mean = None
        val_rel_agg = None
    test_abs, test_rel_mean, test_rel_agg, pred_test, y_test_all, x_test_all = evaluate_model(
        model,
        test_loader,
        progress_label=test_eval_label,
    )

    resolved_cfg = getattr(model, "config", model_cfg)
    actual_obs = resolved_cfg.obs
    actual_J = resolved_cfg.J
    print("model=%s reservoir=%s obs=%s J=%s" % (args.model, args.reservoir, actual_obs, actual_J))
    print("T=%s Ttilde=%s dt=%s" % (args.T, args.Ttilde, args.dt))
    print("train absL2h: %.6f" % train_abs)
    if val_abs is not None:
        print("val   absL2h: %.6f" % val_abs)
    print("test  absL2h: %.6f" % test_abs)
    print("train relL2_mean: %.6f" % train_rel_mean)
    if val_rel_mean is not None:
        print("val   relL2_mean: %.6f" % val_rel_mean)
    print("test  relL2_mean: %.6f" % test_rel_mean)

    per_sample_abs = per_sample_abs_l2h_error(pred_test, y_test_all)
    per_sample_rel = per_sample_rel_l2h_error(pred_test, y_test_all)
    plot_error_histogram([float(v) for v in per_sample_abs.tolist()], os.path.join(args.out_dir, "test_absL2h_hist"))
    plot_error_histogram([float(v) for v in per_sample_rel.tolist()], os.path.join(args.out_dir, "test_relL2_hist"))
    with open(os.path.join(args.out_dir, "test_error_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "main_metric": "abs_l2h",
                "test_absL2h": test_abs,
                "test_relL2": test_rel_mean,
                "test_relL2_mean": test_rel_mean,
                "test_relL2_agg": test_rel_agg,
                "per_sample_absL2h": [float(v) for v in per_sample_abs.tolist()],
                "per_sample_relL2": [float(v) for v in per_sample_rel.tolist()],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    defect_metrics = None
    if args.compute_time_scaled_defect:
        defect_metrics = compute_and_save_defect_outputs(
            args,
            s,
            x_test_all,
            y_test_all,
            per_sample_abs,
            per_sample_rel,
        )

    x_grid = np.linspace(0.0, 1.0, s, endpoint=False)
    for idx in [0, min(1, args.ntest - 1), min(2, args.ntest - 1)]:
        plot_1d_prediction(
            x=x_grid,
            gt=y_test_all[idx],
            pred=pred_test[idx],
            input_u0=x_test_all[idx],
            out_path_no_ext=os.path.join(args.out_dir, "sample_%03d" % idx),
            title_prefix="%s sample %d: " % (args.model, idx),
        )

    if args.save_model:
        save_path = args.save_model
        if save_path == "model.pt":
            save_path = os.path.join(args.out_dir, save_path)
        state = {
            "args": vars(args),
            "model_config": asdict(resolved_cfg),
        }
        if ridge_state is not None:
            for key, value in ridge_state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.detach().cpu()
        if hasattr(model, "weight") and getattr(model, "weight", None) is not None:
            state["W_out"] = model.weight.detach().cpu()
        if hasattr(model, "elm") and getattr(model, "elm", None) is not None:
            state["elm_weight"] = model.elm.weight.detach().cpu()
            state["elm_bias"] = model.elm.bias.detach().cpu()
            state["elm_activation"] = model.elm.activation
        torch.save(state, save_path)
        print("saved model: %s" % save_path)

    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        domain_length = float(getattr(args, "expected_domain_length", None) or 1.0)
        dataset_nx = split_meta.get("dataset_nx", split_meta.get("raw_nx", int(s) * int(args.sub)))
        effective_nx = split_meta.get("effective_nx", int(s))
        run_payload = {
            "args": vars(args),
            "git": get_git_info(Path(__file__).resolve().parent),
            **get_runtime_info(),
            "resolved_obs": actual_obs,
            "resolved_J": actual_J,
            "resolved_config": {
                "target": {
                    "equation": "burgers",
                    "target_nu": args.target_nu,
                    "T": args.T,
                    "dt": args.dt,
                    "nx": int(dataset_nx),
                    "effective_nx": int(effective_nx),
                    "domain_length": domain_length,
                    "solver": args.expected_solver,
                    "dealias": args.expected_dealias,
                    "ic_type": args.expected_ic_type,
                },
                "surrogate": asdict(resolved_cfg),
                "readout": {
                    "ridge_zeta": float(args.ridge_zeta),
                    "ridge_lambda_legacy_input": args.ridge_lambda,
                    "ridge_convention": args.ridge_convention,
                    "ridge_dtype": args.ridge_dtype,
                },
            },
            "config_overrides": getattr(args, "_config_applied", {}),
            "main_metric": "abs_l2h",
            "alpha": float(args.Ttilde / args.T),
            "train_absL2h": train_abs,
            "val_absL2h": val_abs,
            "test_absL2h": test_abs,
            "train_relL2": train_rel_mean,
            "val_relL2": val_rel_mean,
            "test_relL2": test_rel_mean,
            "train_relL2_mean": train_rel_mean,
            "val_relL2_mean": val_rel_mean,
            "test_relL2_mean": test_rel_mean,
            "train_relL2_agg": train_rel_agg,
            "val_relL2_agg": val_rel_agg,
            "test_relL2_agg": test_rel_agg,
            "split": split_meta,
            "metadata_validation": metadata_validation,
            "selection": {
                "selection_metric": "val_absL2h" if val_abs is not None else "test_absL2h_legacy_fallback",
                "selected_by": "single_run",
                "warning_if_legacy_test_selection": val_abs is None,
            },
            "readout": {
                "ridge_parameter_name": "zeta",
                "ridge_zeta": float(args.ridge_zeta),
                "ridge_convention": args.ridge_convention,
                "ridge_dtype": args.ridge_dtype,
                "regularize_bias": False,
                "standardize_features": bool(args.standardize_features),
            },
            "ridge_parameter_name": "zeta",
            "ridge_zeta": float(args.ridge_zeta),
            "ridge_lambda_legacy_input": args.ridge_lambda,
            "ridge_convention": args.ridge_convention,
            "regularize_bias": False,
            "domain_length": domain_length,
            "dataset_nx": int(dataset_nx),
            "raw_nx": int(dataset_nx),
            "effective_nx": int(effective_nx),
            "grid": {
                "dataset_nx": int(dataset_nx),
                "raw_nx": int(dataset_nx),
                "effective_nx": int(effective_nx),
                "sub": int(args.sub),
                "domain_length": domain_length,
            },
            "nx": int(s),
            "dx": float(domain_length / s),
            "num_train_samples": int(args.ntrain),
            "effective_code_lambda_legacy_equivalent": float(args.ntrain * args.ridge_zeta * s),
            "data_file": args.data_file if args.data_mode == "single_split" else args.train_file,
            "data_sha256": file_sha256(args.data_file if args.data_mode == "single_split" else args.train_file),
            "dataset_metadata": normalize_dataset_metadata(dataset_meta),
            "config_name": Path(args.config).stem if args.config else None,
            "config_path": args.config or None,
            "config_hash": file_sha256(args.config) if args.config else None,
            "config_applied": getattr(args, "_config_applied", {}),
            "command_line": get_command_line(),
            "device": args.device,
            "dtype": {
                "data_dtype": args.data_dtype,
                "sim_dtype": args.sim_dtype,
                "ridge_dtype": args.ridge_dtype,
                "input_tensor_dtype": str(x_train.dtype).replace("torch.", ""),
                "feature_tensor_dtype": str(resolved_cfg.dtype).replace("torch.", ""),
                "ridge_tensor_dtype": args.ridge_dtype,
            },
            "target": {
                "equation": "burgers",
                "target_nu": args.target_nu,
                "T": args.T,
                "dt": args.dt,
                "nx": int(dataset_nx),
                "effective_nx": int(effective_nx),
                "domain_length": domain_length,
                "solver": args.expected_solver,
                "dealias": args.expected_dealias,
                "ic_type": args.expected_ic_type,
            },
        }
        if ridge_state is not None:
            for src, dst in [
                ("W_fro_norm", "W_fro_norm"),
                ("W_hs_norm_l2h", "W_hs_norm_l2h"),
                ("d_eff", "effective_dimension"),
                ("cond_zeta", "condition_number"),
            ]:
                if src in ridge_state:
                    run_payload[dst] = to_jsonable(ridge_state[src])
        if defect_metrics is not None:
            run_payload.update(
                {
                    "time_scaled_defect_metric": defect_metrics["defect_metric"],
                    "defect_target_nu": defect_metrics["target_nu"],
                    "defect_target_nu_source": defect_metrics["target_nu_source"],
                    "defect_target_nu_warning": defect_metrics.get("target_nu_warning"),
                    "delta_scale_rms_abs_l2h": defect_metrics["delta_scale_rms_abs_l2h"],
                    "delta_scale_mean_abs_l2h": defect_metrics["delta_scale_mean_abs_l2h"],
                    "delta_scale_std_abs_l2h": defect_metrics["delta_scale_std_abs_l2h"],
                    "corr_error_delta_scale_pearson": defect_metrics["corr_error_delta_scale_pearson"],
                    "corr_error_delta_scale_spearman": defect_metrics["corr_error_delta_scale_spearman"],
                    "applies_directly_to_model1_bound": defect_metrics["applies_directly_to_model1_bound"],
                    "defect_interpretation": defect_metrics["defect_interpretation"],
                }
            )
            if "max_abs_difference_model1_D1" in defect_metrics:
                run_payload["max_abs_difference_model1_D1"] = defect_metrics["max_abs_difference_model1_D1"]
        json.dump(
            to_jsonable(run_payload),
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
