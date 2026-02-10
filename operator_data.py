from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class DatasetBundle:
    grid_train: Optional[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = None
    grid_test: Optional[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = None
    points_train: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] = None
    points_test: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] = None


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_mat_dict(path: str, keys: list[str]) -> dict[str, torch.Tensor]:
    from utilities3 import MatReader

    reader = MatReader(path)
    return {k: reader.read_field(k) for k in keys}


def _coords_from_shape(spatial_shape: tuple[int, ...]) -> torch.Tensor:
    axes = [torch.linspace(0.0, 1.0, n) for n in spatial_shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)


def _pick_indices(total: int, num_pick: int, strategy: str, rng: np.random.Generator) -> np.ndarray:
    num_pick = int(max(1, min(num_pick, total)))
    if strategy == "uniform_idx":
        return np.linspace(0, total - 1, num=num_pick, dtype=np.int64)
    if strategy in {"random_idx", "random", "random_xy"}:
        idx = rng.choice(total, size=num_pick, replace=False)
        idx.sort()
        return idx.astype(np.int64)
    raise ValueError(f"Unknown strategy: {strategy}")


def mat_to_grid(
    mat_dict: dict[str, Any],
    problem: str,
    *,
    sub: int = 1,
    r: int = 1,
    grid_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    p = problem.lower()
    if p in {"burgers", "fourier_1d", "1d"}:
        a = _to_tensor(mat_dict["a"])[:, ::sub]
        u = _to_tensor(mat_dict["u"])[:, ::sub]
        a_grid = a.unsqueeze(-1)
        u_grid = u.unsqueeze(-1)
        meta = {
            "problem": "burgers",
            "spatial_shape": (a_grid.shape[1],),
            "coords": _coords_from_shape((a_grid.shape[1],)),
        }
        return a_grid, u_grid, meta

    if p in {"darcy", "fourier_2d", "2d"}:
        coeff = _to_tensor(mat_dict["coeff"])
        sol = _to_tensor(mat_dict["sol"])
        if r > 1:
            coeff = coeff[:, ::r, ::r]
            sol = sol[:, ::r, ::r]
        if grid_size is not None:
            h = int(((grid_size - 1) / max(r, 1)) + 1)
            coeff = coeff[:, :h, :h]
            sol = sol[:, :h, :h]
        a_grid = coeff.unsqueeze(-1)
        u_grid = sol.unsqueeze(-1)
        meta = {
            "problem": "darcy",
            "spatial_shape": (a_grid.shape[1], a_grid.shape[2]),
            "coords": _coords_from_shape((a_grid.shape[1], a_grid.shape[2])),
        }
        return a_grid, u_grid, meta

    raise ValueError(f"Unsupported mat problem: {problem}")


def mat_to_points(
    mat_dict: dict[str, Any],
    problem: str,
    *,
    num_sensors: int = 128,
    sensor_strategy: str = "uniform_idx",
    num_query: Optional[int] = 256,
    query_strategy: str = "random",
    seed: int = 0,
    sub: int = 1,
    r: int = 1,
    grid_size: Optional[int] = None,
    sensor_indices: Optional[np.ndarray] = None,
    query_indices: Optional[np.ndarray] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    a_grid, u_grid, meta = mat_to_grid(mat_dict, problem, sub=sub, r=r, grid_size=grid_size)

    bsz = a_grid.shape[0]
    spatial_shape = tuple(a_grid.shape[1:-1])
    total_points = int(np.prod(spatial_shape))

    coords = meta["coords"]
    a_flat = a_grid.reshape(bsz, total_points, a_grid.shape[-1])
    u_flat = u_grid.reshape(bsz, total_points, u_grid.shape[-1])

    if sensor_indices is None:
        sensor_indices = _pick_indices(total_points, num_sensors, sensor_strategy, rng)
    sensor_indices = np.asarray(sensor_indices, dtype=np.int64)

    if query_indices is None:
        if query_strategy == "all_grid" or num_query is None:
            query_indices = np.arange(total_points, dtype=np.int64)
        else:
            query_indices = _pick_indices(total_points, num_query, query_strategy, rng)
    query_indices = np.asarray(query_indices, dtype=np.int64)

    x_query = coords[query_indices]
    u_sensors = a_flat[:, sensor_indices, 0]
    y_query = u_flat[:, query_indices, :]

    out_meta = dict(meta)
    out_meta.update(
        {
            "sensor_indices": sensor_indices,
            "query_indices": query_indices,
            "num_sensors": int(len(sensor_indices)),
            "num_query": int(len(query_indices)),
        }
    )
    return x_query, u_sensors, y_query, out_meta


def eon_pkl_to_points(pkl_tuple: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(pkl_tuple, (tuple, list)) or len(pkl_tuple) != 3:
        raise ValueError("EON pkl data must be tuple/list: (t, y, u)")
    t, y, u = pkl_tuple
    return _to_tensor(t), _to_tensor(y), _to_tensor(u)


def _meta_get(meta: dict[str, Any], key: str) -> Any:
    if key not in meta:
        raise ValueError(
            f"eon_pkl_to_grid requires meta['{key}']; pass --eon-meta-file with {key}."
        )
    return meta[key]


def _interp_1d_sensor_to_grid(sensor_vals: np.ndarray, sensor_locs: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(sensor_locs)
    xs = sensor_locs[order]
    ys = sensor_vals[order]
    return np.interp(x_grid, xs, ys)


def eon_pkl_to_grid(
    pkl_tuple: Any,
    meta: dict[str, Any],
    *,
    include_time_channel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    t, y, u = eon_pkl_to_points(pkl_tuple)
    x_grid = np.asarray(meta["x_grid"], dtype=np.float32) if "x_grid" in meta else None

    bsz = y.shape[0]
    spatial_shape: tuple[int, ...]

    if y.ndim == 2 and x_grid is not None and y.shape[1] == x_grid.shape[0]:
        spatial_shape = (int(x_grid.shape[0]),)
        u_grid = y.unsqueeze(-1)
    elif y.ndim == 2 and "grid_shape" in meta:
        spatial_shape = tuple(int(v) for v in np.asarray(meta["grid_shape"]).tolist())
        if int(np.prod(spatial_shape)) != y.shape[1]:
            raise ValueError(
                f"meta['grid_shape'] product ({np.prod(spatial_shape)}) must equal y.shape[1] ({y.shape[1]})."
            )
        u_grid = y.reshape(bsz, *spatial_shape, 1)
    else:
        raise ValueError(
            "Could not infer spatial grid for y. Provide meta['x_grid'] compatible with y or meta['grid_shape']."
        )

    sensor_locs = np.asarray(_meta_get(meta, "sensor_locs"), dtype=np.float32)
    if u.shape[1] != sensor_locs.reshape(-1, sensor_locs.shape[-1] if sensor_locs.ndim > 1 else 1).shape[0]:
        raise ValueError(
            f"u.shape[1] ({u.shape[1]}) must match number of sensor locations ({sensor_locs.shape[0]})."
        )

    if len(spatial_shape) == 1:
        if x_grid is None:
            raise ValueError("meta['x_grid'] is required for 1D interpolation from sensors.")
        sensor_1d = sensor_locs.reshape(-1)
        forcing = np.stack(
            [
                _interp_1d_sensor_to_grid(u[i].cpu().numpy(), sensor_1d, x_grid)
                for i in range(bsz)
            ],
            axis=0,
        )
        a_grid = torch.tensor(forcing, dtype=torch.float32).unsqueeze(-1)
    else:
        grid_prod = int(np.prod(spatial_shape))
        if u.shape[1] == grid_prod:
            a_grid = u.reshape(bsz, *spatial_shape, 1)
        else:
            mean_force = u.mean(dim=1, keepdim=True)
            a_grid = mean_force.reshape(bsz, *([1] * len(spatial_shape)), 1).expand(bsz, *spatial_shape, 1)

    if include_time_channel:
        t_channel = t
        if t_channel.ndim == 1:
            t_channel = t_channel.unsqueeze(-1)
        if t_channel.shape[-1] < 1:
            raise ValueError("t must include at least one coordinate for time channel.")
        t0 = t_channel[:, 0].reshape(bsz, *([1] * len(spatial_shape)), 1)
        t0 = t0.expand(bsz, *spatial_shape, 1)
        a_grid = torch.cat([a_grid, t0], dim=-1)

    out_meta = {
        "problem": "eon_pkl",
        "spatial_shape": spatial_shape,
        "coords": _coords_from_shape(spatial_shape),
        "x_grid": x_grid,
        "sensor_locs": sensor_locs,
        "include_time_channel": include_time_channel,
    }
    return a_grid, u_grid, out_meta


def load_dataset(args: Any, *, problem: str, as_points: bool) -> DatasetBundle:
    bundle = DatasetBundle()

    if args.dataset_format == "mat":
        if problem == "burgers":
            keys = ["a", "u"]
            kwargs = {"sub": getattr(args, "sub", 1)}
        else:
            keys = ["coeff", "sol"]
            kwargs = {
                "r": getattr(args, "r", 1),
                "grid_size": getattr(args, "grid_size", None),
            }

        if args.data_mode == "single_split":
            mat = load_mat_dict(args.data_file, keys)
            if as_points:
                bundle.points_train = mat_to_points(
                    mat,
                    problem,
                    num_sensors=args.num_sensors,
                    sensor_strategy=args.sensor_strategy,
                    num_query=args.num_query,
                    query_strategy=args.query_strategy,
                    seed=args.seed,
                    **kwargs,
                )
            else:
                bundle.grid_train = mat_to_grid(mat, problem, **kwargs)
            return bundle

        tr = load_mat_dict(args.train_file, keys)
        te = load_mat_dict(args.test_file, keys)
        if as_points:
            bundle.points_train = mat_to_points(
                tr,
                problem,
                num_sensors=args.num_sensors,
                sensor_strategy=args.sensor_strategy,
                num_query=args.num_query,
                query_strategy=args.query_strategy,
                seed=args.seed,
                **kwargs,
            )
            bundle.points_test = mat_to_points(
                te,
                problem,
                num_sensors=args.num_sensors,
                sensor_strategy=args.sensor_strategy,
                num_query=args.num_query,
                query_strategy=args.query_strategy,
                seed=args.seed + 1,
                **kwargs,
            )
        else:
            bundle.grid_train = mat_to_grid(tr, problem, **kwargs)
            bundle.grid_test = mat_to_grid(te, problem, **kwargs)
        return bundle

    eon_file = getattr(args, "eon_data_file", None) or args.data_file
    meta_file = getattr(args, "eon_meta_file", None)
    if not meta_file:
        raise ValueError("--eon-meta-file is required when --dataset-format=eon_pkl")
    meta = load_pickle(meta_file)

    if args.data_mode == "single_split":
        data = load_pickle(eon_file)
        if as_points:
            t, y, u = eon_pkl_to_points(data)
            bundle.points_train = (t, u, y.unsqueeze(-1) if y.ndim == 2 else y, {"problem": "eon_pkl"})
        else:
            bundle.grid_train = eon_pkl_to_grid(data, meta)
        return bundle

    train_file = getattr(args, "eon_train_file", None) or args.train_file
    test_file = getattr(args, "eon_test_file", None) or args.test_file
    train_data = load_pickle(train_file)
    test_data = load_pickle(test_file)

    if as_points:
        t_tr, y_tr, u_tr = eon_pkl_to_points(train_data)
        t_te, y_te, u_te = eon_pkl_to_points(test_data)
        bundle.points_train = (t_tr, u_tr, y_tr.unsqueeze(-1) if y_tr.ndim == 2 else y_tr, {"problem": "eon_pkl"})
        bundle.points_test = (t_te, u_te, y_te.unsqueeze(-1) if y_te.ndim == 2 else y_te, {"problem": "eon_pkl"})
    else:
        bundle.grid_train = eon_pkl_to_grid(train_data, meta)
        bundle.grid_test = eon_pkl_to_grid(test_data, meta)
    return bundle
