from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import torch

from pol.metadata import normalize_meta_value

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


class MatReader:
    """Small MATLAB/HDF5 reader retained from the legacy utilities module."""

    def __init__(
        self,
        file_path: str | Path,
        to_torch: bool = True,
        to_cuda: bool = False,
        to_float: bool = True,
    ) -> None:
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = str(file_path)
        self.data = None
        self.old_mat = False
        self._load_file()

    def _load_file(self) -> None:
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            if h5py is None:
                raise
            self.data = h5py.File(self.file_path, "r")
            self.old_mat = False

    def load_file(self, file_path: str | Path) -> None:
        self.close()
        self.file_path = str(file_path)
        self._load_file()

    def read_field(self, field: str):
        if self.data is None:
            raise RuntimeError("No file loaded")
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            try:
                x = x.astype(np.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Field {field!r} is not numeric; use read_meta/read_scalar_meta for MATLAB metadata fields."
                ) from exc
        if self.to_torch:
            tensor = torch.from_numpy(np.asarray(x))
            if self.to_cuda:
                tensor = tensor.cuda()
            return tensor
        return x

    def read_raw(self, field: str):
        if self.data is None:
            raise RuntimeError("No file loaded")
        value = self.data[field]
        if not self.old_mat:
            value = value[()]
        return value

    def read_meta(self, field: str):
        return self.read_raw(field)

    def read_scalar_meta(self, field: str):
        return normalize_meta_value(self.read_meta(field))

    def set_cuda(self, to_cuda: bool) -> None:
        self.to_cuda = to_cuda

    def set_torch(self, to_torch: bool) -> None:
        self.to_torch = to_torch

    def set_float(self, to_float: bool) -> None:
        self.to_float = to_float

    def close(self) -> None:
        if self.data is not None and not self.old_mat and hasattr(self.data, "close"):
            self.data.close()
        self.data = None

    def __enter__(self) -> "MatReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
