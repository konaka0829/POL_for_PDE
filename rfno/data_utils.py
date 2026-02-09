from __future__ import annotations

import numpy as np
import scipy.io
import torch

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


class MatReader:
    def __init__(self, file_path: str, to_float: bool = True):
        self.file_path = file_path
        self.to_float = to_float
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self) -> None:
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            if h5py is None:
                raise ImportError(
                    "h5py is required to read this .mat file format. "
                    "Install h5py or use generated data (--data-source generate)."
                )
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path: str) -> None:
        self.file_path = file_path
        self._load_file()

    def read_field(self, field: str) -> torch.Tensor:
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        return torch.from_numpy(x)


class UnitGaussianNormalizer:
    def __init__(self, x: torch.Tensor, eps: float = 1e-5):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)
        return x * (std + self.eps) + mean
