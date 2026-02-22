import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP1D(nn.Module):
    """Pointwise MLP residual model.

    Input:  x of shape (B, S, C)
    Output: r of shape (B, S, T)
    """

    def __init__(self, in_channels: int, out_channels: int, width: int = 256, depth: int = 4, include_x: bool = False):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if width <= 0 or depth < 1:
            raise ValueError("width must be positive and depth must be >= 1.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.include_x = bool(include_x)
        c_in = self.in_channels + (1 if self.include_x else 0)

        layers: list[nn.Module] = []
        layers.append(nn.Linear(c_in, width))
        for _ in range(depth - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(width, width))
        layers.append(nn.GELU())
        layers.append(nn.Linear(width, self.out_channels))
        self.net = nn.Sequential(*layers)

    def _append_grid(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C)
        if not self.include_x:
            return x
        bsz, size, _ = x.shape
        grid = torch.arange(size, device=x.device, dtype=x.dtype) / float(size)
        grid = grid.view(1, size, 1).expand(bsz, -1, -1)
        return torch.cat([x, grid], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ResidualMLP1D expects x with shape (B,S,C), got {tuple(x.shape)}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"ResidualMLP1D input channel mismatch: expected C={self.in_channels}, got C={x.shape[-1]}"
            )

        x = self._append_grid(x)
        bsz, size, cin = x.shape
        y = self.net(x.reshape(bsz * size, cin))
        return y.reshape(bsz, size, self.out_channels)


class ResidualCNN1D(nn.Module):
    """1D CNN residual model.

    Input:  x of shape (B, S, C)
    Output: r of shape (B, S, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 128,
        depth: int = 4,
        kernel_size: int = 5,
        include_x: bool = False,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if width <= 0 or depth < 1:
            raise ValueError("width must be positive and depth must be >= 1.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.include_x = bool(include_x)
        c_in = self.in_channels + (1 if self.include_x else 0)

        padding = kernel_size // 2
        blocks: list[nn.Module] = [nn.Conv1d(c_in, width, kernel_size, padding=padding)]
        for _ in range(depth - 1):
            blocks.append(nn.GELU())
            blocks.append(nn.Conv1d(width, width, kernel_size, padding=padding))
        blocks.append(nn.GELU())
        blocks.append(nn.Conv1d(width, self.out_channels, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def _append_grid(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C)
        if not self.include_x:
            return x
        bsz, size, _ = x.shape
        grid = torch.arange(size, device=x.device, dtype=x.dtype) / float(size)
        grid = grid.view(1, size, 1).expand(bsz, -1, -1)
        return torch.cat([x, grid], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ResidualCNN1D expects x with shape (B,S,C), got {tuple(x.shape)}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"ResidualCNN1D input channel mismatch: expected C={self.in_channels}, got C={x.shape[-1]}"
            )

        x = self._append_grid(x)
        x = x.permute(0, 2, 1).contiguous()  # (B, C, S)
        y = self.net(x)  # (B, T, S)
        return y.permute(0, 2, 1).contiguous()  # (B, S, T)


class SpectralConv1d(nn.Module):
    """1D spectral convolution used by FNO.

    Input:  x of shape (B, C_in, S)
    Output: y of shape (B, C_out, S)
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or modes <= 0:
            raise ValueError("in_channels, out_channels, and modes must be positive.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)

        scale = 1.0 / max(1, in_channels * out_channels)
        w = scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SpectralConv1d expects x with shape (B,C,S), got {tuple(x.shape)}")
        bsz, cin, size = x.shape
        if cin != self.in_channels:
            raise ValueError(f"SpectralConv1d expected C_in={self.in_channels}, got {cin}")

        x_ft = torch.fft.rfft(x, dim=-1)
        kmax = x_ft.shape[-1]
        m = min(self.modes, kmax)

        out_ft = torch.zeros(bsz, self.out_channels, kmax, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        y = torch.fft.irfft(out_ft, n=size, dim=-1)
        return y


class PointwiseMLP1d(nn.Module):
    """Small pointwise (1x1) MLP in channel space."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int | None = None):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if hidden_channels is None:
            hidden_channels = max(in_channels, out_channels)
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")

        self.fc1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ResidualFNO1D(nn.Module):
    """1D FNO residual model.

    Input:  x of shape (B, S, C)
    Output: r of shape (B, S, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
        use_mlp_head: bool = True,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if width <= 0 or modes <= 0 or n_layers < 1:
            raise ValueError("width and modes must be positive; n_layers must be >= 1.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.modes = int(modes)
        self.n_layers = int(n_layers)

        self.fc0 = nn.Linear(self.in_channels + 1, self.width)
        self.spectral_layers = nn.ModuleList([SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.n_layers)])
        self.skip_layers = nn.ModuleList([nn.Conv1d(self.width, self.width, kernel_size=1) for _ in range(self.n_layers)])

        if use_mlp_head:
            self.head = PointwiseMLP1d(self.width, self.out_channels)
        else:
            self.head = nn.Conv1d(self.width, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ResidualFNO1D expects x with shape (B,S,C), got {tuple(x.shape)}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"ResidualFNO1D input channel mismatch: expected C={self.in_channels}, got C={x.shape[-1]}"
            )

        bsz, size, _ = x.shape
        grid = (torch.arange(size, device=x.device, dtype=x.dtype) / float(size)).view(1, size, 1)
        grid = grid.expand(bsz, -1, -1)

        x = torch.cat([x, grid], dim=-1)  # (B, S, C+1)
        x = self.fc0(x).permute(0, 2, 1).contiguous()  # (B, width, S)

        for spec, skip in zip(self.spectral_layers, self.skip_layers):
            x = F.gelu(spec(x) + skip(x))

        y = self.head(x)  # (B, T, S)
        return y.permute(0, 2, 1).contiguous()  # (B, S, T)


class ELMResidual1D:
    """Extreme Learning Machine residual model.

    fit/predict interface for closed-form ridge regression on fixed random features.

    X shape: (N, D_in)
    Y shape: (N, D_out)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2000,
        lam: float = 1e-6,
        activation: str = "tanh",
        seed: int = 0,
        standardize_x: bool = False,
    ):
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim, output_dim, and hidden_dim must be positive.")
        if lam < 0:
            raise ValueError("lam must be non-negative.")
        if activation not in {"tanh", "relu", "gelu"}:
            raise ValueError("activation must be one of {'tanh','relu','gelu'}.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.lam = float(lam)
        self.activation = activation
        self.seed = int(seed)
        self.standardize_x = bool(standardize_x)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        self.W_in = torch.randn(self.hidden_dim, self.input_dim, generator=gen, dtype=torch.float64)
        self.b = torch.randn(self.hidden_dim, generator=gen, dtype=torch.float64)

        self.W_out: torch.Tensor | None = None
        self.x_mean: torch.Tensor | None = None
        self.x_std: torch.Tensor | None = None

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "relu":
            return F.relu(z)
        return F.gelu(z)

    def _to_f64_cpu(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().to(device="cpu", dtype=torch.float64)
        return torch.as_tensor(x, dtype=torch.float64, device="cpu")

    def _transform_x(self, X: torch.Tensor, fit: bool) -> torch.Tensor:
        if X.ndim != 2:
            raise ValueError(f"ELMResidual1D expects X with shape (N,D_in), got {tuple(X.shape)}")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"ELMResidual1D expected D_in={self.input_dim}, got {X.shape[1]}")

        if not self.standardize_x:
            return X

        if fit:
            self.x_mean = X.mean(dim=0, keepdim=True)
            self.x_std = X.std(dim=0, keepdim=True)

        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("standardize_x=True but scaler parameters are not initialized. Call fit first.")

        return (X - self.x_mean) / (self.x_std + 1e-12)

    def _hidden(self, X: torch.Tensor) -> torch.Tensor:
        # H = sigma(X W_in^T + b), shape (N, M)
        return self._act(X @ self.W_in.t() + self.b.view(1, -1))

    def fit(self, X: torch.Tensor | np.ndarray, Y: torch.Tensor | np.ndarray) -> None:
        X_t = self._to_f64_cpu(X)
        Y_t = self._to_f64_cpu(Y)

        if X_t.ndim != 2:
            raise ValueError(f"ELMResidual1D expects X with shape (N,D_in), got {tuple(X_t.shape)}")
        if Y_t.ndim != 2:
            raise ValueError(f"ELMResidual1D expects Y with shape (N,D_out), got {tuple(Y_t.shape)}")
        if X_t.shape[0] != Y_t.shape[0]:
            raise ValueError(f"X and Y sample size mismatch: {X_t.shape[0]} vs {Y_t.shape[0]}")
        if Y_t.shape[1] != self.output_dim:
            raise ValueError(f"ELMResidual1D expected D_out={self.output_dim}, got {Y_t.shape[1]}")

        X_t = self._transform_x(X_t, fit=True)
        H = self._hidden(X_t)  # (N, M)

        # Closed-form ridge: W_out = (H^T H + lam I)^(-1) H^T Y
        ht_h = H.t() @ H
        if self.lam > 0.0:
            ht_h = ht_h + self.lam * torch.eye(self.hidden_dim, dtype=torch.float64)
        ht_y = H.t() @ Y_t
        self.W_out = torch.linalg.solve(ht_h, ht_y)

    def predict(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        if self.W_out is None:
            raise RuntimeError("ELMResidual1D is not fitted. Call fit before predict.")

        X_t = self._to_f64_cpu(X)
        X_t = self._transform_x(X_t, fit=False)
        H = self._hidden(X_t)
        Y_hat = H @ self.W_out
        return Y_hat.to(dtype=torch.float32)
