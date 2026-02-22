import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1)

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n = x.shape
        x_ft = torch.fft.rfft(x)
        n_modes = min(self.modes1, x_ft.shape[-1])

        out_ft = torch.zeros(b, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :n_modes] = self.compl_mul1d(x_ft[:, :, :n_modes], self.weights1[:, :, :n_modes])

        return torch.fft.irfft(out_ft, n=n)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n1, n2 = x.shape
        x_ft = torch.fft.rfft2(x)

        m1 = min(self.modes1, n1)
        m2 = min(self.modes2, x_ft.shape[-1])

        out_ft = torch.zeros(
            b,
            self.out_channels,
            n1,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        return torch.fft.irfft2(out_ft, s=(n1, n2))


class MLP1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO1dGeneric(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        modes: int,
        width: int,
        use_grid: bool = True,
    ) -> None:
        super().__init__()
        self.use_grid = bool(use_grid)
        self.width = int(width)

        in_features = int(in_dim) + (1 if self.use_grid else 0)
        self.p = nn.Linear(in_features, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, modes)
        self.conv1 = SpectralConv1d(self.width, self.width, modes)
        self.conv2 = SpectralConv1d(self.width, self.width, modes)
        self.conv3 = SpectralConv1d(self.width, self.width, modes)

        self.mlp0 = MLP1d(self.width, self.width, self.width)
        self.mlp1 = MLP1d(self.width, self.width, self.width)
        self.mlp2 = MLP1d(self.width, self.width, self.width)
        self.mlp3 = MLP1d(self.width, self.width, self.width)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.q = MLP1d(self.width, int(out_dim), self.width * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)

        x = self.p(x)
        x = x.permute(0, 2, 1)

        x = F.gelu(self.mlp0(self.conv0(x)) + self.w0(x))
        x = F.gelu(self.mlp1(self.conv1(x)) + self.w1(x))
        x = F.gelu(self.mlp2(self.conv2(x)) + self.w2(x))
        x = self.mlp3(self.conv3(x)) + self.w3(x)

        x = self.q(x)
        return x.permute(0, 2, 1)

    def get_grid(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        b, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32, device=device)
        return gridx.reshape(1, size_x, 1).repeat(b, 1, 1)


class FNO2dGeneric(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        modes1: int,
        modes2: int,
        width: int,
        use_grid: bool = True,
        use_instance_norm: bool = True,
    ) -> None:
        super().__init__()
        self.use_grid = bool(use_grid)
        self.use_instance_norm = bool(use_instance_norm)
        self.width = int(width)

        in_features = int(in_dim) + (2 if self.use_grid else 0)
        self.p = nn.Linear(in_features, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)

        self.mlp0 = MLP2d(self.width, self.width, self.width)
        self.mlp1 = MLP2d(self.width, self.width, self.width)
        self.mlp2 = MLP2d(self.width, self.width, self.width)
        self.mlp3 = MLP2d(self.width, self.width, self.width)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        if self.use_instance_norm:
            self.norm0 = nn.InstanceNorm2d(self.width)
            self.norm1 = nn.InstanceNorm2d(self.width)
            self.norm2 = nn.InstanceNorm2d(self.width)
            self.norm3 = nn.InstanceNorm2d(self.width)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()

        self.q = MLP2d(self.width, int(out_dim), self.width * 4)

    def _block(self, x: torch.Tensor, conv: nn.Module, mlp: nn.Module, w: nn.Module, norm: nn.Module) -> torch.Tensor:
        x1 = norm(conv(norm(x)))
        x1 = mlp(x1)
        x2 = w(x)
        return x1 + x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)

        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x = F.gelu(self._block(x, self.conv0, self.mlp0, self.w0, self.norm0))
        x = F.gelu(self._block(x, self.conv1, self.mlp1, self.w1, self.norm1))
        x = F.gelu(self._block(x, self.conv2, self.mlp2, self.w2, self.norm2))
        x = self._block(x, self.conv3, self.mlp3, self.w3, self.norm3)

        x = self.q(x)
        return x.permute(0, 2, 3, 1)

    def get_grid(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        b, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(b, 1, size_y, 1)

        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(b, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)
