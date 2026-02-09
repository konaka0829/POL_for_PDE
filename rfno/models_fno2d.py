import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation_fn(name: str):
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


def _complex_init(shape: tuple[int, ...], spectral_init: str) -> torch.Tensor:
    if spectral_init == "uniform":
        # Keep compatibility: PyTorch complex rand with real/imag in [0, 1).
        return torch.rand(*shape, dtype=torch.cfloat)
    if spectral_init == "uniform_sym":
        real = 2.0 * torch.rand(*shape) - 1.0
        imag = 2.0 * torch.rand(*shape) - 1.0
        return torch.complex(real, imag)
    if spectral_init == "normal":
        real = torch.randn(*shape)
        imag = torch.randn(*shape)
        return torch.complex(real, imag)
    raise ValueError(f"Unsupported spectral_init: {spectral_init}")


class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        spectral_init: str = "uniform",
        spectral_init_scale: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.spectral_init = spectral_init
        self.spectral_init_scale = spectral_init_scale

        base_scale = 1 / (in_channels * out_channels)
        scale = base_scale * spectral_init_scale
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(
            scale * _complex_init(shape, spectral_init)
        )
        self.weights2 = nn.Parameter(
            scale * _complex_init(shape, spectral_init)
        )

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, activation: str = "gelu"):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.act = _activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    """FNO2d compatible with fourier_2d.py and with feature extraction for RFNO."""

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        padding: int = 9,
        spectral_gain: float = 1.0,
        skip_gain: float = 1.0,
        spectral_init: str = "uniform",
        spectral_init_scale: float = 1.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.spectral_gain = spectral_gain
        self.skip_gain = skip_gain
        self.activation = activation
        self.act = _activation_fn(activation)

        self.p = nn.Linear(3, width)

        self.conv0 = SpectralConv2d(
            width, width, modes1, modes2, spectral_init=spectral_init, spectral_init_scale=spectral_init_scale
        )
        self.conv1 = SpectralConv2d(
            width, width, modes1, modes2, spectral_init=spectral_init, spectral_init_scale=spectral_init_scale
        )
        self.conv2 = SpectralConv2d(
            width, width, modes1, modes2, spectral_init=spectral_init, spectral_init_scale=spectral_init_scale
        )
        self.conv3 = SpectralConv2d(
            width, width, modes1, modes2, spectral_init=spectral_init, spectral_init_scale=spectral_init_scale
        )

        self.mlp0 = MLP(width, width, width, activation=activation)
        self.mlp1 = MLP(width, width, width, activation=activation)
        self.mlp2 = MLP(width, width, width, activation=activation)
        self.mlp3 = MLP(width, width, width, activation=activation)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.q = MLP(width, 1, width * 4, activation=activation)

    def get_grid(self, shape, device: torch.device) -> torch.Tensor:
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = self.act(self.spectral_gain * x1 + self.skip_gain * x2)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = self.act(self.spectral_gain * x1 + self.skip_gain * x2)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = self.act(self.spectral_gain * x1 + self.skip_gain * x2)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        return self.spectral_gain * x1 + self.skip_gain * x2

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        x = self._backbone(x)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.q(x)
        return x.permute(0, 2, 3, 1)

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if not name.startswith("q."):
                param.requires_grad = False
