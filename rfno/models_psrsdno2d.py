from __future__ import annotations

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


def _sample_log_uniform(min_val: float, max_val: float, size: int, generator: torch.Generator) -> torch.Tensor:
    if min_val <= 0 or max_val <= 0 or min_val > max_val:
        raise ValueError("log-uniform bounds must satisfy 0 < min <= max.")
    log_min = float(np.log(min_val))
    log_max = float(np.log(max_val))
    u = torch.rand(size, generator=generator, dtype=torch.float32)
    return torch.exp(log_min + (log_max - log_min) * u)


def _sample_dictionary_parameters(
    dict_size: int,
    alpha_min: float,
    alpha_max: float,
    beta_min: float,
    beta_max: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dict_size <= 0:
        raise ValueError("dict_size must be positive.")

    alpha = _sample_log_uniform(alpha_min, alpha_max, dict_size, generator=generator)
    beta = _sample_log_uniform(beta_min, beta_max, dict_size, generator=generator)
    p = torch.randint(0, 2, (dict_size,), generator=generator, dtype=torch.int64)
    q = torch.randint(0, 2, (dict_size,), generator=generator, dtype=torch.int64)
    s = torch.randint(0, 2, (dict_size,), generator=generator, dtype=torch.int64)

    # Ensure identity exists in the dictionary.
    p[0] = 0
    q[0] = 0
    s[0] = 0
    return alpha, beta, p, q, s


def _build_psi_block(
    kx_values: torch.Tensor,
    ky_values: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    eps_norm: float,
) -> torch.Tensor:
    device = alpha.device
    dtype = torch.float32
    ctype = torch.cfloat
    d = alpha.shape[0]
    kx = kx_values.to(device=device, dtype=dtype).reshape(-1, 1)
    ky = ky_values.to(device=device, dtype=dtype).reshape(1, -1)
    kappa = kx.pow(2) + ky.pow(2)

    psi = torch.empty((d, kx.shape[0], ky.shape[1]), dtype=ctype, device=device)
    for idx in range(d):
        num = torch.ones((kx.shape[0], ky.shape[1]), dtype=ctype, device=device)
        if int(p[idx].item()) == 1:
            num = num * (1j * kx.to(ctype))
        if int(q[idx].item()) == 1:
            num = num * (1j * ky.to(ctype))

        if int(s[idx].item()) == 1:
            denom = alpha[idx] + beta[idx] * kappa
            psi_d = num / denom.to(ctype)
        else:
            psi_d = num

        rms = torch.sqrt(torch.mean(torch.abs(psi_d) ** 2) + float(eps_norm))
        psi[idx] = psi_d / rms

    return psi


class PhysicsShapedSpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        dict_size: int = 32,
        alpha_min: float = 1e-1,
        alpha_max: float = 1e1,
        beta_min: float = 1e-2,
        beta_max: float = 1e0,
        seed: int = 0,
        complex_mixing: bool = True,
        eps_norm: float = 1e-12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.dict_size = dict_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.seed = seed
        self.complex_mixing = complex_mixing
        self.eps_norm = eps_norm

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        alpha, beta, p, q, s = _sample_dictionary_parameters(
            dict_size=dict_size,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            beta_min=beta_min,
            beta_max=beta_max,
            generator=g,
        )
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.register_buffer("p", p)
        self.register_buffer("q", q)
        self.register_buffer("s", s)

        scale = 1.0 / max(1, in_channels * out_channels)
        a_real = scale * torch.randn(dict_size, in_channels, out_channels, generator=g, dtype=torch.float32)
        if complex_mixing:
            a_imag = scale * torch.randn(dict_size, in_channels, out_channels, generator=g, dtype=torch.float32)
        else:
            a_imag = torch.zeros_like(a_real)
        a_mixing = torch.complex(a_real, a_imag)
        self.register_buffer("a_mixing", a_mixing)

        kx_pos = torch.arange(modes1, dtype=torch.float32)
        kx_neg = torch.arange(-modes1, 0, dtype=torch.float32)
        ky = torch.arange(modes2, dtype=torch.float32)

        psi_pos = _build_psi_block(kx_pos, ky, self.alpha, self.beta, self.p, self.q, self.s, eps_norm=eps_norm)
        psi_neg = _build_psi_block(kx_neg, ky, self.alpha, self.beta, self.p, self.q, self.s, eps_norm=eps_norm)
        self.register_buffer("psi_pos", psi_pos)
        self.register_buffer("psi_neg", psi_neg)

        weights1 = torch.einsum("dxy,dio->ioxy", psi_pos, a_mixing)
        weights2 = torch.einsum("dxy,dio->ioxy", psi_neg, a_mixing)
        self.register_buffer("weights1", weights1)
        self.register_buffer("weights2", weights2)

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


class PSRSDNO2d(nn.Module):
    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        padding: int = 9,
        spectral_gain: float = 1.0,
        skip_gain: float = 1.0,
        activation: str = "gelu",
        psrsdno_dict_size: int = 32,
        psrsdno_alpha_min: float = 1e-1,
        psrsdno_alpha_max: float = 1e1,
        psrsdno_beta_min: float = 1e-2,
        psrsdno_beta_max: float = 1e0,
        psrsdno_seed: int = 0,
        psrsdno_complex_mixing: bool = True,
        psrsdno_eps_norm: float = 1e-12,
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

        conv_kwargs = {
            "in_channels": width,
            "out_channels": width,
            "modes1": modes1,
            "modes2": modes2,
            "dict_size": psrsdno_dict_size,
            "alpha_min": psrsdno_alpha_min,
            "alpha_max": psrsdno_alpha_max,
            "beta_min": psrsdno_beta_min,
            "beta_max": psrsdno_beta_max,
            "complex_mixing": psrsdno_complex_mixing,
            "eps_norm": psrsdno_eps_norm,
        }
        self.conv0 = PhysicsShapedSpectralConv2d(**conv_kwargs, seed=psrsdno_seed + 0)
        self.conv1 = PhysicsShapedSpectralConv2d(**conv_kwargs, seed=psrsdno_seed + 1)
        self.conv2 = PhysicsShapedSpectralConv2d(**conv_kwargs, seed=psrsdno_seed + 2)
        self.conv3 = PhysicsShapedSpectralConv2d(**conv_kwargs, seed=psrsdno_seed + 3)

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
