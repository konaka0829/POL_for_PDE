from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class EncoderOutputs:
    x_raw: torch.Tensor
    x_pre_list: List[torch.Tensor]
    z0_list: List[torch.Tensor]


class FixedEncoder1D:
    def __init__(
        self,
        *,
        s: int,
        device: torch.device,
        dtype: torch.dtype,
        args: argparse.Namespace,
    ):
        if s <= 1:
            raise ValueError("s must be >= 2")
        self.s = int(s)
        self.device = device
        self.dtype = dtype
        self.args = args
        self.complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        self.encoder = getattr(args, "encoder", "linear")
        self.encoder_center = bool(getattr(args, "encoder_center", 0))
        self.encoder_standardize = bool(getattr(args, "encoder_standardize", 0))
        self.encoder_standardize_eps = float(getattr(args, "encoder_standardize_eps", 1e-6))

        self.input_scale = float(getattr(args, "input_scale", 1.0))
        self.input_shift = float(getattr(args, "input_shift", 0.0))

        self.encoder_post = getattr(args, "encoder_post", "none")
        self.encoder_tanh_gamma = float(getattr(args, "encoder_tanh_gamma", 1.0))
        self.encoder_clip_c = float(getattr(args, "encoder_clip_c", 3.0))

        self.encoder_fourier_mode = getattr(args, "encoder_fourier_mode", "lowpass")
        self.encoder_fourier_kmin = int(getattr(args, "encoder_fourier_kmin", 0))
        self.encoder_fourier_kmax = int(getattr(args, "encoder_fourier_kmax", 16))
        self.encoder_fourier_seed = int(getattr(args, "encoder_fourier_seed", 0))
        self.encoder_fourier_amp_std = float(getattr(args, "encoder_fourier_amp_std", 0.5))
        self.encoder_fourier_output_scale = float(getattr(args, "encoder_fourier_output_scale", 1.0))

        self.encoder_randconv_kernel_size = int(getattr(args, "encoder_randconv_kernel_size", 33))
        self.encoder_randconv_seed = int(getattr(args, "encoder_randconv_seed", 0))
        self.encoder_randconv_std = float(getattr(args, "encoder_randconv_std", 1.0))
        self.encoder_randconv_normalize = getattr(args, "encoder_randconv_normalize", "l2")

        self.encoder_fourier_rfm_C = int(getattr(args, "encoder_fourier_rfm_C", 1))
        self.encoder_fourier_rfm_mode = getattr(args, "encoder_fourier_rfm_mode", "sum")
        self.encoder_fourier_rfm_activation = getattr(args, "encoder_fourier_rfm_activation", "tanh")
        self.encoder_fourier_rfm_kmin = int(getattr(args, "encoder_fourier_rfm_kmin", 0))
        self.encoder_fourier_rfm_kmax = int(getattr(args, "encoder_fourier_rfm_kmax", 16))
        self.encoder_fourier_rfm_seed = int(getattr(args, "encoder_fourier_rfm_seed", 0))
        self.encoder_fourier_rfm_theta_scale = float(getattr(args, "encoder_fourier_rfm_theta_scale", 1.0))
        self.encoder_fourier_rfm_output_scale = float(
            getattr(args, "encoder_fourier_rfm_output_scale", 1.0)
        )

        self.encoder_poly_a1 = float(getattr(args, "encoder_poly_a1", 1.0))
        self.encoder_poly_a2 = float(getattr(args, "encoder_poly_a2", 0.0))
        self.encoder_poly_a3 = float(getattr(args, "encoder_poly_a3", 0.0))

        if self.encoder_standardize_eps <= 0.0:
            raise ValueError("encoder_standardize_eps must be positive")
        if self.encoder_clip_c <= 0.0:
            raise ValueError("encoder_clip_c must be positive")
        if self.encoder_fourier_rfm_C <= 0:
            raise ValueError("encoder_fourier_rfm_C must be positive")
        if self.encoder_randconv_kernel_size <= 0:
            raise ValueError("encoder_randconv_kernel_size must be positive")

        if self.encoder == "fourier_filter":
            self._check_k_range(self.encoder_fourier_kmin, self.encoder_fourier_kmax, "encoder-fourier")
        if self.encoder == "fourier_rfm":
            self._check_k_range(
                self.encoder_fourier_rfm_kmin,
                self.encoder_fourier_rfm_kmax,
                "encoder-fourier-rfm",
            )
        if self.encoder == "randconv" and self.encoder_randconv_kernel_size > self.s:
            raise ValueError("encoder_randconv_kernel_size must be in [1, s]")

        self.idx = torch.arange(self.s // 2 + 1, dtype=torch.long, device=self.device)
        dx = 1.0 / float(self.s)
        self.k_phys = 2.0 * torch.pi * torch.fft.rfftfreq(self.s, d=dx, device=self.device).to(
            dtype=self.dtype
        )

        self.fourier_g: Optional[torch.Tensor] = (
            self._init_fourier_filter() if self.encoder == "fourier_filter" else None
        )
        self.randconv_w_hat: Optional[torch.Tensor] = (
            self._init_randconv() if self.encoder == "randconv" else None
        )
        if self.encoder == "fourier_rfm":
            self.fourier_rfm_theta_hat, self.fourier_rfm_chi = self._init_fourier_rfm()
        else:
            self.fourier_rfm_theta_hat, self.fourier_rfm_chi = None, None

    def _check_k_range(self, kmin: int, kmax: int, name: str) -> None:
        if not (0 <= kmin <= kmax <= self.s // 2):
            raise ValueError(
                f"{name} k-range must satisfy 0 <= kmin <= kmax <= s//2 (got {kmin}, {kmax})"
            )

    def _init_fourier_filter(self) -> torch.Tensor:
        mask = ((self.idx >= self.encoder_fourier_kmin) & (self.idx <= self.encoder_fourier_kmax)).to(
            dtype=self.dtype
        )
        mask_c = mask.to(self.complex_dtype)
        mode = self.encoder_fourier_mode
        if mode == "lowpass":
            mask = (self.idx <= self.encoder_fourier_kmax).to(dtype=self.dtype)
            return mask.to(self.complex_dtype)
        if mode == "bandpass":
            return mask_c

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.encoder_fourier_seed)
        nfreq = self.s // 2 + 1

        if mode == "randphase":
            phase = 2.0 * math.pi * torch.rand((nfreq,), generator=gen, dtype=torch.float32)
            phase[0] = 0.0
            if self.s % 2 == 0:
                phase[-1] = 0.0
            g = torch.polar(torch.ones_like(phase), phase).to(self.complex_dtype)
            return mask_c * g.to(self.device)

        if mode == "randamp":
            amp = 1.0 + self.encoder_fourier_amp_std * torch.randn(
                (nfreq,), generator=gen, dtype=torch.float32
            )
            amp[0] = amp[0].abs()
            if self.s % 2 == 0:
                amp[-1] = amp[-1].abs()
            return mask_c * amp.to(self.device, dtype=self.dtype).to(self.complex_dtype)

        if mode == "randcomplex":
            phase = 2.0 * math.pi * torch.rand((nfreq,), generator=gen, dtype=torch.float32)
            phase[0] = 0.0
            if self.s % 2 == 0:
                phase[-1] = 0.0
            amp = self.encoder_fourier_amp_std * torch.abs(
                torch.randn((nfreq,), generator=gen, dtype=torch.float32)
            )
            g = amp * torch.polar(torch.ones_like(phase), phase)
            return mask_c * g.to(self.device, dtype=self.complex_dtype)

        raise ValueError(f"Unsupported encoder_fourier_mode: {mode}")

    def _init_randconv(self) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.encoder_randconv_seed)

        L = self.encoder_randconv_kernel_size
        eps = 1e-12
        w_small = self.encoder_randconv_std * torch.randn((L,), generator=gen, dtype=torch.float32)
        if self.encoder_randconv_normalize == "l1":
            w_small = w_small / (w_small.abs().sum() + eps)
        elif self.encoder_randconv_normalize == "l2":
            w_small = w_small / (torch.sqrt((w_small.pow(2)).sum()) + eps)
        elif self.encoder_randconv_normalize != "none":
            raise ValueError(
                f"Unsupported encoder_randconv_normalize: {self.encoder_randconv_normalize}"
            )

        w_padded = torch.zeros((self.s,), dtype=torch.float32)
        w_padded[:L] = w_small
        w_padded = torch.roll(w_padded, shifts=-(L // 2), dims=0)
        w_hat = torch.fft.rfft(w_padded.to(self.device, dtype=self.dtype), dim=-1)
        return w_hat

    def _init_fourier_rfm(self) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.encoder_fourier_rfm_seed)

        theta = self.encoder_fourier_rfm_theta_scale * torch.randn(
            (self.encoder_fourier_rfm_C, self.s), generator=gen, dtype=torch.float32
        )
        theta = theta.to(self.device, dtype=self.dtype)
        theta_hat = torch.fft.rfft(theta, dim=-1)

        chi = (
            (self.idx >= self.encoder_fourier_rfm_kmin)
            & (self.idx <= self.encoder_fourier_rfm_kmax)
        ).to(dtype=self.dtype)
        return theta_hat, chi.to(self.complex_dtype)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        act = self.encoder_fourier_rfm_activation
        if act == "tanh":
            return torch.tanh(x)
        if act == "relu":
            return torch.relu(x)
        if act == "identity":
            return x
        raise ValueError(f"Unsupported encoder_fourier_rfm_activation: {act}")

    def _post(self, x_aff: torch.Tensor) -> torch.Tensor:
        if self.encoder_post == "none":
            return x_aff
        if self.encoder_post == "tanh":
            return torch.tanh(self.encoder_tanh_gamma * x_aff)
        if self.encoder_post == "clip":
            c = self.encoder_clip_c
            return torch.clamp(x_aff, -c, c)
        raise ValueError(f"Unsupported encoder_post: {self.encoder_post}")

    @torch.no_grad()
    def encode(self, x_batch: torch.Tensor) -> EncoderOutputs:
        x_raw = x_batch.to(device=self.device, dtype=self.dtype)
        if x_raw.ndim != 2 or x_raw.shape[-1] != self.s:
            raise ValueError(f"x_batch must have shape (B, {self.s}), got {tuple(x_raw.shape)}")

        x = x_raw
        if self.encoder_center:
            x = x - x.mean(dim=-1, keepdim=True)
        if self.encoder_standardize:
            mu = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mu) / (std + self.encoder_standardize_eps)

        x_pre_list: List[torch.Tensor]
        if self.encoder == "linear":
            x_pre_list = [x]
        elif self.encoder == "fourier_filter":
            if self.fourier_g is None:
                raise RuntimeError("fourier filter coefficients are not initialized")
            x_hat = torch.fft.rfft(x, dim=-1)
            y = torch.fft.irfft(x_hat * self.fourier_g, n=self.s, dim=-1)
            y = self.encoder_fourier_output_scale * y
            x_pre_list = [y]
        elif self.encoder == "randconv":
            if self.randconv_w_hat is None:
                raise RuntimeError("randconv kernel is not initialized")
            x_hat = torch.fft.rfft(x, dim=-1)
            y = torch.fft.irfft(x_hat * self.randconv_w_hat, n=self.s, dim=-1)
            x_pre_list = [y]
        elif self.encoder == "fourier_rfm":
            if self.fourier_rfm_theta_hat is None or self.fourier_rfm_chi is None:
                raise RuntimeError("fourier_rfm parameters are not initialized")
            x_hat = torch.fft.rfft(x, dim=-1)
            conv_acts: List[torch.Tensor] = []
            for c in range(self.encoder_fourier_rfm_C):
                conv = torch.fft.irfft(
                    self.fourier_rfm_chi * x_hat * self.fourier_rfm_theta_hat[c],
                    n=self.s,
                    dim=-1,
                )
                conv_acts.append(self._apply_activation(conv))

            if self.encoder_fourier_rfm_mode == "ensemble":
                x_pre_list = [self.encoder_fourier_rfm_output_scale * t for t in conv_acts]
            elif self.encoder_fourier_rfm_mode == "sum":
                scale = 1.0 / math.sqrt(float(self.encoder_fourier_rfm_C))
                y = scale * torch.stack(conv_acts, dim=0).sum(dim=0)
                x_pre_list = [self.encoder_fourier_rfm_output_scale * y]
            elif self.encoder_fourier_rfm_mode == "mean":
                y = torch.stack(conv_acts, dim=0).mean(dim=0)
                x_pre_list = [self.encoder_fourier_rfm_output_scale * y]
            else:
                raise ValueError(
                    f"Unsupported encoder_fourier_rfm_mode: {self.encoder_fourier_rfm_mode}"
                )
        elif self.encoder == "poly_deriv":
            x_hat = torch.fft.rfft(x, dim=-1)
            x_x = torch.fft.irfft((1j * self.k_phys) * x_hat, n=self.s, dim=-1)
            y = self.encoder_poly_a1 * x + self.encoder_poly_a2 * x.pow(2) + self.encoder_poly_a3 * x_x
            x_pre_list = [y]
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder}")

        z0_list: List[torch.Tensor] = []
        for x_pre in x_pre_list:
            x_aff = self.input_scale * x_pre + self.input_shift
            z0_list.append(self._post(x_aff))

        return EncoderOutputs(x_raw=x_raw, x_pre_list=x_pre_list, z0_list=z0_list)
