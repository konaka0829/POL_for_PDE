"""Random Feature Model for 1D Burgers operator learning.

This module ports the vvRF Burgers implementation into this repository
without runtime dependency on the reference repository.

Core behavior follows AGENT.md + reference `RFM.py`:
- Random features are built in Fourier space with GRF coefficients.
- Feature map is `sig_rf * ELU(K_fine * irfft(chi * conv_ft))`.
- Training forms normal-equation terms directly via trapz integrals:
  `AstarY[j] = sum_i <phi_j(a_i), y_i>`,
  `AstarA[j,l] = sum_i <phi_j(a_i), phi_l(a_i)>`, then `AstarA /= m`.
- Solve `alpha` by `solve` (regularized) or `torch.linalg.lstsq` (unregularized).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class GaussianRFcoeff:
    """Periodic GRF coefficients in rFFT convention (positive modes only)."""

    def __init__(self, size: int, kmax: int | None = None, device: torch.device | None = None):
        if size % 2 != 0:
            raise ValueError("`size` must be even for rfft/irfft.")
        self.device = device
        self.size = size
        self.kfreq = size // 2
        if kmax is None:
            self.kmax = self.kfreq
        else:
            self.kmax = min(int(kmax), self.kfreq)

    def sample(self, n_samples: int) -> torch.Tensor:
        iid = torch.randn(n_samples, self.kmax, 2, device=self.device)
        return (iid[..., 0] - 1j * iid[..., 1]) / 2.0

    def zeropad(self, ctensor: torch.Tensor) -> torch.Tensor:
        coeff = torch.zeros(*ctensor.shape[:-1], self.kfreq + 1, dtype=torch.cfloat, device=self.device)
        coeff[..., 1 : self.kmax + 1] = ctensor
        return coeff


class RandomFeatureModel:
    """RFM for Burgers 1D using Fourier-domain random features."""

    def __init__(
        self,
        K: int,
        n: int,
        m: int,
        ntest: int,
        lamreg: float = 0.0,
        nu_rf: float = 2.5e-3,
        al_rf: float = 4.0,
        bsize_train: int | None = None,
        bsize_test: int | None = None,
        bsize_grf_train: int | None = None,
        bsize_grf_test: int | None = None,
        bsize_grf_sample: int | None = None,
        device: torch.device | None = None,
        al_g: float = 2.0,
        tau_g: float = 7.5,
        sig_g: float | None = None,
        kmax: int | None = None,
        sig_rf: float = 1.0,
        K_fine: int = 8192,
    ):
        if K % 2 != 0:
            raise ValueError("K must be even.")
        self.K = int(K)
        self.n = int(n)
        self.m = int(m)
        self.ntest = int(ntest)
        self.lamreg = float(lamreg)
        self.nu_rf = float(nu_rf)
        self.al_rf = float(al_rf)
        self.sig_rf = float(sig_rf)
        self.device = device

        self.bsize_train = 50 if bsize_train is None else int(bsize_train)
        self.bsize_test = 50 if bsize_test is None else int(bsize_test)
        self.bsize_grf_train = 20 if bsize_grf_train is None else int(bsize_grf_train)
        self.bsize_grf_test = self.m if bsize_grf_test is None else int(bsize_grf_test)
        if bsize_grf_sample is None:
            self.bsize_grf_sample = self.m if self.m <= 256 else max(1, self.m // 16)
        else:
            self.bsize_grf_sample = int(bsize_grf_sample)

        self.al_g = float(al_g)
        self.tau_g = float(tau_g)
        self.sig_g = float(self.tau_g ** (0.5 * (2.0 * self.al_g - 1.0))) if sig_g is None else float(sig_g)

        kfreq = self.K // 2
        if kmax is None:
            self.kmax = kfreq
        else:
            self.kmax = min(int(kmax), kfreq)
        self.K_fine = int(K_fine)

        self.h = 1.0 / self.K
        self.kwave = torch.arange(0, self.K // 2 + 1, device=self.device)

        self.grf = GaussianRFcoeff(self.K, kmax=self.kmax, device=self.device)
        self.grf_g = torch.zeros((self.m, self.kmax), dtype=torch.cfloat, device=self.device)
        self.resample()

        self.al_model = torch.zeros(self.m, device=self.device)
        self.AstarA = torch.zeros((self.m, self.m), device=self.device)
        self.AstarY = torch.zeros(self.m, device=self.device)

        self.input_train: torch.Tensor | None = None
        self.output_train: torch.Tensor | None = None
        self.output_train_noisy: torch.Tensor | None = None
        self.input_test: torch.Tensor | None = None
        self.output_test: torch.Tensor | None = None

    def load_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.input_train = x
        self.output_train = y
        if self.output_train_noisy is None:
            self.output_train_noisy = y

    def load_test(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.input_test = x
        self.output_test = y

    def resample(self) -> None:
        for grf_batch in torch.split(self.grf_g, self.bsize_grf_sample):
            grf_batch[...] = self.grf.sample(grf_batch.shape[0])

    @staticmethod
    def act_filter(r: torch.Tensor, al_rf: float) -> torch.Tensor:
        return F.relu(torch.minimum(2 * r, torch.pow(0.5 + r, -al_rf)))

    def rf_batch(self, a_batch: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
        pi = math.pi
        sqrt_eig = self.sig_g * (
            (4 * (pi**2) * (self.kwave[1 : self.kmax + 1] ** 2) + self.tau_g**2) ** (-self.al_g / 2.0)
        )
        wave_func = RandomFeatureModel.act_filter(torch.abs(self.nu_rf * self.kwave * 2 * pi), self.al_rf)

        a_ft = torch.fft.rfft(a_batch)
        g_scaled = math.sqrt(2.0) * sqrt_eig * g_batch
        conv = torch.einsum("nk,mk->nmk", a_ft[..., 1 : self.kmax + 1], g_scaled)

        return self.sig_rf * F.elu(self.K_fine * torch.fft.irfft(wave_func * self.grf.zeropad(conv), n=self.K))

    def fit(self, a_batch: torch.Tensor | None = None, y_batch: torch.Tensor | None = None) -> None:
        if a_batch is None and y_batch is None:
            if self.input_train is None or self.output_train_noisy is None:
                raise ValueError("Training data is not loaded.")
            input_train = self.input_train
            output_train = self.output_train_noisy
        else:
            if a_batch is None or y_batch is None:
                raise ValueError("Both a_batch and y_batch are required together.")
            input_train = a_batch
            output_train = y_batch

        loader = DataLoader(
            TensorDataset(input_train, output_train),
            batch_size=min(self.bsize_train, input_train.shape[0]),
            shuffle=True,
        )
        self.AstarY = torch.zeros(self.m, device=self.device)
        self.AstarA = torch.zeros((self.m, self.m), device=self.device)

        for a, y in loader:
            a = a.to(self.device)
            y = y.to(self.device)

            for j0 in range(0, self.m, self.bsize_grf_train):
                j1 = min(j0 + self.bsize_grf_train, self.m)
                rf_j = self.rf_batch(a, self.grf_g[j0:j1, :])
                self.AstarY[j0:j1] += torch.sum(
                    torch.trapz(torch.einsum("nmk,nk->nm", rf_j, y), dx=self.h), dim=0
                )

                for l0 in range(0, j1, self.bsize_grf_train):
                    l1 = min(l0 + self.bsize_grf_train, j1)
                    rf_l = self.rf_batch(a, self.grf_g[l0:l1, :])
                    self.AstarA[l0:l1, j0:j1] += torch.sum(
                        torch.trapz(torch.einsum("nlk,nmk->nlm", rf_l, rf_j), dx=self.h), dim=0
                    )

        self.AstarA = self.AstarA + self.AstarA.T - torch.diag(torch.diag(self.AstarA))
        self.AstarA /= self.m

        if self.lamreg > 0:
            system = self.AstarA + self.lamreg * torch.eye(self.m, device=self.device)
            self.al_model = torch.linalg.solve(system, self.AstarY)
        else:
            self.al_model = torch.linalg.lstsq(self.AstarA, self.AstarY).solution.squeeze()

    def predict(self, a: torch.Tensor) -> torch.Tensor:
        a = a.to(self.device)
        single_input = a.ndim == 1
        if single_input:
            a = a.unsqueeze(0)

        output = torch.zeros_like(a)
        for g, alpha in zip(torch.split(self.grf_g, self.bsize_grf_test), torch.split(self.al_model, self.bsize_grf_test)):
            output += torch.einsum("m,nmk->nk", alpha, self.rf_batch(a, g))
        output = output / self.m

        return output.squeeze(0) if single_input else output
