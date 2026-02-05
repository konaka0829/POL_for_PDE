"""Sanity check for DarcyRFFeatures feature chunking."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rfm_features import DarcyRFFeatures, grf_sample_2d  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch_size = 2
    s = 9
    m = 6

    a_batch = torch.rand((batch_size, s, s), device=device)
    theta1 = grf_sample_2d(
        m,
        s,
        tau=1.2,
        alpha=2.0,
        device=device,
        generator=torch.Generator(device=device).manual_seed(1),
    )
    theta2 = grf_sample_2d(
        m,
        s,
        tau=1.2,
        alpha=2.0,
        device=device,
        generator=torch.Generator(device=device).manual_seed(2),
    )

    base_kwargs = dict(
        s_plus=1 / 12,
        s_minus=-1 / 3,
        delta_sig=0.15,
        eta=1e-4,
        dt=0.03,
        heat_steps=2,
        f_const=1.0,
    )

    features_full = DarcyRFFeatures(
        theta1=theta1,
        theta2=theta2,
        feature_chunk_size=m,
        **base_kwargs,
    )
    features_chunked = DarcyRFFeatures(
        theta1=theta1,
        theta2=theta2,
        feature_chunk_size=2,
        **base_kwargs,
    )

    out_full = features_full(a_batch)
    out_chunked = features_chunked(a_batch)

    torch.testing.assert_close(out_full, out_chunked, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    main()
