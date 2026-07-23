import math

import torch

from pol.paper1.model1 import decode_equispaced_point_observation_to_real_fourier
from pol.paper1.random_features import RandomFeatureMap
from pol.paper1.target_representation import real_fourier_synthesis


def test_model1_q_gt_even_j_copies_observable_and_zero_pads():
    coeff=torch.arange(15,dtype=torch.float64)[None]/10
    raw=real_fourier_synthesis(coeff,16,domain_length=1.0)
    phi=raw/math.sqrt(16)
    decoded=decode_equispaced_point_observation_to_real_fourier(phi,17,domain_length=1.0)
    torch.testing.assert_close(decoded[:,:15],coeff,atol=1e-12,rtol=1e-12)
    assert torch.equal(decoded[:,15:],torch.zeros_like(decoded[:,15:]))


def test_model1_observable_result_unchanged():
    coeff=torch.randn(2,9,dtype=torch.float64)
    raw=real_fourier_synthesis(coeff,16,domain_length=1.0)
    decoded=decode_equispaced_point_observation_to_real_fourier(raw/4,9,domain_length=1.0)
    torch.testing.assert_close(decoded,coeff,atol=1e-12,rtol=1e-12)


def test_random_features_reproducible_scaled_and_skip():
    phi=torch.randn(3,5,dtype=torch.float64)
    kwargs=dict(J=5,width=4,activation="identity",seed=7,weight_scale=.2,bias_scale=.1,dtype=phi.dtype,device="cpu")
    a=RandomFeatureMap.create(**kwargs); b=RandomFeatureMap.create(**kwargs)
    assert torch.equal(a.A,b.A) and torch.equal(a.c,b.c)
    out=a(phi)
    torch.testing.assert_close(out[:,:5],phi)
    torch.testing.assert_close(out[:,5:],(phi@a.A.T+a.c)/2)
    c=RandomFeatureMap.create(**{**kwargs,"seed":8})
    assert not torch.equal(a.A,c.A)
