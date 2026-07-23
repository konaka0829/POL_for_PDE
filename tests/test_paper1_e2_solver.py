import math

import pytest
import torch

from pol.paper1.solvers import solve_reaction_diffusion_final_state


def solve(u,**kwargs):
    return solve_reaction_diffusion_final_state(
        u,nu=.1,alpha=0.,beta=0.,T=.02,dt=.001,domain_length=kwargs.pop("L",1.0),
        nonlinear_filter="two_thirds",**kwargs)


def test_rd_zero_batch_dtype_and_single_consistency():
    zero=torch.zeros(2,32,dtype=torch.float64)
    result=solve(zero)
    assert torch.equal(result.values,zero) and result.values.dtype==zero.dtype
    torch.testing.assert_close(result.values[:1],solve(zero[:1]).values)


def test_rd_pure_diffusion_mode_and_domain_length():
    L=2.0; x=torch.arange(64,dtype=torch.float64)*L/64
    u=torch.cos(2*math.pi*x/L)[None]
    result=solve(u,L=L)
    exact=u*math.exp(-.1*(2*math.pi/L)**2*.02)
    torch.testing.assert_close(result.values,exact,atol=2e-4,rtol=2e-4)


def test_rd_time_alignment_and_nonfinite_context():
    u=torch.zeros(1,16,dtype=torch.float64)
    with pytest.raises(ValueError,match="aligned"):
        solve_reaction_diffusion_final_state(u,nu=.1,alpha=1,beta=1,T=.015,dt=.01,domain_length=1)
    u[0,0]=float("nan")
    with pytest.raises(FloatingPointError,match="context=bad"):
        solve_reaction_diffusion_final_state(u,nu=.1,alpha=1,beta=1,T=.01,dt=.01,domain_length=1,context="bad")


def test_rd_dt_refinement_converges():
    torch.manual_seed(2)
    u=.1*torch.randn(1,32,dtype=torch.float64)
    coarse=solve_reaction_diffusion_final_state(u,nu=.1,alpha=1,beta=1,T=.02,dt=.002,domain_length=1).values
    fine=solve_reaction_diffusion_final_state(u,nu=.1,alpha=1,beta=1,T=.02,dt=.001,domain_length=1).values
    finer=solve_reaction_diffusion_final_state(u,nu=.1,alpha=1,beta=1,T=.02,dt=.0005,domain_length=1).values
    assert torch.linalg.vector_norm(fine-finer)<torch.linalg.vector_norm(coarse-fine)
