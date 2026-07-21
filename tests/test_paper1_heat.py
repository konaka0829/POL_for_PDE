import math
import pytest
import torch
from pol.paper1.grids import periodic_grid
from pol.paper1.heat import heat_regime, solve_heat_exact

@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_heat_constant_cos_sin_batch_and_domain(dtype):
    L=2.0; n=64; nu=.2; T=.3; x=periodic_grid(n,L,dtype=dtype)
    fields=torch.stack((torch.ones_like(x),torch.cos(2*torch.pi*3*x/L),torch.sin(2*torch.pi*3*x/L)))
    got=solve_heat_exact(fields,nu=nu,T=T,domain_length=L)
    decay=math.exp(-nu*T*(2*math.pi*3/L)**2); expected=fields.clone(); expected[1:]*=decay
    assert got.dtype==dtype and got.shape==fields.shape
    assert torch.allclose(got,expected,atol=2e-5 if dtype==torch.float32 else 1e-12,rtol=2e-5 if dtype==torch.float32 else 1e-12)

def test_heat_regimes_and_invalid():
    assert heat_regime(target_nu=.1,target_T=1,surrogate_nu=.05,surrogate_T=1)[0]=="stable"
    assert heat_regime(target_nu=.1,target_T=1,surrogate_nu=.2,surrogate_T=1)[0]=="unstable"
    with pytest.raises(ValueError,match="exact-match"): heat_regime(target_nu=.1,target_T=1,surrogate_nu=.1,surrogate_T=1)
    with pytest.raises(ValueError): solve_heat_exact(torch.ones(8),nu=-1,T=1,domain_length=1)
