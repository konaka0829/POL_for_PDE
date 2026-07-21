import torch
from pol.paper1.heat import heat_multiplier_vector
from pol.paper1.readouts import fit_centered_affine_ridge,l2_analysis_matrix,l2_synthesis_matrix

def test_analysis_synthesis_and_ideal_response():
    D=l2_analysis_matrix(9,32,domain_length=1.0); S=l2_synthesis_matrix(9,32,domain_length=1.0)
    assert torch.allclose(D@S,torch.eye(9,dtype=torch.float64),atol=1e-12)
    m=heat_multiplier_vector(9,target_nu=.1,target_T=.1,surrogate_nu=.05,surrogate_T=.1,domain_length=1)
    assert torch.allclose((torch.diag(m)@D)@S,torch.diag(m),atol=1e-12)
    assert torch.equal(m[1::2],m[2::2])

def test_centered_affine_ridge_bias_zero_and_positive():
    g=torch.Generator().manual_seed(3); x=torch.randn(40,5,generator=g,dtype=torch.float64); W=torch.randn(3,5,generator=g,dtype=torch.float64); b=torch.tensor([1.,-2.,.5],dtype=torch.float64); y=x@W.T+b
    exact=fit_centered_affine_ridge(x,y,0.0); assert torch.allclose(exact.W,W,atol=1e-11); assert torch.allclose(exact.b,b,atol=1e-11)
    ridge=fit_centered_affine_ridge(x,y,1e-4); assert torch.isfinite(ridge.W).all() and torch.isfinite(ridge.b).all()
