"""Numerical audit boundaries, without fitting or timing assertions."""
import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('numerical_audit',Path(__file__).resolve().parents[1]/'scripts/diagnose_regularization_numerics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_counts_keep_normal_boundary_distinct_from_subnormal_and_nonfinite():
    tiny=torch.finfo(torch.float32).tiny
    tensor=torch.tensor([0.,tiny,tiny/2,-tiny/2,float('inf'),float('nan')],dtype=torch.float32)
    result=module.tensor_counts([tensor,torch.tensor([1,2],dtype=torch.int64)])
    assert result==dict(elements=6,nonfinite=2,zero=1,subnormal=2)


def test_float64_uses_its_own_range():
    # Below float32 tiny is still normal float64, not a float64 subnormal.
    result=module.tensor_counts([torch.tensor([1e-100],dtype=torch.float64)])
    assert result['elements']==1 and result['subnormal']==0
