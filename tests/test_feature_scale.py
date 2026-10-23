from copy import deepcopy
import numpy as np
import pytest
from rl_stocks.reporting.feature_scale import summarize_scale
from rl_stocks.data.generation.datasets import normalizer_from


N=dict(names=['x'],means=[10.],scales=[2.])


def test_exact_frozen_transform_statistics_and_fixed_strict_thresholds():
    rows=[[0.],[4.],[10.],[16.],[20.],[32.]];before=deepcopy(rows);normal=deepcopy(N)
    result=summarize_scale(rows,['x'],N)['x'];z=np.array([normalizer_from(N).transform(r,['x'])[0] for r in rows])
    assert result['mean']==pytest.approx(z.mean()) and result['std']==pytest.approx(z.std())
    assert result['fraction_abs_above_3']==3/6 and result['fraction_abs_above_5']==1/6
    assert result['fraction_abs_above_10']==1/6 and result['maximum_abs']==11
    assert rows==before and N==normal


@pytest.mark.parametrize('rows,names,norm',[
    ([],['x'],N),([[float('nan')]],['x'],N),([[1,2]],['x'],N),
    ([[1]],['other'],N),([[1]],['x'],dict(names=['x'],means=[0.],scales=[0.])),
    ([[float('inf')]],['x'],N)])
def test_invalid_sources_or_schema_reject(rows,names,norm):
    with pytest.raises(ValueError):summarize_scale(rows,names,norm)
