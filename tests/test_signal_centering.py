import numpy as np
import pytest
from rl_stocks.inference.signal_centering import center_forecasts


def inputs():
    return dict(prediction=[.1,.3,.7,.9],ridge=[.01,.03,.05,.07],
        sessions=['2018-01-02']*2+['2018-01-03']*2,symbols=['A','B']*2,
        available=['2018-01-02T21:15:00+00:00']*2+['2018-01-03T21:15:00+00:00']*2,
        entry_sessions=['2018-01-03']*2+['2018-01-04']*2,expected_symbols=['A','B'])


def test_fixed_mean_anchor_preserves_ranks_differences_and_inputs():
    x=inputs();before=list(x['prediction']);out,proof=center_forecasts(**x)
    np.testing.assert_allclose(out,[-.08,.12,-.04,.16],rtol=0,atol=1e-15)
    assert out[1]-out[0]==pytest.approx(.2) and out[3]-out[2]==pytest.approx(.2)
    assert x['prediction']==before and len(proof)==2
    # Changing another session's forecasts cannot affect the first session.
    x['prediction'][2:]=[-50,100];changed,_=center_forecasts(**x)
    np.testing.assert_array_equal(changed[:2],out[:2])


def test_missing_duplicate_nonfinite_late_and_non_utc_panels_fail_closed():
    for key,value in [('symbols',['A']*4),('prediction',[float('nan')]*4),
        ('available',['2018-01-03T00:00:00+00:00']*4),('entry_sessions',['2018-01-02']*4),
        ('available',['2018-01-02T21:15:00-05:00']*2+['2018-01-03T21:15:00+00:00']*2),
        ('expected_symbols',['A']),('ridge',[1.])]:
        x=inputs();x[key]=value
        with pytest.raises(ValueError):center_forecasts(**x)
