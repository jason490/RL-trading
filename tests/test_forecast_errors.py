import numpy as np
import pytest
from rl_stocks.reporting.forecast_errors import decompose


def test_unequal_panels_are_row_weighted_and_signs_are_reporting_only():
    r=decompose([-2,2,3],[-1,-1,1],['a','a','b'])
    assert r['mse']==pytest.approx(14/3)
    assert r['daily_common_error_mse']==pytest.approx(2)
    assert r['daily_within_error_mse']==pytest.approx(8/3)
    assert r['negative_sign_precision']==.5
    assert r['negative_sign_recall']==1
    assert r['prediction_target_covariance']==pytest.approx(4/3)


def test_constant_forecast_has_no_undefined_sign_metric_fabrication():
    r=decompose([1,2],[0,0],['a','a'])
    assert r['negative_sign_precision'] is None and r['negative_sign_recall'] is None
    assert r['prediction_variance']==0 and r['prediction_target_covariance']==0


@pytest.mark.parametrize('y,p,d',[([],[],[]),([1],[float('nan')],['a']),([1],[1],[]),([1],[1],[None])])
def test_invalid_vectors_rejected(y,p,d):
    with pytest.raises(ValueError):decompose(y,p,d)
