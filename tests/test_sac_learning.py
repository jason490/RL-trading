import numpy as np
import pytest
from rl_stocks.reporting.sac_learning import moments,target_parts


def test_timeout_bootstraps_true_terminal_does_not_and_entropy_sign_matches_sac():
    p=target_parts([1,1,1],[10,10,10],[-2,-2,-2],[0,1,1],[0,0,1],.1,.9)
    np.testing.assert_allclose(p['discounted_target_q'],[9,0,9])
    np.testing.assert_allclose(p['discounted_entropy'],[.18,0,.18])
    np.testing.assert_allclose(p['target'],[10.18,1,10.18])


def test_summary_uses_only_nonempty_finite_vectors():
    assert moments([-1,0,1])['mean_abs']==pytest.approx(2/3)
    for values in [[],[float('nan')],[[1]]]:
        with pytest.raises(ValueError):moments(values)


@pytest.mark.parametrize('args',[([1],[1],[-1],[0],[1],.1,.9),([1],[1],[-1],[0],[0],0,.9),([1],[float('inf')],[-1],[0],[0],.1,.9),([1],[1,2],[-1],[0],[0],.1,.9)])
def test_malformed_target_parts_rejected(args):
    with pytest.raises(ValueError):target_parts(*args)
