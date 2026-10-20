from copy import deepcopy
import pytest
from test_participation import fixture
from rl_stocks.reporting.offpolicy import saved_episode


def test_no_missing_directional_or_protection_fields_are_fabricated():
    r=fixture();del r['directional'];out=saved_episode(r)
    assert not out['directional_pnl_reconstructed'] and out['held_protection_fraction'] is None
    assert out['short_fraction']==0
    assert out['net_excess_pp']==pytest.approx(out['saved_path_noncost_excess_pp']-out['incremental_cost_drag_pp'])


@pytest.mark.parametrize('mutate',[lambda r:r.update(split='test'),lambda r:r['rows'][0].update(session='2024-01-02'),lambda r:r['metrics'].update(costs=60),lambda r:r['metrics'].update(net_return=.02),lambda r:r['rows'][0].update(exposure=float('nan')),lambda r:r['baselines']['buy_hold']['rows'].append(dict(session='2021-01-05'))])
def test_nonmatching_or_invalid_saved_evidence_rejected(mutate):
    r=deepcopy(fixture());mutate(r)
    with pytest.raises(ValueError):saved_episode(r)
