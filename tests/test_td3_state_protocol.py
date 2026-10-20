from copy import deepcopy
import pytest
from rl_stocks.core.contracts import digest
from rl_stocks.training import td3_state


def test_protocol_locks_current_parameters(monkeypatch):
    p = td3_state.protocol()
    assert p['algorithm'] == 'td3' and p['parameters']['policy_delay'] == 2
    assert p['screen_gate']['minimum_selected_and_final_stock_and_etf_excess_pp'] == -2
    actual = td3_state.resolved_parameters('td3')
    actual['policy_delay'] = 1
    monkeypatch.setattr(td3_state, 'resolved_parameters', lambda _: deepcopy(actual))
    with pytest.raises(ValueError, match='parameters changed'):
        td3_state.protocol()


@pytest.mark.parametrize('change', ['test_split', 'future_bar', 'unverified'])
def test_split_rejection_after_identity_verification(monkeypatch, change):
    data = dict(manifest=dict(included_splits=['train', 'validation'], unverified_override=False),
                episodes={'x': dict(split='train', bars=[dict(session='2020-12-30')])})
    if change == 'test_split':
        data['episodes']['x']['split'] = 'test'
    elif change == 'future_bar':
        data['episodes']['x']['bars'][0]['session'] = '2024-01-02'
    else:
        data['manifest']['unverified_override'] = True
    monkeypatch.setattr(td3_state, 'protocol', lambda: dict(dataset=digest(data)))
    with pytest.raises(ValueError, match='contract mismatch'):
        td3_state.validate_payload(data)


def test_dataset_identity_rejection():
    with pytest.raises(ValueError, match='identity mismatch'):
        td3_state.validate_payload({})
