from copy import deepcopy

import pytest

from rl_stocks.features.fundamentals.snapshot import build_flow_snapshot, build_snapshot
from test_financial_snapshot import NS, fact, payload


def test_distinct_flow_contract_needs_no_balance_and_preserves_input():
    p = payload()
    p['facts'] = p['facts'][:3]
    before = deepcopy(p)
    result = build_flow_snapshot(p, '10-K')
    assert result['schema'] == 'filing-flow-snapshot-v1'
    assert result['names'] == ['filing_revenue_growth', 'filing_net_margin']
    assert result['values'] == pytest.approx([0.2, -0.1])
    assert result['arithmetic_ready'] and not result['financial_features_enabled']
    assert not result['publication_verified'] and not result['historical_price_link_verified']
    assert not build_snapshot(p, '10-K')['arithmetic_ready']
    assert p == before


@pytest.mark.parametrize('index', [0, 1, 2])
def test_missing_required_flow_does_not_fill_or_emit_partial_values(index):
    p = payload()
    del p['facts'][index]
    result = build_flow_snapshot(p, '10-K')
    assert not result['arithmetic_ready'] and result['values'] is None


def test_parent_income_not_aliased_to_consolidated_profit_loss():
    p = payload()
    p['facts'][1]['concept'] = NS + 'ProfitLoss'
    assert not build_flow_snapshot(p, '10-K')['arithmetic_ready']


def test_conflicting_revenue_no_shorter_interval_fallback():
    p = payload()
    p['facts'].extend([fact('Revenues', 130, '2019-02-03'),
                       fact('Revenues', 30, '2019-11-03'),
                       fact('NetIncomeLoss', 3, '2019-11-03')])
    result = build_flow_snapshot(p, '10-K')
    assert result['reason'] == 'conflicting_current_flow_values'
    assert result['values'] is None


@pytest.mark.parametrize('index', [0, 2])
def test_flow_denominator_must_be_positive(index):
    p = payload()
    p['facts'][index]['value'] = '0'
    assert build_flow_snapshot(p, '10-K')['reason'] == 'invalid_revenue_denominator'


def test_balance_not_used_but_source_coherence_still_enforced():
    p = payload()
    p['facts'][4]['value'] = '-100'
    assert build_flow_snapshot(p, '10-K')['arithmetic_ready']
    p['facts'][4]['cik'] = 'different'
    with pytest.raises(ValueError, match='mixed'):
        build_flow_snapshot(p, '10-K')


def test_legacy_and_flow_ratios_match_when_both_ready():
    p = payload()
    assert build_flow_snapshot(p, '10-K')['values'] == build_snapshot(p, '10-K')['values'][:2]
