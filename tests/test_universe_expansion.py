from dataclasses import asdict
from datetime import date

import pytest

from rl_stocks.core.contracts import Request
from rl_stocks.data.generation.datasets import generate, read_dataset, Normalizer
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.storage.immutable import put
from rl_stocks.data.universe.config import read_universe, symbols, SECTORS


def test_expansion_manifest_and_common_panel(tmp_path):
    expanded = read_universe('configs/universes/sector_expansion.json')
    base = read_universe('configs/universes/sector_research.json')
    assert set(symbols(base)) <= set(symbols(expanded))
    assert len(symbols(expanded, 'tradable')) == 59
    assert all(sum(i['kind'] == 'stock' and i['sector'] == sector
                   for i in expanded['instruments']) == 5 for sector in SECTORS)
    universe = {'instruments': [{'symbol': name, 'kind': 'stock', 'role': 'tradable'}
                                for name in ('SPY', 'QQQ')]}
    request = Request(('SPY', 'QQQ'), date(2020, 1, 1), date(2023, 7, 1))
    bars = SyntheticProvider().bars(request)
    snapshot = put(tmp_path / 'raw', bars, {'dividends_verified': False,
        'coverage': {s: {'missing': [], 'calendar_verified': False} for s in request.symbols}})
    options = dict(allow_unverified=True, evaluation_symbols=['SPY'], include_test=False)
    control = {'instruments': universe['instruments'][:1]}
    a = read_dataset(generate(snapshot, tmp_path / 'data', universe=control, **options))
    b = read_dataset(generate(snapshot, tmp_path / 'data', universe=universe, **options))
    validation = lambda data: {k: v for k, v in data['episodes'].items() if v['split'] == 'validation'}
    assert validation(a) == validation(b)
    assert {e['symbol'] for e in b['episodes'].values() if e['split'] == 'train'} == {'SPY', 'QQQ'}
    assert all(e['split'] != 'test' for e in b['episodes'].values())
    assert b['manifest']['included_splits'] == ['train', 'validation']
    train_rows = [row for e in b['episodes'].values() if e['split'] == 'train' for row in e['features']]
    assert asdict(Normalizer.fit(b['names'], train_rows)) == {
        key: tuple(value) for key, value in b['normalizer'].items()}
    for invalid in ([], ['MISSING'], ['SPY', 'SPY']):
        with pytest.raises(ValueError, match='evaluation symbols'):
            generate(snapshot, tmp_path / 'data', universe=universe, allow_unverified=True,
                     evaluation_symbols=invalid, include_test=False)
    with pytest.raises(ValueError, match='required symbols'):
        generate(snapshot, tmp_path / 'data', universe=universe, allow_unverified=True)


def test_report_does_not_confuse_best_checkpoint_with_stable_improvement(monkeypatch):
    import copy
    from rl_stocks.reporting.universe import summarize_universe
    protocol = {'arms': ['control', 'expanded'], 'seeds': [11, 23, 37], 'algorithm': 'ppo',
        'device': 'cpu', 'steps': 50000, 'gate': {'minimum_mean_selected_gain_pp': 2,
            'minimum_mean_final_gain_pp': 2, 'minimum_paired_final_positive_seeds': 2,
            'maximum_selected_drawdown_worsening_pp': 2, 'maximum_final_drawdown_worsening_pp': 2}}
    monkeypatch.setattr('rl_stocks.reporting.universe.verify_panels', lambda _: {
        arm: {'episodes': {'SPY-validation-0': {'split': 'validation'}}} for arm in protocol['arms']})
    data = {'datasets': {arm: {'id': arm} for arm in protocol['arms']}}
    runs = []
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            best, final = (0.2, 0.18) if arm == 'control' else (0.5, -0.1)
            points = [{'steps': step, 'phase': 'post_update', 'score': best if step == 10000 else final,
                       'mean_max_drawdown_pct': -10.0} for step in range(10000, 50001, 10000)]
            runs.append({'dataset': arm, 'seed': seed, 'algorithm': 'ppo', 'status': 'complete',
                'device': 'cpu', 'actual_steps': 50000, 'requested_steps': 50000, 'smoke': False,
                'source_hash': 'source', 'git': {'dirty': ''}, 'reward_config': {'kind': 'net_log_equity', 'scale': 1.0},
                'simulation': {}, 'versions': {}, 'selection_metric': 'mean', 'resolved_parameters': {},
                'selected_metrics': {'SPY-validation-0': {'policy': {'net_return': best, 'max_drawdown': -0.1},
                    'baselines': {'buy_hold': {'net_return': 0.3}}}}, 'validation_checkpoints': points,
                'selected_score': best, 'selected_step': 10000, 'bundle_id': arm + str(seed)})
    result = summarize_universe(runs, protocol, data, {'source': 'same-learning-source'})
    assert result['complete'] and not result['comparison']['passed']
    assert result['comparison']['mean_selected_gain_pp'] == pytest.approx(30)
    assert result['comparison']['mean_final_gain_pp'] == pytest.approx(-28)
    with pytest.raises(ValueError, match='duplicate'):
        summarize_universe(runs + runs[:1], protocol, data, {'source': 'same-learning-source'})
    changed = copy.deepcopy(runs)
    changed[-1]['selected_metrics']['SPY-validation-0']['baselines']['buy_hold']['net_return'] = 0.4
    with pytest.raises(ValueError, match='baselines'):
        summarize_universe(changed, protocol, data, {'source': 'same-learning-source'})
