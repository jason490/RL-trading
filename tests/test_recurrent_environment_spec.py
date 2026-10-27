"""Fixed actual-smoke view contract on synthetic sources; no training or APIs."""
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta

import pytest

from test_core import bar
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.training import recurrent_environment_spec as spec
from rl_stocks.training.recurrent_loop import LoopConfig


def source_fixture(monkeypatch):
    protocol = spec.protocol()
    names = [f'x{i}' for i in range(7)]
    source = dict(schema='daily-indicators-v1', names=names,
                  normalizer=asdict(Normalizer(tuple(names), (0.,)*7, (1.,)*7)),
                  manifest=dict(universe=protocol['symbols'], included_splits=['train', 'validation'],
                                unverified_override=False, provenance=dict(capabilities=dict(feed='synthetic'))),
                  episodes={})
    for symbol in protocol['symbols']:
        for split, start, rows in [('train', date(2016, 1, 1), 140),
                                   ('validation', date(2021, 1, 4), 80)]:
            bars = []
            for i in range(rows):
                value = asdict(bar(str(start+timedelta(days=i))))
                value['symbol'] = symbol
                bars.append(value)
            source['episodes'][symbol+'-'+split] = dict(symbol=symbol, split=split,
                bars=bars, features=[[i/100.]*7 for i in range(rows)], atr=[2.]*rows)
    protocol['source_dataset'] = digest(source)
    protocol['normalizer_sha256'] = digest(source['normalizer'])
    monkeypatch.setattr(spec, 'protocol', lambda: deepcopy(protocol))
    return source


def test_registered_counts_and_account_limits():
    value = spec.protocol()
    loop = LoopConfig(**value['loop'])
    assert loop.maximum_updates == value['expected_updates'] == 4
    assert value['maximum_adam_steps'] == 12
    assert value['execution']['maximum_incremental_study_units'] == .1
    assert value['execution']['allocation_deadline_seconds'] == 900
    assert value['execution']['project_teardown'] == 130
    assert value['execution']['minimum_account_balance'] == 150
    assert value['execution']['account_teardown_balance'] == 170


def test_exact_source_prefix_and_frozen_normalization_without_mutation(monkeypatch):
    source = source_fixture(monkeypatch)
    before = deepcopy(source)
    view = spec.build_view(source)
    assert source == before and view['normalizer'] == source['normalizer']
    assert view['manifest']['source_manifest_sha256'] == digest(source['manifest'])
    assert len(view['episodes']) == 12
    for key, ep in view['episodes'].items():
        expected = 129 if ep['split'] == 'train' else 65
        for field in ('bars', 'features', 'atr'):
            assert ep[field] == source['episodes'][key][field][:expected]
        assert len(ep['bars']) == expected
    spec.validate_view(view)
    view['episodes']['SPY-train']['features'][0][0] = 9
    assert source['episodes']['SPY-train']['features'][0][0] == 0


@pytest.mark.parametrize('kind', ['normalizer', 'rows', 'source', 'symbols', 'test', 'future', 'alignment'])
def test_view_mutations_reject(monkeypatch, kind):
    source = source_fixture(monkeypatch)
    view = spec.build_view(source)
    if kind == 'normalizer':
        view['normalizer']['means'] = (1.,)*7
    elif kind == 'rows':
        view['episodes']['SPY-train']['bars'].pop()
    elif kind == 'source':
        view['manifest']['recurrent_environment_smoke']['source_dataset'] = '0'*64
    elif kind == 'symbols':
        view['manifest']['universe'].pop()
    elif kind == 'test':
        view['episodes']['SPY-validation']['split'] = 'test'
    elif kind == 'future':
        view['episodes']['SPY-validation']['bars'][0]['session'] = '2024-01-01'
    else:
        view['episodes']['SPY-train']['bars'][0]['symbol'] = 'QQQ'
    with pytest.raises(ValueError):
        spec.validate_view(view)


def test_changed_original_source_cannot_be_substituted(monkeypatch):
    source = source_fixture(monkeypatch)
    source['episodes']['SPY-train']['features'][0][0] = .1
    with pytest.raises(ValueError, match='source'):
        spec.build_view(source)


def test_environment_worker_rejects_locally_before_outputs(tmp_path):
    from rl_stocks.training.recurrent_environment_worker import execute_colab
    with pytest.raises(RuntimeError, match='Colab CPU'):
        execute_colab({}, tmp_path/'artifacts/recurrent-loop')
    assert not (tmp_path/'artifacts').exists()
