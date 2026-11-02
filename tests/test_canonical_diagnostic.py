from copy import deepcopy
from pathlib import Path

import pytest

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.canonical_diagnostic import inference_only, protocol, validate_window


def test_protocol_requires_fixed_unmodified_declaration(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = protocol(root)
    assert spec['source_steps'] == 50000 and spec['predictions'] == 19136
    assert spec['performance_promotion'] is False and spec['development_allowed'] is False
    destination = tmp_path / 'configs/experiments'
    destination.mkdir(parents=True)
    (destination / 'canonical_task_replay.json').write_text('{}')
    with pytest.raises(ValueError, match='fixed'):
        protocol(tmp_path)


def test_full_window_boundary_contract_rejects_future_even_if_digest_matches():
    episode = {'split': 'train', 'bars': [{'session': s} for s in
               ('2017-12-29', '2018-01-02', '2020-12-02')]}
    episodes = {'fixture': episode}
    spec = {'episodes': 1, 'predictions': 2, 'window_episodes_sha256': digest(episodes),
            'anchor_session': '2017-12-29', 'first_execution_session': '2018-01-02',
            'last_session': '2020-12-02'}
    validate_window({'episodes': episodes}, spec)
    for change in ('split', 'last', 'count'):
        altered = deepcopy(episodes)
        if change == 'split':
            altered['fixture']['split'] = 'validation'
        elif change == 'last':
            altered['fixture']['bars'][-1]['session'] = '2024-01-01'
        else:
            altered['fixture']['bars'].pop()
        altered_spec = {**spec, 'window_episodes_sha256': digest(altered)}
        with pytest.raises(ValueError):
            validate_window({'episodes': altered}, altered_spec)


def test_inference_guard_rejects_optimization_native_prediction_and_restores():
    from stable_baselines3 import SAC
    import torch
    original = SAC.predict
    with pytest.raises(LookupError):
        with inference_only():
            for method in (SAC.learn, SAC.train, SAC.predict, torch.Tensor.backward,
                           torch.optim.Adam.step):
                with pytest.raises(RuntimeError, match='forbids'):
                    method(None)
            raise LookupError('handled diagnostic failure')
    assert SAC.predict is original
