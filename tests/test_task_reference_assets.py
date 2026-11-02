import io
import json
from pathlib import Path
import tarfile

import pytest
import torch
from stable_baselines3 import SAC

from test_policy_architecture import environment
from rl_stocks.environment.observations.task import InstrumentTypeObservation
from rl_stocks.evaluation.canonical_diagnostic import inference_only
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.rl.sac.task_factory import create_task_sac
from rl_stocks.runtime.colab.canonical_market_probe import file_sha
from rl_stocks.training.task_reference_assets import bind_archive, check_model, protocol


@pytest.fixture(autouse=True)
def no_local_optimization():
    with inference_only():
        yield


def initial_model(role='masked_shared'):
    env = environment()
    symbol = env.episodes[0]['symbol']
    wrapped = InstrumentTypeObservation(env, [{'symbol':symbol, 'kind':'etf',
        'source':'synthetic fixture', 'known_from':'2010-01-01'}], task_role=role)
    return create_task_sac(wrapped, task_role=role, smoke=False)


@pytest.mark.parametrize('role', ['masked_shared', 'tagged_shared', 'tagged_routed'])
def test_initial_physical_checkpoint_consistency_without_inference(tmp_path, role):
    model = initial_model(role)
    path = tmp_path / 'initial.zip'
    model.save(path)
    loaded = SAC.load(path, device='cpu')
    before = fingerprint(loaded.policy.state_dict())
    result = check_model(loaded, role=role, steps=0)
    assert result['updates'] == 0 and result['policy_sha256'] == before
    assert fingerprint(loaded.policy.state_dict()) == before
    assert len(loaded.actor.optimizer._optimizer_step_pre_hooks) == 0


@pytest.mark.parametrize('corruption', ['counter', 'moments', 'role', 'weights', 'hyperparameters'])
def test_physical_check_rejects_counters_missing_moments_role_nonfinite_or_drift(corruption):
    model = initial_model()
    role, steps = 'masked_shared', 0
    if corruption == 'counter':
        model._n_updates = 1
    elif corruption == 'moments':
        model.num_timesteps, model._n_updates, steps = 10000, 9000, 10000
    elif corruption == 'role':
        role = 'tagged_shared'
    elif corruption == 'weights':
        with torch.no_grad():
            next(model.actor.parameters()).flatten()[0] = float('nan')
    else:
        model.batch_size = 64
    with pytest.raises(ValueError):
        check_model(model, role=role, steps=steps)
    assert len(model.actor.optimizer._optimizer_step_pre_hooks) == 0


def archive_fixture(tmp_path):
    folder = tmp_path / 'control'
    folder.mkdir()
    (folder / 'initial.zip').write_bytes(b'fixture checkpoint bytes; never loaded')
    archive = tmp_path / 'archive.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        raw = json.dumps({'source':'fixture'}).encode()
        member = tarfile.TarInfo('artifacts/completion.json')
        member.size = len(raw)
        stream.addfile(member, io.BytesIO(raw))
        stream.add(folder / 'initial.zip', arcname='artifacts/task-screen/initial.zip')
    return archive, folder


def test_extracted_files_are_bound_to_actual_archive_bytes(tmp_path):
    archive, folder = archive_fixture(tmp_path)
    proof, completion = bind_archive(archive, folder, expected_sha=file_sha(archive))
    assert proof['initial.zip'] == file_sha(folder / 'initial.zip')
    assert completion['source'] == 'fixture'
    (folder / 'initial.zip').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='differs'):
        bind_archive(archive, folder, expected_sha=file_sha(archive))


def test_archive_identity_and_extra_local_file_reject(tmp_path):
    archive, folder = archive_fixture(tmp_path)
    with pytest.raises(ValueError, match='checksum'):
        bind_archive(archive, folder, expected_sha='a'*64)
    (folder / 'unbound.zip').write_bytes(b'not in original archive')
    with pytest.raises(ValueError, match='file set'):
        bind_archive(archive, folder, expected_sha=file_sha(archive))


def test_declaration_preserves_training_gates_and_denies_allocation():
    spec = protocol(Path(__file__).resolve().parents[1])
    assert spec['colab_allocation_allowed_by_this_declaration'] is False
    assert spec['additional_control_training_allowed'] is False
    assert spec['budget']['shared_baseline_units'] == 39.92585919127071
    assert spec['budget']['maximum_cumulative_study_units'] == 2.
    assert spec['retired_test_access'] is False


@pytest.mark.parametrize('corruption', ['equity', 'baseline', 'directional'])
def test_raw_economic_checks_reject_inconsistent_saved_metrics(corruption):
    from copy import deepcopy
    from dataclasses import asdict
    from rl_stocks.environment.execution.simulator import SimulationConfig
    from rl_stocks.evaluation.runner import _evaluate
    from rl_stocks.training.task_reference_assets import check_economic_report
    env = environment()
    episode = deepcopy(env.episodes[0])
    for key in ('bars', 'features', 'atr'):
        episode[key] = episode[key][:9]
    class Cash:
        def predict(self, observation, deterministic=True):
            return (0., 0., 0., 0.), None
    data = {'episodes':{'fixture':episode}, 'names':env.names, 'normalizer':asdict(env.normalizer)}
    result = _evaluate(Cash(), data, 'train', SimulationConfig(), 'execution_full')['fixture']
    baseline = deepcopy(result['baselines'])
    check_economic_report(result, episode, baseline)
    if corruption == 'equity':
        result['rows'][-1]['equity'] += 10.
    elif corruption == 'baseline':
        result['baselines']['buy_hold']['metrics']['net_return'] += .01
    else:
        result['directional']['long_net_pnl'] += 1.
    with pytest.raises(ValueError, match='inconsistent'):
        check_economic_report(result, episode, baseline)


def test_replay_hash_covers_numpy_shape_dtype_and_bytes():
    from types import SimpleNamespace
    import numpy as np
    from rl_stocks.training.task_reference_assets import replay_fingerprint
    names = ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')
    def fixture():
        return SimpleNamespace(**{name:np.arange(6, dtype=np.float32).reshape(2,3) for name in names})
    original = replay_fingerprint(fixture())
    assert replay_fingerprint(fixture()) == original
    for change in ('shape', 'dtype', 'value'):
        replay = fixture()
        if change == 'shape':
            replay.rewards = replay.rewards.reshape(3,2)
        elif change == 'dtype':
            replay.rewards = replay.rewards.view(np.int32)
        else:
            replay.rewards[0,0] += 1.
        assert replay_fingerprint(replay) != original
