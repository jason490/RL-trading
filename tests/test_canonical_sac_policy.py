from copy import deepcopy
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from test_canonical_tanh import fixture
from test_trpo_feasibility import ContractEnv
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.canonical_actor import _unscale, OUTPUT
from rl_stocks.inference.canonical_math import certified_tanh
from rl_stocks.inference.canonical_sac_export import export_policy
from rl_stocks.inference.canonical_sac_policy import CanonicalSACPolicy, SCHEMA


def envelope():
    body = deepcopy(fixture()['actor'])
    body.update(schema=SCHEMA, algorithm='sac', output_transform=OUTPUT)
    return {'actor': body, 'sha256': digest(body)}


def test_standard_relu_and_squash_contract_is_immutable():
    actor = CanonicalSACPolicy(envelope(), expected_source='a' * 64)
    negative = actor.predict([-1.] + [0.] * 20)
    positive = actor.predict([1.] + [0.] * 20)
    assert negative[0] == 0 and positive[0] == _unscale(certified_tanh(1.))
    assert positive[1:] == tuple(_unscale(certified_tanh(v)) for v in (2., -2., .5))
    changed = actor.manifest
    changed['actor']['input_size'] = 22
    assert actor.manifest['actor']['input_size'] == 21
    with pytest.raises(FrozenInstanceError):
        actor._encoded = 'changed'
    for value in ([0.] * 22, [float('nan')] * 21, [True] * 21):
        with pytest.raises(ValueError):
            actor.predict(value)
    wrong = envelope()
    wrong['actor']['algorithm'] = 'ppo'
    wrong['sha256'] = digest(wrong['actor'])
    with pytest.raises(ValueError, match='contract'):
        CanonicalSACPolicy(wrong)


def test_actor_load_is_stdlib_only_and_rejects_duplicate_fields(tmp_path):
    path = tmp_path / 'actor.json'
    path.write_bytes(canonical(envelope()))
    source = str(Path(__file__).resolve().parents[1] / 'src')
    program = ('import sys;sys.path.insert(0,sys.argv[1]);'
               'from rl_stocks.inference.canonical_sac_policy import CanonicalSACPolicy;'
               'a=CanonicalSACPolicy.load(sys.argv[2]);a.predict([0.]*21);'
               'assert "torch" not in sys.modules and "numpy" not in sys.modules')
    subprocess.run([sys.executable, '-I', '-c', program, source, str(path)],
                   check=True, timeout=30, env=os.environ.copy())
    path.write_text('{"actor":{},"actor":{},"sha256":"bad"}')
    with pytest.raises(ValueError, match='duplicate'):
        CanonicalSACPolicy.load(path)


def test_trusted_native_export_preserves_rng_source_and_no_learning(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('local optimization forbidden')
    monkeypatch.setattr(SAC, 'learn', forbidden)
    monkeypatch.setattr(SAC, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    model = SAC('MlpPolicy', ContractEnv(), policy_kwargs={'net_arch': [64, 64]},
                seed=11, device='cpu')
    path = tmp_path / 'initial.zip'
    model.save(path)
    original = checksum(path)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    output = export_policy(path, tmp_path / 'reference.json')
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_rng[0] and actual_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
    assert checksum(path) == original and model.num_timesteps == 0
    actor = CanonicalSACPolicy.load(output, expected_source=original)
    x = np.random.default_rng(9).normal(size=(8, 21)).astype(np.float32)
    predictions = [actor.predict(row) for row in x]
    assert predictions == [actor.predict(row) for row in x]
    assert np.isfinite(predictions).all() and np.max(np.abs(predictions)) <= 1
    assert not model.actor.optimizer.state and not model.critic.optimizer.state
    with pytest.raises(FileExistsError):
        export_policy(path, output)
    # Explicit corruption fixture, not a fit/update.
    with torch.no_grad():
        model.actor.mu.weight[0, 0] = float('nan')
    model.save(tmp_path / 'nonfinite.zip')
    with pytest.raises(ValueError, match='finite binary32'):
        export_policy(tmp_path / 'nonfinite.zip', tmp_path / 'bad.json')


def test_unreviewed_architecture_rejected(tmp_path):
    model = SAC('MlpPolicy', ContractEnv(), policy_kwargs={'net_arch': [32, 32]},
                seed=11, device='cpu')
    model.save(tmp_path / 'different.zip')
    with pytest.raises(ValueError, match='standard SAC21/64x64'):
        export_policy(tmp_path / 'different.zip', tmp_path / 'different.json')
