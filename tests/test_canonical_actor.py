from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
import subprocess
import sys

import pytest

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_actor import CanonicalActor, OUTPUT, SCHEMA, encode32
from rl_stocks.inference.canonical_math import CONTRACT, bits32, certified_tanh


def layer(inputs, outputs, bias=0.):
    return {'weights': [[encode32(0.)] * inputs for _ in range(outputs)],
            'biases': [encode32(bias)] * outputs}


def fixture(role='tagged_routed'):
    body = {'schema': SCHEMA, 'arithmetic': CONTRACT, 'role': role,
            'input_size': 22, 'hidden_sizes': [64, 64], 'output_size': 4,
            'source_model_sha256': 'a' * 64, 'output_transform': OUTPUT,
            'layers': [layer(22, 64), layer(64, 64)],
            'heads': ({'stock': layer(64, 4, -1.), 'etf': layer(64, 4, 1.)}
                      if role == 'tagged_routed' else {'shared': layer(64, 4)})}
    return {'actor': body, 'sha256': digest(body)}


def test_routing_and_immutable_state():
    envelope = fixture()
    actor = CanonicalActor(envelope)
    stock = actor.predict([0.] * 21 + [-1.])
    etf = actor.predict([0.] * 21 + [1.])
    assert all(abs(v + certified_tanh(1.)) < 1e-7 for v in stock)
    assert all(abs(v - certified_tanh(1.)) < 1e-7 for v in etf)
    envelope['actor']['heads'].clear()
    view = actor.manifest
    view['actor']['heads'].clear()
    assert actor.predict([0.] * 21 + [1.]) == etf
    with pytest.raises(FrozenInstanceError):
        actor._role = 'masked_shared'
    for bad in ([0.] * 22, [0.] * 21, [float('nan')] * 21 + [1.]):
        with pytest.raises(ValueError):
            actor.predict(bad)


@pytest.mark.parametrize('role', ['masked_shared', 'tagged_shared'])
def test_shared_tag_contract(role):
    actor = CanonicalActor(fixture(role))
    tag = 0 if role == 'masked_shared' else -1
    assert actor.predict([0.] * 21 + [tag]) == (0.,) * 4
    with pytest.raises(ValueError, match='tag'):
        actor.predict([0.] * 21 + [1 if tag == 0 else 0])


@pytest.mark.parametrize('mutation', [
    lambda b: b.update(arithmetic='native'),
    lambda b: b.update(extra=True),
    lambda b: b['layers'][0]['weights'][0].pop(),
    lambda b: b['heads']['stock']['biases'].__setitem__(0, '7f800000'),
    lambda b: b['heads'].update(shared=layer(64, 4)),
])
def test_schema_rejection(mutation):
    envelope = fixture()
    mutation(envelope['actor'])
    envelope['sha256'] = digest(envelope['actor'])
    with pytest.raises(ValueError):
        CanonicalActor(envelope)


def test_checksums_sources_files(tmp_path):
    envelope = fixture()
    with pytest.raises(ValueError, match='source'):
        CanonicalActor(envelope, expected_source='b' * 64)
    with pytest.raises(ValueError, match='role'):
        CanonicalActor(envelope, expected_role='masked_shared')
    bad = deepcopy(envelope)
    bad['actor']['role'] = 'tagged_shared'
    with pytest.raises(ValueError, match='checksum'):
        CanonicalActor(bad)
    path = tmp_path / 'actor.json'
    path.write_text(json.dumps(envelope))
    assert CanonicalActor.load(path).manifest == envelope
    path.write_text('{"actor":{},"actor":{},"sha256":"x"}')
    with pytest.raises(ValueError, match='duplicate'):
        CanonicalActor.load(path)
    path.write_bytes(b' ' * (512 * 1024 + 1))
    with pytest.raises(ValueError, match='size'):
        CanonicalActor.load(path)


def test_fresh_process_no_native_backend(tmp_path):
    path = tmp_path / 'actor.json'
    path.write_text(json.dumps(fixture()))
    code = '''
import decimal, json, sys
from rl_stocks.inference.canonical_actor import CanonicalActor
from rl_stocks.inference.canonical_math import bits32
decimal.getcontext().prec = 2
decimal.getcontext().traps[decimal.Inexact] = True
actor = CanonicalActor.load(sys.argv[1])
print(json.dumps([bits32(v) for v in actor.predict([0.] * 21 + [1.])]))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
    expected = [bits32(v) for v in CanonicalActor(fixture()).predict([0.] * 21 + [1.])]
    for selector in ('avx2', 'default'):
        env = {**os.environ, 'ATEN_CPU_CAPABILITY': selector, 'MKL_CBWR': 'COMPATIBLE'}
        result = subprocess.run([sys.executable, '-c', code, str(path)], env=env,
                                check=True, capture_output=True, text=True, timeout=30)
        assert json.loads(result.stdout) == expected


@pytest.mark.parametrize('role', ['masked_shared', 'tagged_shared', 'tagged_routed'])
def test_export_trusted_checkpoint_without_fit(tmp_path, monkeypatch, role):
    import random
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from test_task_bundle import bundle
    from rl_stocks.inference.canonical_export import export_actor

    def forbidden(*args, **kwargs):
        pytest.fail('local optimization attempted')
    for method in ('learn', 'train'):
        monkeypatch.setattr(SAC, method, forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    model, _, _ = bundle(tmp_path, role)
    before = {k: v.clone() for k, v in model.actor.state_dict().items()}
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    source = tmp_path / 'model.zip'
    path = export_actor(source, tmp_path / 'actor.json')
    actor = CanonicalActor.load(path, expected_role=role,
                               expected_source=hashlib.sha256(source.read_bytes()).hexdigest())
    assert random.getstate() == py_rng
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert np.random.get_state()[2:] == np_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert all(torch.equal(v, before[k]) for k, v in model.actor.state_dict().items())
    x = [.125] * 21 + [0. if role == 'masked_shared' else 1.]
    assert actor.predict(x) == CanonicalActor.load(path).predict(x)
    with pytest.raises(FileExistsError):
        export_actor(source, path)


def test_hidden_exact_dot_relu_and_output_scaling():
    import numpy as np
    envelope = fixture('tagged_shared')
    body = envelope['actor']
    body['layers'][0]['weights'][0][:3] = [encode32(v) for v in (1e8, 1., -1e8)]
    body['layers'][0]['biases'][1] = encode32(-10.)
    body['layers'][1]['weights'][0][0] = encode32(1.)
    body['layers'][1]['weights'][0][1] = encode32(100.)
    body['heads']['shared']['weights'][0][0] = encode32(1.)
    envelope['sha256'] = digest(body)
    actual = CanonicalActor(envelope).predict([1.] * 21 + [1.])
    squashed = np.float32(certified_tanh(1.))
    expected = np.float32(-1.) + np.float32(.5) * (squashed + np.float32(1.)) * np.float32(2.)
    assert bits32(actual[0]) == bits32(expected)
    assert actual[1:] == (0.,) * 3


def test_export_rejects_mutated_geometry_and_nonfinite(tmp_path, monkeypatch):
    import torch
    from stable_baselines3 import SAC
    from test_task_bundle import bundle
    from rl_stocks.inference.canonical_export import export_actor
    def forbidden(*args, **kwargs):
        pytest.fail('local optimization attempted')
    for method in ('learn', 'train'):
        monkeypatch.setattr(SAC, method, forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    model, _, _ = bundle(tmp_path)
    monkeypatch.setattr(SAC, 'load', lambda *a, **kw: model)
    source = tmp_path / 'model.zip'
    model.actor.latent_pi[1] = torch.nn.Tanh()
    with pytest.raises(ValueError, match='hidden layers'):
        export_actor(source, tmp_path / 'bad.json')
    assert not (tmp_path / 'bad.json').exists()
    model.actor.latent_pi[1] = torch.nn.ReLU()
    with torch.no_grad():
        model.actor.mu.stock.bias[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        export_actor(source, tmp_path / 'bad.json')
