"""Local untrained/manual-Adam snapshots; actual learning remains Colab-only."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.rl.recurrent_sac.prediction import FrozenRecurrentActor
from rl_stocks.training.recurrent_sac_checkpoint import load_checkpoint, save_checkpoint
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner

LINEAGE = dict(source='fixture', dataset='synthetic', observation_schema='fixture-v1')


def learner():
    with torch.random.fork_rng():
        torch.manual_seed(11)
        return RecurrentSACLearner(RecurrentSACNetworks(3, 8), max_updates=4)


def manual_adam(model):
    # Serialization fixture only: no backward, optimizer.step, or data-based fitting.
    model.updates = model.attempts = 1
    model.completed_steps = dict.fromkeys(model.optimizers, 1)
    for optimizer in model.optimizers.values():
        for group in optimizer.param_groups:
            for param in group['params']:
                optimizer.state[param] = dict(step=torch.tensor(1.),
                                               exp_avg=torch.full_like(param, .01),
                                               exp_avg_sq=torch.full_like(param, .002))


@pytest.mark.parametrize('manual', [False, True])
def test_all_tensors_raw_entropy_adam_and_frozen_carried_prediction_restore(tmp_path, manual):
    model = learner()
    if manual:
        manual_adam(model)
    before = torch.get_rng_state().clone()
    destination = tmp_path/'checkpoint'
    save_checkpoint(model, destination, lineage=LINEAGE)
    restored, manifest = load_checkpoint(destination, expected_lineage=LINEAGE)
    assert torch.equal(before, torch.get_rng_state())
    assert restored.updates == int(manual) and manifest['lineage'] == LINEAGE
    for key, value in model.networks.state_dict().items():
        torch.testing.assert_close(value, restored.networks.state_dict()[key], rtol=0, atol=0)
    assert torch.equal(model.log_alpha, restored.log_alpha)
    for role, optimizer in model.optimizers.items():
        a, b = optimizer.state_dict(), restored.optimizers[role].state_dict()
        assert a['param_groups'] == b['param_groups']
        for param, state in a['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, b['state'][param][name], rtol=0, atol=0)
    first = FrozenRecurrentActor(model.networks, observation_schema='fixture-v1')
    second = FrozenRecurrentActor(restored.networks, observation_schema='fixture-v1')
    state1 = state2 = None
    for i in range(6):
        action1, state1 = first.predict([i, .2, -.1], observation_schema='fixture-v1', state=state1)
        action2, state2 = second.predict([i, .2, -.1], observation_schema='fixture-v1', state=state2)
        np.testing.assert_array_equal(action1, action2)
        np.testing.assert_array_equal(state1, state2)
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(FileExistsError):
        save_checkpoint(model, destination, lineage=LINEAGE)


def test_local_update_rejects_before_any_mutation_or_gradient(tmp_path):
    model = learner()
    before = deepcopy(model.networks.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='Colab CPU'):
        model.update(None)
    assert model.attempts == model.updates == 0 and not model.failed
    assert all(not optimizer.state for optimizer in model.optimizers.values())
    assert all(p.grad is None for p in model.networks.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        assert torch.equal(value, model.networks.state_dict()[key])


def test_checksum_and_lineage_rejection(tmp_path):
    destination = tmp_path/'checkpoint'
    save_checkpoint(learner(), destination, lineage=LINEAGE)
    with pytest.raises(ValueError, match='lineage'):
        load_checkpoint(destination, expected_lineage={})
    state = destination/'state.pt'
    state.write_bytes(state.read_bytes()+b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(destination, expected_lineage=LINEAGE)


def test_inconsistent_native_adam_counter_rejects_even_with_updated_file_hash(tmp_path):
    destination = tmp_path/'checkpoint'
    model = learner()
    manual_adam(model)
    save_checkpoint(model, destination, lineage=LINEAGE)
    state = torch.load(destination/'state.pt', weights_only=True)
    state['optimizers']['actor']['state'][0]['step'] = torch.tensor(2.)
    torch.save(state, destination/'state.pt')
    manifest = json.loads((destination/'manifest.json').read_text())
    manifest['sha256'] = hashlib.sha256((destination/'state.pt').read_bytes()).hexdigest()
    (destination/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Adam'):
        load_checkpoint(destination, expected_lineage=LINEAGE)


def test_partial_failed_learner_cannot_be_saved(tmp_path):
    model = learner()
    model.failed = True
    with pytest.raises(ValueError, match='partial'):
        save_checkpoint(model, tmp_path/'failed', lineage=LINEAGE)
    assert not (tmp_path/'failed').exists()


def test_frozen_private_copy_reset_shape_schema_and_caller_state():
    model = learner()
    predictor = FrozenRecurrentActor(model.networks, observation_schema='fixture-v1')
    initial, state = predictor.predict([1, 2, 3], observation_schema='fixture-v1')
    before = state.copy()
    carried, _ = predictor.predict([1, 2, 3], observation_schema='fixture-v1', state=state)
    reset, _ = predictor.predict([1, 2, 3], observation_schema='fixture-v1', state=state,
                                 episode_start=True)
    np.testing.assert_array_equal(initial, reset)
    np.testing.assert_array_equal(state, before)
    assert not np.array_equal(initial, carried)
    with torch.no_grad():
        model.networks.actor.mean.bias.add_(10)
    again, _ = predictor.predict([1, 2, 3], observation_schema='fixture-v1')
    np.testing.assert_array_equal(initial, again)
    with pytest.raises(ValueError, match='schema'):
        predictor.predict([1, 2, 3], observation_schema='other')
    with pytest.raises(ValueError, match='state'):
        predictor.predict([1, 2, 3], observation_schema='fixture-v1', state=np.zeros((1, 8)))


def test_missing_lineage_and_unsupported_geometry_reject(tmp_path):
    with pytest.raises(ValueError, match='lineage'):
        save_checkpoint(learner(), tmp_path/'empty', lineage={})
    assert not (tmp_path/'empty').exists()
    with pytest.raises(ValueError):
        RecurrentSACLearner(RecurrentSACNetworks(3, 8), max_updates=True)
