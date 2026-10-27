"""Fixed protocol and segmentation/row matching fixtures, without native learning."""
from copy import deepcopy

import numpy as np
import pytest

from rl_stocks.training.recurrent_sac_smoke import protocol, replay_store, PreparedSmoke


def arrays():
    n = protocol()['replay_rows']
    observations = np.arange(n*21, dtype=np.float32).reshape(n, 1, 21)/100
    nxt = np.concatenate([observations[1:], observations[-1:] + 1])
    # Match native terminal next observation at the segment break (not reset obs).
    nxt[511] = 7
    dones = np.zeros((n, 1), dtype=np.float32)
    dones[511] = 1
    return dict(observations=observations, next_observations=nxt,
                actions=np.zeros((n, 1, 4), dtype=np.float32),
                rewards=np.arange(n, dtype=np.float32).reshape(n, 1)/1000,
                dones=dones, timeouts=dones.copy())


def test_fixed_scope_and_shared_project_limits():
    spec = protocol()
    assert spec['maximum_total_model_updates'] == 2*spec['gradient_updates_per_arm'] == 8
    assert spec['maximum_total_adam_steps'] == 24
    assert spec['environment_transitions_collected'] == 0
    assert spec['execution']['project_ceiling'] == 150
    assert spec['execution']['project_teardown'] == 130
    assert spec['execution']['minimum_account_balance'] == 150
    assert spec['execution']['account_teardown_balance'] == 170


def test_segmentation_matching_rows_and_native_timeout_mask():
    data = arrays()
    store = replay_store(data)
    assert store.index.valid_window_count(burn_in=16, unroll=16) == 939
    sequences = store.index.sample(batch_size=8, burn_in=16, unroll=16,
                                  rng=np.random.default_rng(11))
    batch = store.gather(sequences)
    rows = batch.insertion_ids[:, 16:].reshape(-1)
    prepared = PreparedSmoke(None, None, {}, {}, (batch,), (rows,), data, {})
    native = prepared.native_sample(0)
    np.testing.assert_array_equal(native.observations.numpy(), batch.observations[:, 16:-1].reshape(-1, 21))
    np.testing.assert_array_equal(native.next_observations.numpy(), batch.observations[:, 17:].reshape(-1, 21))
    np.testing.assert_array_equal(native.actions.numpy(), batch.actions[:, 16:].reshape(-1, 4))
    assert native.discounts is None and native.rewards.shape == (128, 1)
    # Force a boundary-inclusive native sample: timeout bootstraps, true terminal does not.
    prepared.loss_rows = (np.array([511]),)
    assert prepared.native_sample(0).dones.item() == 0
    data['timeouts'][511] = 0
    assert prepared.native_sample(0).dones.item() == 1


def test_unflagged_reset_observation_is_rejected_without_source_mutation():
    data = arrays()
    data['dones'][511] = 0
    data['timeouts'][511] = 0
    before = deepcopy(data)
    with pytest.raises(ValueError, match='continue'):
        replay_store(data)
    for name in data:
        np.testing.assert_array_equal(data[name], before[name])


def test_actual_worker_rejects_local_before_output_or_native_construction(tmp_path):
    from rl_stocks.training.recurrent_sac_smoke_worker import execute_colab
    destination = tmp_path/'forbidden'
    with pytest.raises(RuntimeError, match='Colab CPU'):
        execute_colab({}, destination)
    assert not destination.exists()


def test_native_audit_requires_gradients_and_fixed_phase_order():
    import torch
    from rl_stocks.training.recurrent_sac_smoke_worker import OptimizerAudit
    parameters = {name: torch.tensor(1., requires_grad=True) for name in ('actor', 'critic', 'entropy')}
    optimizers = {name: torch.optim.Adam([p]) for name, p in parameters.items()}
    audit = OptimizerAudit(optimizers, 4)
    try:
        with pytest.raises(ValueError, match='order'):
            audit.before('actor', optimizers['actor'])
        assert not any(audit.attempts.values())
        with pytest.raises(ValueError, match='gradients'):
            audit.before('entropy', optimizers['entropy'])
        assert audit.steps['entropy'] == 0  # failed pre-step is never a completed update
        assert not any(opt.state for opt in optimizers.values())
    finally:
        audit.close()
    assert not audit.handles
