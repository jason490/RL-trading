"""Fixed training-prefix smoke preparation; no optimizer or environment rollout."""
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from rl_stocks.core.contracts import digest
from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.training.distributional_critics import capture_global_rng
from rl_stocks.training.recurrent_payload import RecurrentPayloadStore
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner
from rl_stocks.training.sac_update_input import validate_payload
from rl_stocks.training.sac_update_worker import prepare as prepare_control, preserved_rng

PROTOCOL_SHA256 = 'a530831e6f8a1c28b6e629ef8143fce43d91e93aaa8826ed39ea6e4ba38fea12'
ROOT = Path(__file__).resolve().parents[3]


def protocol():
    value = json.loads((ROOT/'configs/experiments/recurrent_sac_smoke.json').read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('fixed recurrent SAC learned smoke required')
    return value


def replay_store(arrays):
    """Segment identities are replay ordinals; source symbols are unavailable."""
    spec = protocol()
    store = RecurrentPayloadStore(spec['replay_rows'], spec['observation_size'])
    episode = step = 0
    for i in range(spec['replay_rows']):
        done, timeout = bool(arrays['dones'][i, 0]), bool(arrays['timeouts'][i, 0])
        store.append_raw(observation=arrays['observations'][i, 0],
                         next_observation=arrays['next_observations'][i, 0],
                         action=arrays['actions'][i, 0], reward=arrays['rewards'][i, 0],
                         environment=0, episode=episode, step=step,
                         symbol=f'REPLAY-SEGMENT-{episode}',
                         terminated=done and not timeout, truncated=done and timeout)
        step += 1
        if done:
            episode += 1
            step = 0
    return store


@dataclass
class PreparedSmoke:
    learner: RecurrentSACLearner
    control: object
    recurrent_rng: dict
    control_rng: dict
    batches: tuple
    loss_rows: tuple
    arrays: dict
    control_initial: dict

    def native_sample(self, update_index):
        rows = self.loss_rows[update_index]
        arrays = self.arrays
        def tensor(name):
            return torch.tensor(arrays[name][rows, 0].copy())
        done = arrays['dones'][rows, 0] * (1 - arrays['timeouts'][rows, 0])
        return ReplayBufferSamples(observations=tensor('observations'), actions=tensor('actions'),
                                   next_observations=tensor('next_observations'),
                                   dones=torch.tensor(done[:, None]),
                                   rewards=tensor('rewards').reshape(-1, 1), discounts=None)


def prepare_smoke(payload):
    spec = protocol()
    arrays = validate_payload(payload)
    if digest(payload) != spec['input_identity']:
        raise ValueError('exact original training-prefix input required')
    store = replay_store(arrays)
    rng = np.random.default_rng(spec['seed'])
    batches, loss_rows = [], []
    for _ in range(spec['gradient_updates_per_arm']):
        sequences = store.index.sample(batch_size=spec['windows_per_batch'],
                                       burn_in=spec['burn_in'], unroll=spec['unroll'], rng=rng)
        batch = store.gather(sequences)
        rows = batch.insertion_ids[:, spec['burn_in']:].reshape(-1).copy()
        # Explicitly verify the flattened control consumes identical action/obs/reward rows.
        for name, actual in [('observations', batch.observations[:, spec['burn_in']:-1]),
                             ('next_observations', batch.observations[:, spec['burn_in']+1:]),
                             ('actions', batch.actions[:, spec['burn_in']:]),
                             ('rewards', batch.rewards[:, spec['burn_in']:])]:
            expected = arrays[name][rows, 0].reshape(actual.shape)
            if not np.array_equal(actual, expected):
                raise ValueError('matched recurrent/native loss rows differ')
        rows.flags.writeable = False
        batches.append(batch)
        loss_rows.append(rows)
    control = None
    try:
        with preserved_rng():
            control, initial = prepare_control(payload, 'original_defaults')
            control_rng = capture_global_rng()
            random.seed(spec['seed'])
            np.random.seed(spec['seed'])
            torch.manual_seed(spec['seed'])
            networks = RecurrentSACNetworks(spec['observation_size'], spec['hidden_size'])
            learner = RecurrentSACLearner(networks, max_updates=spec['gradient_updates_per_arm'],
                                          learning_rate=spec['learning_rate'],
                                          gamma=spec['gamma'], tau=spec['tau'])
            recurrent_rng = capture_global_rng()
        return PreparedSmoke(learner, control, recurrent_rng, control_rng, tuple(batches),
                              tuple(loss_rows), arrays, initial)
    except BaseException:
        if control is not None:
            control.get_env().close()
        raise
