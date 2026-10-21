"""Audit retained recurrent models without data access or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from rl_stocks.rl.registry.prediction import PredictionStream
from rl_stocks.environment.observations.state import LEGACY_FIELDS, EXECUTION_FIELDS
from rl_stocks.core.contracts import digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    torch.set_num_threads(1)
    records = []
    for metadata in sorted((root / 'artifacts/colab').glob('*/retrieved/artifacts/runs/*recurrent_ppo*/metadata.json')):
        info = json.loads(metadata.read_text())
        if info.get('git', {}).get('revision') != 'e6ea781f04799dffaccb3e93c9ad49c1de714217' or info.get('requested_steps') != 100000:
            continue
        manifests = list(metadata.parents[2].glob('frozen/*/manifest.json'))
        assert len(manifests) == 1
        manifest_path = manifests[0]
        manifest = json.loads(manifest_path.read_text())
        assert digest(manifest) == manifest_path.parent.name
        assert manifest['metadata']['algorithm'] == 'recurrent_ppo'
        assert manifest['metadata']['seed'] == info['seed']
        assert len(manifest['names']) == 7
        assert manifest['metadata'].get('observation', {'profile': 'legacy'})['profile'] == 'legacy'
        for phase, name in [('selected', 'best.zip'), ('final', 'latest.zip')]:
            path = metadata.parent / name
            before = {str(p.relative_to(root)): sha(p) for p in (metadata, path, manifest_path)}
            model = RecurrentPPO.load(path, device='cpu')
            model.policy.set_training_mode(False)
            shape = model.observation_space.shape
            assert shape == (12,), 'Historical input contract changed'
            rng = np.random.default_rng(11)
            observations = rng.normal(0, .25, size=(16, *shape)).astype(np.float32)
            updates = model._n_updates
            steps = model.num_timesteps
            tensors = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
            stream = PredictionStream(model)
            state = None
            actions = []
            reset_actions = []
            for i, obs in enumerate(observations):
                action = stream.predict(obs)
                reference, state = model.predict(obs, state=state, episode_start=np.array([i == 0]), deterministic=True)
                np.testing.assert_array_equal(action, reference)
                actions.append(action)
                reset_actions.append(PredictionStream(model).predict(obs))
            restarted = PredictionStream(model)
            for obs, expected in zip(observations, actions):
                np.testing.assert_array_equal(restarted.predict(obs), expected)
            assert steps == model.num_timesteps and updates == model._n_updates
            assert all(torch.equal(v, model.policy.state_dict()[k]) for k, v in tensors.items())
            assert all(sha(root / p) == digest for p, digest in before.items())
            lstm = model.policy.lstm_actor
            records.append(dict(seed=info['seed'], phase=phase, source_files=before,
                                observation_size=shape[0], market_inputs=len(manifest['names']), market_names=manifest['names'], portfolio_inputs=len(LEGACY_FIELDS),
                                missing_explicit_execution_fields=list(EXECUTION_FIELDS), actual_steps=steps,
                                actor_hidden_size=lstm.hidden_size, actor_layers=lstm.num_layers,
                                separate_critic_lstm=model.policy.lstm_critic is not None,
                                carried_sequence_exact=True, independent_stream_reset_exact=True,
                                synthetic_reset_vs_carried_max_action_delta=float(np.max(np.abs(np.asarray(actions)-reset_actions))),
                                unchanged_weights_updates_and_sources=True))
    assert len(records) == 6 and {r['seed'] for r in records} == {11, 23, 37}
    return dict(records=records, synthetic_observations_per_model=16, training_performed=False,
                dataset_opened=False, test_evaluated=False, colab_allocated=False,
                conclusion='Historical recurrent models use legacy12 inputs. Memory carry/reset works on fixed synthetic probes; this does not establish return improvement or equivalence to execution_full21.',
                objective_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    result = audit(Path.cwd())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Audited six retained recurrent phases; no fitting or dataset access.')
