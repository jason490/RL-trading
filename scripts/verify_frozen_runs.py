"""Verify retrieved feed-forward/recurrent bundles against selected weights on causal episode prefixes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.data.generation.datasets import read_dataset, normalizer_from
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.inference.bundle import FrozenPolicy, checksum
from rl_stocks.rl.registry.factory import CLASSES


def check_recurrent_prediction(selected, frozen, observation, expected_state, actual_state, episode_start):
    """Check carried hidden/cell state, repeatability and caller-state immutability."""
    previous = None if actual_state is None else tuple(s.copy() for s in actual_state)
    expected, new_expected = selected.predict(observation, state=expected_state,
        episode_start=np.array([episode_start], dtype=bool), deterministic=True)
    actual, new_actual = frozen.predict_recurrent(observation, state=actual_state, episode_start=episode_start)
    repeat, repeat_state = frozen.predict_recurrent(observation, state=actual_state, episode_start=episode_start)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, repeat)
    for x, y, z in zip(new_expected, new_actual, repeat_state, strict=True):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(y, z)
    if previous is not None:
        for old, current in zip(previous, actual_state, strict=True):
            np.testing.assert_array_equal(old, current)
    return actual, new_expected, new_actual


def verify(roots, data_root):
    torch.set_num_threads(1)
    datasets, records, seen = {}, [], set()
    for root in roots:
        for path in sorted((root / 'runs').glob('*/metadata.json')):
            metadata = json.loads(path.read_text())
            if metadata['status'] != 'complete' or metadata['bundle_id'] in seen:
                raise ValueError('complete unique runs required')
            if metadata['algorithm'] not in ('ppo', 'sac', 'td3', 'recurrent_ppo'):
                raise ValueError('unsupported frozen policy algorithm')
            identity = metadata['dataset']
            if identity not in datasets:
                datasets[identity] = read_dataset(data_root / identity)
            dataset = datasets[identity]
            bundle = root / 'frozen' / metadata['bundle_id']
            feed = dataset['manifest']['provenance']['capabilities']['feed']
            frozen = FrozenPolicy(bundle, dataset['schema'], feed, 'cpu')
            if (frozen.manifest['names'] != dataset['names']
                    or frozen.manifest['normalizer'] != dataset['normalizer']
                    or frozen.manifest['data_manifest'] != dataset['manifest']):
                raise ValueError('frozen feature order, normalization or data manifest differs from dataset')
            selected = CLASSES[metadata['algorithm']].load(path.parent / 'best.zip', device='cpu')
            if checksum(path.parent / 'best.zip') != checksum(bundle / 'model.zip'):
                raise ValueError('frozen weights differ from selected weights')
            is_recurrent = metadata['algorithm'] == 'recurrent_ppo'
            before = canonical(asdict(frozen.normalizer))
            count, symbols = 0, []
            for episode in dataset['episodes'].values():
                if episode['split'] != 'validation':
                    continue
                env = TradingEnv([episode], normalizer_from(dataset['normalizer']),
                                 dataset['names'], SimulationConfig(**metadata['simulation']),
                                 observation_profile=metadata.get('observation',{}).get('profile','legacy'))
                try:
                    observation, _ = env.reset(seed=11, options={'episode': 0})
                    expected_state = actual_state = None
                    first_observation = observation.copy()
                    first_action = None
                    for index in range(3):
                        if is_recurrent:
                            actual, expected_state, actual_state = check_recurrent_prediction(
                                selected, frozen, observation, expected_state, actual_state, index == 0)
                        else:
                            expected = selected.predict(observation, deterministic=True)[0]
                            actual = frozen.predict(observation)
                            np.testing.assert_array_equal(actual, expected)
                            np.testing.assert_array_equal(actual, frozen.predict(observation))
                        if first_action is None:
                            first_action = actual.copy()
                        count += 1
                        observation, _, terminated, truncated, _ = env.step(actual)
                        if terminated or truncated:
                            break
                    if is_recurrent:
                        reset_action, _ = frozen.predict_recurrent(first_observation, state=None, episode_start=True)
                        np.testing.assert_array_equal(reset_action, first_action)
                finally:
                    env.close()
                symbols.append(episode['symbol'])
            if not count or before != canonical(asdict(frozen.normalizer)):
                raise ValueError('missing predictions or mutated normalization')
            for schema, source in (('incompatible-schema', feed), (dataset['schema'], 'incompatible-feed')):
                try:
                    FrozenPolicy(bundle, schema, source, 'cpu')
                except ValueError:
                    pass
                else:
                    raise ValueError('incompatible bundle request accepted')
            try:
                invalid = [float('nan')] * frozen.observation_size
                if is_recurrent:
                    frozen.predict_recurrent(invalid, state=None, episode_start=True)
                else:
                    frozen.predict(invalid)
            except ValueError:
                pass
            else:
                raise ValueError('nonfinite observation accepted')
            records.append({'bundle_id': bundle.name, 'dataset': identity, 'algorithm': metadata['algorithm'],
                'seed': metadata['seed'], 'symbols_checked': symbols, 'predictions_checked': count,
                'exact_prediction_parity': True, 'deterministic_repeat': True,
                'recurrent_sequence_and_reset_parity': is_recurrent,
                'normalization_and_data_match_dataset': True,
                'normalization_unchanged': True, 'schema_feed_nonfinite_rejection': True,
                'selected_weights_sha256': checksum(path.parent / 'best.zip')})
            seen.add(bundle.name)
    if not records:
        raise ValueError('no completed runs')
    return {'passed': True, 'bundles': records, 'device': 'cpu', 'training_performed': False,
            'test_evaluated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--data-root', type=Path, default=Path('data/datasets'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.roots, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print('Verified', len(value['bundles']), 'frozen bundles; exact CPU predictions and immutable normalization.')


if __name__ == '__main__':
    main()
