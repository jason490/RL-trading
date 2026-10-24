"""Frozen-source SAC inference only; execute with the pinned source PYTHONPATH."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import inspect
import json
import os
import signal
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC

from rl_stocks.core.files import write_once
from rl_stocks.core.provenance import source_identity
from rl_stocks.data.generation.datasets import normalizer_from, read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.training.sac_state import PROTOCOL_SHA256, protocol, validate_payload

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'src/rl_stocks/evaluation/soft_consistency.py'
_spec = importlib.util.spec_from_file_location('soft_consistency_numeric', HELPER)
numeric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(numeric)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, state):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def model_state(model):
    return numeric.fingerprint(dict(policy=model.policy.state_dict(), alpha=model.log_ent_coef,
        actor_optimizer=model.actor.optimizer.state_dict(), critic_optimizer=model.critic.optimizer.state_dict(),
        entropy_optimizer=model.ent_coef_optimizer.state_dict(), updates=model._n_updates))


def sample(model, observation):
    tensor = torch.as_tensor(observation[None], dtype=torch.float32)
    action, logp = model.actor.action_log_prob(tensor)
    return action, float(logp.item()), tensor


def rollout(model, env, reset_seed, action_seed, horizon, deadline):
    observation, info = env.reset(seed=reset_seed)
    initial_hash = numeric.fingerprint(observation)
    start = dict(symbol=info['symbol'], session=info['session'], bar_index=env.i,
                 episode_index=env.episode_index, observation_sha256=initial_hash)
    rewards, logps = [], []
    initial_q = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(action_seed)
        for _ in range(horizon):
            if time.monotonic() >= deadline:
                raise TimeoutError('soft-consistency diagnostic deadline exceeded')
            action, logp, tensor = sample(model, observation)
            if initial_q is None:
                initial_q = [float(q.item()) for q in model.critic(tensor, action)]
            observation, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
            rewards.append(reward)
            logps.append(logp)
            if terminated or truncated:
                break
        if not (terminated or truncated):
            raise ValueError('rollout did not reach the fixed training horizon or cutoff')
        tail_q, tail_logp = 0.0, 0.0
        if not terminated:
            action, tail_logp, tensor = sample(model, observation)
            tail_q = min(float(q.item()) for q in model.critic_target(tensor, action))
    parts = numeric.soft_target(rewards, logps, gamma=model.gamma,
        alpha=float(model.log_ent_coef.exp().item()), terminated=bool(terminated),
        bootstrap_q=tail_q, bootstrap_log_probability=tail_logp)
    return start, dict(action_seed=action_seed, horizon=len(rewards), terminated=bool(terminated),
                       truncated=bool(truncated), initial_q=initial_q,
                       min_q_minus_soft_target=min(initial_q) - parts['target'], **parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    journal = Path('.runtime/sac-soft-consistency-v1.json')
    if args.output.exists() or args.artifacts.exists() or journal.exists():
        raise ValueError('new diagnostic output/artifact/journal paths required; automatic restart disabled')
    spec_path = ROOT / 'configs/experiments/sac_soft_consistency.json'
    spec = json.loads(spec_path.read_text())
    if (spec['horizon'] != protocol()['episode_steps']
            or spec['windows'] != len(spec['window_reset_seeds'])
            or len(set(spec['window_reset_seeds'])) != spec['windows']
            or spec['windows'] < 2 or spec['trajectories_per_window'] < 1):
        raise ValueError('fixed training horizon and unique diagnostic sampling contract required')
    deadline = STARTED + spec['maximum_seconds']
    args.artifacts.mkdir(parents=True)
    journal.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status='running', protocol_sha256=sha(spec_path), records=[],
                 training_performed=False, optimizer_steps=0, colab_allocated=False)
    save(journal, state)
    env = None
    try:
        export = (ROOT / spec['source_export']).resolve()
        if (not Path(inspect.getfile(TradingEnv)).resolve().is_relative_to(export)
                or source_identity() != spec['source_package_hash']):
            raise ValueError('must execute using exact frozen source PYTHONPATH')
        source_path = ROOT / spec['source_state']
        if sha(source_path) != spec['source_state_sha256']:
            raise ValueError('fixed source study state checksum mismatch')
        source = json.loads(source_path.read_text())
        if source['status'] != 'complete' or not source['cleanup_verified']:
            raise ValueError('completed owned study with verified cleanup required')
        paths = list((Path(source['completed'][spec['source_task']]['root']) / 'runs').glob('*/metadata.json'))
        if len(paths) != 1:
            raise ValueError('one fixed source run required')
        metadata_path = paths[0]
        model_path = metadata_path.parent / 'latest.zip'
        if sha(metadata_path) != spec['metadata_sha256'] or sha(model_path) != spec['model_sha256']:
            raise ValueError('fixed saved model/metadata checksum mismatch')
        m = json.loads(metadata_path.read_text())
        if (m['algorithm'] != 'sac' or m['seed'] != spec['source_seed'] or m['smoke']
                or m['actual_steps'] != spec['source_steps'] or m['status'] != 'complete'
                or m['study_protocol_sha256'] != PROTOCOL_SHA256
                or m['dataset'] != spec['dataset'] or m['reward_config'] != spec['reward']
                or m['observation']['profile'] != spec['observation_profile']
                or m['source_hash'] != spec['source_package_hash']
                or m['git']['revision'] != spec['source_revision']
                or m['start_sampling'] != {'mode': 'legacy_uniform'}):
            raise ValueError('source policy training contract mismatch')
        dataset_path = ROOT / 'data/datasets' / spec['dataset']
        data = validate_payload(read_dataset(dataset_path))
        episodes = [e for e in data['episodes'].values() if e['split'] == 'train']
        if len(episodes) != 26 or any(b['session'] > '2020-12-31' for e in episodes for b in e['bars']):
            raise ValueError('fixed26-symbol training-only rollouts required')
        normalizer = normalizer_from(data['normalizer'])
        config = SimulationConfig(**m['simulation'])
        if asdict(config) != asdict(SimulationConfig()):
            raise ValueError('fixed simulator defaults required')
        env = TradingEnv(episodes, normalizer, data['names'], config, random_start=True,
                         episode_steps=spec['horizon'], reward_scale=spec['reward']['scale'],
                         reward_kind=spec['reward']['kind'], start_sampling='legacy_uniform',
                         observation_profile=spec['observation_profile'])
        torch.set_num_threads(1)
        model = SAC.load(model_path, device='cpu')
        if (model.gamma != 0.99 or model._vec_normalize_env is not None
                or model.observation_space != env.observation_space or model.action_space != env.action_space):
            raise ValueError('saved SAC value/reward/observation contract mismatch')
        model.policy.set_training_mode(False)
        model.critic_target.eval()
        before = model_state(model)
        rng_before = numeric.fingerprint((torch.get_rng_state(), np.random.get_state()))
        norm_before = numeric.fingerprint(asdict(normalizer))
        protected = {str(p): sha(p) for p in (source_path, metadata_path, model_path,
                                             dataset_path / 'dataset.json', spec_path, HELPER, Path(__file__))}
        with torch.inference_mode():
            for window, reset_seed in enumerate(spec['window_reset_seeds']):
                rows, starts = [], []
                for repeat in range(spec['trajectories_per_window']):
                    action_seed = spec['trajectory_seed_base'] + window * spec['trajectories_per_window'] + repeat
                    start, row = rollout(model, env, reset_seed, action_seed, spec['horizon'], deadline)
                    rows.append(row)
                    starts.append(start)
                if any(start != starts[0] for start in starts):
                    raise ValueError('paired action-noise repetitions changed starting state')
                record = dict(window=window, reset_seed=reset_seed, start=starts[0], trajectories=rows,
                    mean_min_q_minus_soft_target=float(np.mean([r['min_q_minus_soft_target'] for r in rows])))
                write_once(args.artifacts / f'window-{window:03}.json', (json.dumps(record, indent=2) + '\n').encode())
                state['records'].append(record)
                state['elapsed_seconds'] = time.monotonic() - STARTED
                save(journal, state)
                if (window + 1) % 8 == 0:
                    print(f"Frozen training windows {window + 1}/{spec['windows']}", flush=True)
        if (model_state(model) != before or numeric.fingerprint(asdict(normalizer)) != norm_before
                or numeric.fingerprint((torch.get_rng_state(), np.random.get_state())) != rng_before
                or any(sha(p) != value for p, value in protected.items())
                or source_identity() != spec['source_package_hash']):
            raise ValueError('diagnostic changed model/optimizer/RNG/normalizer/source state')
        summary = numeric.summarize_windows(state['records'])
        rows = [row for record in state['records'] for row in record['trajectories']]
        full = sum(row['horizon'] == spec['horizon'] for row in rows) / len(rows)
        flag = (summary['descriptive_lower_two_se'] >= spec['evidence_flag']['minimum_descriptive_lower_two_se_gap']
                and full >= spec['evidence_flag']['minimum_full_horizon_trajectory_fraction'])
        result = dict(schema='saved-sac-soft-consistency-result-v1', **summary,
            trajectories=len(rows), full_horizon_trajectory_fraction=full,
            evidence_flag=flag, alpha=float(model.log_ent_coef.detach().exp().item()),
            mean_target=float(np.mean([r['target'] for r in rows])),
            mean_initial_min_q=float(np.mean([min(r['initial_q']) for r in rows])),
            mean_bootstrap_discount=float(np.mean([r['bootstrap_discount'] for r in rows])),
            protocol_sha256=sha(spec_path), protected_source_sha256=protected,
            source_revision=spec['source_revision'], source_package_hash=spec['source_package_hash'],
            model_optimizer_updates_fingerprint=before, normalization_fingerprint=norm_before,
            weights_optimizers_updates_normalization_rng_unchanged=True,
            source_files_unchanged=True, optimizer_steps=0, training_performed=False,
            colab_allocated=False, validation_or_test_scored=False, checkpoint_selected=False,
            objective_achieved=False, elapsed_seconds=time.monotonic() - STARTED,
            interpretation='Fixed flat-account training-reset multistep soft-target consistency with learned target-critic bootstrap. Conditional sampling uncertainty only, not unbiased true-Q calibration, future-market confidence, wrong-gradient proof or trading performance.')
        write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
        state.update(status='complete', summary_sha256=sha(args.output))
        print(json.dumps({k: result[k] for k in ('mean_min_q_minus_soft_target', 'descriptive_lower_two_se',
              'full_horizon_trajectory_fraction', 'evidence_flag')}, indent=2), flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                     error_type=type(error).__name__)
        raise
    finally:
        if env is not None:
            env.close()
            state['environment_closed'] = True
        state['elapsed_seconds'] = time.monotonic() - STARTED
        save(journal, state)


if __name__ == '__main__':
    def terminate(signum, frame):
        raise KeyboardInterrupt('soft-consistency diagnostic interrupted')
    signal.signal(signal.SIGTERM, terminate)
    main()
