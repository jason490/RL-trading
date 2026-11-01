"""Fixed1024-step task SAC smoke; optimization is allowed only in owned Colab CPU."""
from copy import deepcopy
from pathlib import Path
import re
import time

import numpy as np
import torch
from stable_baselines3 import SAC

from rl_stocks.core.contracts import digest
from rl_stocks.core.provenance import repository_state, source_identity
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.evaluation.task_conditioning import evaluate_task_training
from rl_stocks.inference.task_bundle import FrozenTaskPolicy, freeze_task
from rl_stocks.reporting.html import report
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.risk.exposure import execution_contract, filter_exposure
from rl_stocks.rl.sac.task_factory import create_task_sac
from rl_stocks.rl.sac.task_policy import check_role
from rl_stocks.training.checkpoint_export import export_artifacts
from rl_stocks.training.curves import curve_record, render_curve
from rl_stocks.training.native_post_updates import post_update_checkpoints
from rl_stocks.training.recurrent_loop import _atomic_json
from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.task_conditioning import PROTOCOL_SHA256, protocol, validate_type_manifest
from rl_stocks.training.task_environment import build_task_environment
from rl_stocks.training.task_optimizer import TaskOptimizerAudit


def verify_restoration(model, restored):
    optimizers = {'actor': model.actor.optimizer, 'critic': model.critic.optimizer,
                  'entropy': model.ent_coef_optimizer}
    restored_optimizers = {'actor': restored.actor.optimizer,
                           'critic': restored.critic.optimizer,
                           'entropy': restored.ent_coef_optimizer}
    if (fingerprint(model.policy.state_dict()) != fingerprint(restored.policy.state_dict())
            or not torch.equal(model.log_ent_coef, restored.log_ent_coef)
            or any(fingerprint(opt.state_dict()) !=
                   fingerprint(restored_optimizers[name].state_dict())
                   for name, opt in optimizers.items())
            or restored._n_updates != model._n_updates
            or restored.num_timesteps != model.num_timesteps):
        raise ValueError('actual learned task optimizer restoration failed')
    for name in ('observations', 'next_observations', 'actions', 'rewards',
                 'dones', 'timeouts'):
        if not np.array_equal(getattr(model.replay_buffer, name),
                              getattr(restored.replay_buffer, name)):
            raise ValueError('actual task replay restoration failed')
    if (model.replay_buffer.pos != restored.replay_buffer.pos
            or model.replay_buffer.full != restored.replay_buffer.full):
        raise ValueError('actual task replay cursor restoration failed')


def execute_smoke_colab(dataset, type_manifest, output, *, task_role, seed=11):
    require_colab_cpu()  # Before output, model construction, RNG or optimizer mutation.
    spec = protocol()
    role = check_role(task_role)
    if type(seed) is not int or seed != spec['screen_seed']:
        raise ValueError('fixed smoke seed11 required')
    if (digest(dataset) != spec['dataset']
            or digest(dataset['normalizer']) != spec['normalizer_sha256']):
        raise ValueError('fixed task dataset/normalizer identity required')
    evidence_sha = validate_type_manifest(dataset, type_manifest)
    output = Path(output)
    if output.exists() or output.parent.name != 'artifacts':
        raise ValueError('fresh owned artifacts smoke output required')
    started = time.monotonic()
    environment = audit = model = None
    history = []
    metadata = {'algorithm': 'sac', 'task_role': role, 'seed': seed,
                'protocol_sha256': PROTOCOL_SHA256, 'dataset': spec['dataset'],
                'type_manifest_sha256': evidence_sha, 'source': source_identity(),
                'repository': repository_state(Path.cwd()), 'smoke_only': True,
                'development_evaluated': False, 'test_evaluated': False,
                'objective_achieved': False, 'automatic_retry': False}
    if (metadata['repository']['dirty'] != ''
            or not re.fullmatch(r'[a-f0-9]{40}', metadata['repository']['revision'] or '')):
        raise ValueError('committed clean source revision required')
    output.mkdir(parents=True)
    try:
        with preserved_rng():
            torch.set_num_threads(1)
            environment = build_task_environment(dataset, type_manifest, task_role=role, smoke=True)
            model = create_task_sac(environment, task_role=role, seed=seed, smoke=True)
            from rl_stocks.training.task_backend import task_execution_backend
            metadata['execution_backend'] = task_execution_backend(model.device)
            audit = TaskOptimizerAudit(model, maximum=960)
            initial = deepcopy(model.policy.state_dict())
            model.save(output / 'initial.zip')
            _atomic_json(output / 'input.json', metadata)
            export_artifacts(output.parent)
            stocks = set(dataset['manifest']['evaluation_symbols']) - {'SPY', 'QQQ', 'IWM', 'DIA'}
            if len(stocks) != 22:
                raise ValueError('original26 fitted evaluation panel required')

            def checkpoint(current, steps):
                audit.verify(current, steps, 64)
                current.save(output / f'checkpoint-{steps}.zip')
                export_artifacts(output.parent)  # Weights survive a subsequent replay failure.
                before = fingerprint((current.policy.state_dict(),
                                      {k: v.state_dict() for k, v in audit.optimizers.items()},
                                      current.log_ent_coef))
                with preserved_rng(), torch.no_grad():
                    results = evaluate_task_training(current, dataset, type_manifest)
                after = fingerprint((current.policy.state_dict(),
                                     {k: v.state_dict() for k, v in audit.optimizers.items()},
                                     current.log_ent_coef))
                if before != after:
                    raise ValueError('task evaluation mutated learned state')
                point = curve_record(results, history, steps, 'training-window',
                                     time.monotonic() - started,
                                     {'optimizer_steps': dict(audit.counts)})
                point['groups'] = group_metrics(results, stocks)
                point['holding_path_sha256'] = digest(
                    {k: v['baselines']['buy_hold'] for k, v in results.items()})
                history.append(point)
                _atomic_json(output / 'curve.json', history)
                render_curve(history, output / 'curve.html')
                if steps == 1024:
                    report(results, output / 'final-training.html', 'Task smoke; training-only, no performance gate')
                export_artifacts(output.parent)

            with post_update_checkpoints(model, every=512, checkpoint=checkpoint):
                model.learn(total_timesteps=1024)
            audit.verify(model, 1024, 64)
            if [point['steps'] for point in history] != [512, 1024]:
                raise ValueError('two post-update smoke checkpoints required')
            if not any(not torch.equal(value, model.policy.state_dict()[key])
                       for key, value in initial.items()):
                raise ValueError('smoke must change actual learned weights')
            if role == 'tagged_routed':
                for family in ('actor', 'critic'):
                    for group in ('stock', 'etf'):
                        if not any(name.startswith(family + '/') and '.' + group + '.' in name
                                   for name in audit.active):
                            raise ValueError('each routed head group requires actual nonzero gradients')
            model.save_replay_buffer(output / 'replay.pkl')
            restored = SAC.load(output / 'checkpoint-1024.zip')
            restored.load_replay_buffer(output / 'replay.pkl')
            verify_restoration(model, restored)
            bundle_metadata = {**metadata, 'observation': environment.unwrapped.observation_spec,
                               'action_execution': execution_contract('unrestricted', 'unit_magnitude')}
            bundle_path = freeze_task(output / 'checkpoint-1024.zip', dataset, task_role=role,
                                      type_manifest=type_manifest, metadata=bundle_metadata,
                                      output=output / 'frozen')
            frozen = FrozenTaskPolicy(bundle_path, feature_schema=dataset['schema'],
                                      feed=dataset['manifest']['provenance']['capabilities']['feed'])
            parity = []
            for record in type_manifest['records']:
                x = np.zeros(21, dtype=np.float32)
                tag = 0 if role == 'masked_shared' else (1 if record['kind'] == 'etf' else -1)
                expected = filter_exposure(model.predict(np.append(x, tag), deterministic=True)[0],
                                           'unit_magnitude')
                actual = frozen.predict(x, symbol=record['symbol'], session=record['known_from'])
                if not np.array_equal(expected, actual):
                    raise ValueError('actual learned frozen task parity failed')
                parity.append({'symbol': record['symbol'], 'exact': True})
            _atomic_json(output / 'run.json', {**metadata, 'status': 'complete',
                'actual_transitions': model.num_timesteps, 'optimizer_steps': dict(audit.counts),
                'optimizer_restore_exact': True, 'replay_restore_exact': True,
                'active_gradients': audit.active, 'sampling': environment.unwrapped.sampling_statistics(),
                'execution': environment.unwrapped.execution_statistics(), 'frozen_parity': parity,
                'inclusive_elapsed_seconds': time.monotonic() - started})
            return history
    except BaseException as error:
        _atomic_json(output / 'failure.json', {**metadata, 'error_type': type(error).__name__,
            'actual_transitions': None if model is None else model.num_timesteps,
            'optimizer_steps': None if audit is None else dict(audit.counts)})
        raise
    finally:
        try:
            if audit is not None:
                audit.close()
            if environment is not None:
                environment.close()
        finally:
            export_artifacts(output.parent)
