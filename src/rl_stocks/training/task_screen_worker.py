"""Fixed seed11 task comparison cell; remote registration remains profile-gated."""
import json
from pathlib import Path
import re
import time

import torch
from stable_baselines3 import SAC

from rl_stocks.core.contracts import digest
from rl_stocks.core.provenance import repository_state, source_identity
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.evaluation.task_conditioning import evaluate_task_training
from rl_stocks.inference.task_bundle import freeze_task
from rl_stocks.reporting.html import report
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.risk.exposure import execution_contract
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
from rl_stocks.training.task_selection import select_training_checkpoint
from rl_stocks.training.task_smoke_worker import verify_restoration


def execute_screen_colab(dataset, type_manifest, output, *, task_role):
    require_colab_cpu()
    role = check_role(task_role)
    spec = protocol()
    if digest(dataset) != spec['dataset'] or digest(dataset['normalizer']) != spec['normalizer_sha256']:
        raise ValueError('fixed research dataset/preprocessing required')
    evidence_sha = validate_type_manifest(dataset, type_manifest)
    git = repository_state(Path.cwd())
    if git['dirty'] != '' or not re.fullmatch(r'[a-f0-9]{40}', git['revision'] or ''):
        raise ValueError('clean committed screen source required')
    output = Path(output)
    if output.exists() or output.parent.name != 'artifacts':
        raise ValueError('fresh owned screen artifacts required')
    output.mkdir(parents=True)
    metadata = dict(algorithm='sac', task_role=role, seed=11, protocol_sha256=PROTOCOL_SHA256,
        dataset=spec['dataset'], type_manifest_sha256=evidence_sha, repository=git,
        source=source_identity(), development_evaluated=False, test_evaluated=False,
        performance_promotion=False, automatic_retry=False)
    env = model = audit = None
    history = []
    started = time.monotonic()
    stocks = set(dataset['manifest']['evaluation_symbols']) - {'SPY', 'QQQ', 'IWM', 'DIA'}
    try:
        with preserved_rng():
            torch.set_num_threads(1)
            env = build_task_environment(dataset, type_manifest, task_role=role, smoke=False)
            model = create_task_sac(env, task_role=role, seed=11, smoke=False)
            audit = TaskOptimizerAudit(model, maximum=99000)
            model.save(output / 'initial.zip')
            _atomic_json(output / 'input.json', metadata)
            export_artifacts(output.parent)

            def checkpoint(current, steps):
                audit.verify(current, steps, 1000)
                current.save(output / f'checkpoint-{steps}.zip')
                export_artifacts(output.parent)
                before = fingerprint((current.policy.state_dict(),
                    {k: v.state_dict() for k, v in audit.optimizers.items()}, current.log_ent_coef))
                with preserved_rng(), torch.no_grad():
                    results = evaluate_task_training(current, dataset, type_manifest)
                if before != fingerprint((current.policy.state_dict(),
                        {k: v.state_dict() for k, v in audit.optimizers.items()}, current.log_ent_coef)):
                    raise ValueError('training checkpoint inference mutated learned state')
                metrics = group_metrics(results, stocks)
                point = curve_record(results, history, steps, 'training-window',
                                     time.monotonic() - started, {'optimizer_steps': dict(audit.counts)})
                point['metrics'] = metrics
                point['holding_path_sha256'] = digest({k: v['baselines']['buy_hold'] for k, v in results.items()})
                history.append(point)
                _atomic_json(output / f'training-{steps}.json', results)
                _atomic_json(output / 'curve.json', history)
                render_curve(history, output / 'curve.html')
                export_artifacts(output.parent)

            with post_update_checkpoints(model, every=10000, checkpoint=checkpoint):
                model.learn(total_timesteps=100000)
            audit.verify(model, 100000, 1000)
            if role == 'tagged_routed':
                for family in ('actor', 'critic'):
                    for group in ('stock', 'etf'):
                        if not any(name.startswith(family + '/') and '.' + group + '.' in name
                                   for name in audit.active):
                            raise ValueError('each routed head requires actual nonzero gradients')
            selection = select_training_checkpoint(history)
            sampling = env.unwrapped.sampling_statistics()
            expected = {r['symbol'] for r in type_manifest['records']}
            visits = sampling['year_counts']['per_symbol']
            if set(visits) != expected or any(v['starts'] < 1 or v['transitions'] < 1 for v in visits.values()):
                raise ValueError('all111 training symbols must actually be visited')
            model.save_replay_buffer(output / 'replay.pkl')
            restored = SAC.load(output / 'checkpoint-100000.zip', device='cpu')
            restored.load_replay_buffer(output / 'replay.pkl')
            verify_restoration(model, restored)
            for phase, point in [('selected', selection['selected']), ('final', selection['final'])]:
                steps = point['steps']
                results = json.loads((output / f'training-{steps}.json').read_text())
                report(results, output / f'{phase}-training.html', f'{phase}: training-only task screen')
                freeze_task(output / f'checkpoint-{steps}.zip', dataset, task_role=role,
                    type_manifest=type_manifest, metadata={**metadata, 'phase': phase, 'steps': steps,
                        'observation': env.unwrapped.observation_spec,
                        'action_execution': execution_contract('unrestricted', 'unit_magnitude')},
                    output=output / f'frozen-{phase}')
            _atomic_json(output / 'training-cell.json', dict(steps=100000, period='training-window',
                phase='final', seed=11, protocol_sha256=PROTOCOL_SHA256,
                holding_path_sha256=history[-1]['holding_path_sha256'], metrics=history[-1]['metrics']))
            _atomic_json(output / 'run.json', {**metadata, 'status': 'complete',
                'actual_transitions': model.num_timesteps, 'optimizer_steps': dict(audit.counts),
                'selection': selection, 'sampling': sampling, 'active_gradients': audit.active,
                'execution': env.unwrapped.execution_statistics(),
                'optimizer_restore_exact': True, 'replay_restore_exact': True,
                'inclusive_elapsed_seconds': time.monotonic() - started})
    except BaseException as error:
        _atomic_json(output / 'failure.json', {**metadata, 'error_type': type(error).__name__,
            'actual_transitions': None if model is None else model.num_timesteps})
        raise
    finally:
        try:
            if audit is not None:
                audit.close()
            if env is not None:
                env.close()
        finally:
            export_artifacts(output.parent)
