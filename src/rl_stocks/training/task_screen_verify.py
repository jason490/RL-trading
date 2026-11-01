"""Independent saved100k cell replay; inference only, with no local fitting."""
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.evaluation.task_conditioning import evaluate_task_training
from rl_stocks.inference.task_bundle import FrozenTaskPolicy
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.risk.exposure import filter_exposure
from rl_stocks.training.curves import curve_record
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.task_conditioning import PROTOCOL_SHA256, protocol, validate_type_manifest
from rl_stocks.training.task_optimizer import TaskOptimizerAudit
from rl_stocks.training.task_selection import select_training_checkpoint


def verify_cell(folder, dataset, evidence, *, role, source, git):
    folder = Path(folder)
    spec = protocol()
    if digest(dataset) != spec['dataset'] or validate_type_manifest(dataset, evidence) != '7c4a078fcec05be000d720e2d9625c6cd5d7c03b5d1d6502e46357f526ba1590':
        raise ValueError('fixed independently verified task inputs required')
    run = json.loads((folder / 'run.json').read_text())
    if (run.get('status') != 'complete' or run.get('task_role') != role or run.get('seed') != 11
            or run.get('protocol_sha256') != PROTOCOL_SHA256 or run.get('source') != source
            or run.get('repository') != git or run.get('actual_transitions') != 100000
            or run.get('development_evaluated') is not False or run.get('test_evaluated') is not False
            or run.get('dataset') != spec['dataset']
            or run.get('optimizer_steps') != dict.fromkeys(('actor', 'critic', 'entropy'), 99000)):
        raise ValueError('complete bound seed11 training-only screen required')
    history = json.loads((folder / 'curve.json').read_text())
    selection = select_training_checkpoint(history)
    if digest(selection) != digest(run['selection']):
        raise ValueError('saved training-only selection mismatch')
    stocks = set(dataset['manifest']['evaluation_symbols']) - {'SPY', 'QQQ', 'IWM', 'DIA'}
    checked = []
    final = None
    initial_state = None
    with preserved_rng(), torch.no_grad():
        torch.set_num_threads(1)
        for steps in [0] + spec['selection']['checkpoints']:
            name = 'initial.zip' if steps == 0 else f'checkpoint-{steps}.zip'
            model = SAC.load(folder / name, device='cpu')
            if (model.policy.task_role != role or model.batch_size != 256 or model.learning_starts != 1000
                    or model.observation_space.shape != (22,)
                    or any(not torch.isfinite(v).all() for v in model.policy.state_dict().values())):
                raise ValueError('finite physical research task checkpoint required')
            audit = TaskOptimizerAudit(model, maximum=99000)
            try:
                audit.counts = dict.fromkeys(audit.counts, max(0, steps - 1000))
                audit.verify(model, steps, 1000)
            finally:
                audit.close()
            state = fingerprint(model.policy.state_dict())
            if steps == 0:
                initial_state = state
                continue
            results = evaluate_task_training(model, dataset, evidence)
            retained = json.loads((folder / f'training-{steps}.json').read_text())
            if digest(results) != digest(retained):
                raise ValueError('saved checkpoint training replay differs from retained full results')
            saved = history[len(checked)]
            point = curve_record(results, checked, steps, 'training-window', saved['elapsed_seconds'],
                                 {'optimizer_steps': dict.fromkeys(('actor', 'critic', 'entropy'), steps - 1000)})
            point['metrics'] = group_metrics(results, stocks)
            point['holding_path_sha256'] = digest({k: v['baselines']['buy_hold'] for k, v in results.items()})
            if digest(point) != digest(saved) or fingerprint(model.policy.state_dict()) != state:
                raise ValueError('independently reconstructed training curve/state mismatch')
            checked.append(point)
            final = model
        if fingerprint(final.policy.state_dict()) == initial_state:
            raise ValueError('actual learned final state required')
        final.load_replay_buffer(folder / 'replay.pkl')
        replay = final.replay_buffer
        if replay.size() != 100000 or not replay.full or replay.pos != 0:
            raise ValueError('complete100k replay required')
        for name in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
            if not np.isfinite(getattr(replay, name)).all():
                raise ValueError('finite full replay required')
        bundles = {}
        for phase, point in [('selected', selection['selected']), ('final', selection['final'])]:
            paths = list((folder / f'frozen-{phase}').iterdir())
            if len(paths) != 1:
                raise ValueError('one selected/final frozen task bundle required')
            frozen = FrozenTaskPolicy(paths[0], feature_schema=dataset['schema'],
                feed=dataset['manifest']['provenance']['capabilities']['feed'])
            model = SAC.load(folder / f"checkpoint-{point['steps']}.zip", device='cpu')
            if fingerprint(frozen._model.policy.state_dict()) != fingerprint(model.policy.state_dict()):
                raise ValueError('selected/final frozen checkpoint mismatch')
            for record in evidence['records']:
                x = np.linspace(-1, 1, 21, dtype=np.float32)
                tag = 0 if role == 'masked_shared' else (1 if record['kind'] == 'etf' else -1)
                expected = filter_exposure(model.predict(np.append(x, tag), deterministic=True)[0], 'unit_magnitude')
                if not np.array_equal(expected, frozen.predict(x, symbol=record['symbol'], session=record['known_from'])):
                    raise ValueError('independent learned frozen parity mismatch')
            bundles[phase] = paths[0].name
    cell = dict(steps=100000, period='training-window', phase='final', seed=11,
        protocol_sha256=PROTOCOL_SHA256, holding_path_sha256=checked[-1]['holding_path_sha256'],
        metrics=checked[-1]['metrics'])
    if digest(cell) != digest(json.loads((folder / 'training-cell.json').read_text())):
        raise ValueError('independent final fitted gate cell mismatch')
    expected_symbols = {r['symbol'] for r in evidence['records']}
    visits = run['sampling']['year_counts']['per_symbol']
    if set(visits) != expected_symbols or any(v['starts'] < 1 or v['transitions'] < 1 for v in visits.values()):
        raise ValueError('actual complete111-symbol training coverage required')
    return dict(task_role=role, replayed_checkpoints=len(checked), selected_step=selection['selected']['steps'],
                frozen_bundles=bundles, training_cell=cell, performance_promotion=False)
