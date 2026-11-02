"""Consistency checks for immutable Colab controls; no native prediction or fit.

This verifies retained training artifacts, not original native replay acceptance.
Reference selected/final evaluation is a separate declared phase.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
from stable_baselines3 import SAC
import torch

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.canonical_diagnostic import inference_only
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.evaluation.direction import directional_attribution
from rl_stocks.evaluation.metrics import metrics as economic_metrics
from rl_stocks.evaluation.runner import baselines
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.training_fit import training_window
from rl_stocks.inference.canonical_bundle import _checked_files, _metadata
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.rl.sac.task_policy import TaskSACPolicy
from rl_stocks.runtime.colab.canonical_market_probe import file_sha
from rl_stocks.training.curves import curve_record
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.task_conditioning import protocol as base_protocol, validate_type_manifest
from rl_stocks.training.task_optimizer import TaskOptimizerAudit
from rl_stocks.training.task_selection import select_training_checkpoint

PROTOCOL_SHA256 = '8c8f7bb00b1ba4c38e01d6dbbee401033d2b4b4d895ad9dbab2cbaf0b37e79a8'


def protocol(root):
    spec = json.loads((Path(root) / 'configs/experiments/task_reference_execution.json').read_text())
    if digest(spec) != PROTOCOL_SHA256 or digest(base_protocol()) != spec['base_protocol_sha256']:
        raise ValueError('fixed reference execution and unchanged research gates required')
    engineering = Path(root) / 'results/reference-inference-2026-11-02/market-cross-machine-acceptance.json'
    if file_sha(engineering) != spec['engineering_acceptance_sha256']:
        raise ValueError('fixed complete cross-machine engineering acceptance required')
    return spec


def bind_archive(archive, folder, *, expected_sha):
    """Bind every extracted control file to its original hash-pinned archive."""
    archive, folder = Path(archive), Path(folder)
    if file_sha(archive) != expected_sha:
        raise ValueError('immutable original control archive checksum required')
    bound, completion = {}, None
    prefix = 'artifacts/task-screen/'
    names, total = set(), 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if (member.name in names or name.is_absolute() or '..' in name.parts
                    or '\\' in member.name or not (member.isfile() or member.isdir())):
                raise ValueError('ordinary unique control archive members required')
            names.add(member.name)
            total += member.size
            if len(names) > 64 or total > 1024*1024*1024:
                raise ValueError('bounded control archive required')
            if not member.isfile():
                continue
            if member.name == 'artifacts/completion.json':
                completion = json.load(stream.extractfile(member))
            elif member.name.startswith(prefix):
                relative = member.name[len(prefix):]
                path = folder / relative
                if path.is_symlink() or not path.is_file():
                    raise ValueError('original extracted control file missing or substituted')
                expected = hashlib.file_digest(stream.extractfile(member), 'sha256').hexdigest()
                if file_sha(path) != expected:
                    raise ValueError('extracted control differs from immutable archive')
                bound[relative] = expected
            elif member.name != 'artifacts/bootstrap-log.txt':
                raise ValueError('unexpected control archive file')
    if completion is None or not bound or set(bound) != {
            str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}:
        raise ValueError('complete original control file set required')
    return bound, completion


def check_model(model, *, role, steps):
    spec = base_protocol()
    if type(steps) is not int or steps not in [0] + spec['selection']['checkpoints']:
        raise ValueError('declared native checkpoint step required')
    if (type(model.policy) is not TaskSACPolicy or model.policy.task_role != role
            or any(member.task_role != role for member in
                   (model.actor, model.critic, model.critic_target))
            or model.observation_space.shape != (22,) or model.action_space.shape != (4,)
            or not np.array_equal(model.action_space.low, -np.ones(4))
            or not np.array_equal(model.action_space.high, np.ones(4))
            or model.seed != 11 or model.policy.net_arch != [64, 64]
            or any(getattr(model, name) != spec['parameters'][name] for name in
                   ('batch_size', 'buffer_size', 'learning_starts', 'gradient_steps',
                    'tau', 'gamma', 'ent_coef', 'learning_rate'))
            or model.train_freq.frequency != 1 or model.train_freq.unit.value != 'step'
            or any(t.dtype != torch.float32 or not torch.isfinite(t).all()
                   for t in model.policy.state_dict().values())
            or not torch.isfinite(model.log_ent_coef).all()):
        raise ValueError('finite fixed native research checkpoint contract required')
    audit = TaskOptimizerAudit(model, maximum=99000)
    try:
        audit.counts = dict.fromkeys(audit.counts, max(0, steps-1000))
        audit.verify(model, steps, 1000)
        return {'steps': steps, 'updates': max(0, steps-1000),
                'policy_sha256': fingerprint(model.policy.state_dict()),
                'optimizer_state_sha256': fingerprint({k: opt.state_dict()
                                                       for k, opt in audit.optimizers.items()}),
                'raw_alpha_sha256': fingerprint(model.log_ent_coef)}
    finally:
        audit.close()



def check_economic_report(result, episode, baseline):
    if (digest(result['metrics']) != digest(economic_metrics(result['rows'], result['events']))
            or digest(result['baselines']) != digest(baseline)
            or digest(result['directional']) != digest(directional_attribution(
                episode, result['rows'], result['events']))):
        raise ValueError('raw native economic report is inconsistent')


def verify_control(root, dataset, *, role):
    root = Path(root)
    spec = protocol(root)
    origin = spec['controls'][role]
    base = base_protocol()
    if digest(dataset) != base['dataset'] or digest(dataset['normalizer']) != base['normalizer_sha256']:
        raise ValueError('fixed verified control dataset/preprocessing required')
    view = training_window(dataset, original_panel_only=True)
    folder = root / 'artifacts/colab' / origin['session'] / 'retrieved/artifacts/task-screen'
    archive = root / 'artifacts/colab' / origin['session'] / 'artifacts.tar.gz'
    bound, completion = bind_archive(archive, folder, expected_sha=origin['archive_sha256'])
    run = json.loads((folder / 'run.json').read_text())
    inputs = json.loads((folder / 'input.json').read_text())
    if (run['status'] != 'complete' or run['task_role'] != role or run['seed'] != 11
            or run['repository'] != {'dirty': '', 'revision': origin['source_revision']}
            or run['source'] != origin['source_hash'] or run['dataset'] != base['dataset']
            or run['protocol_sha256'] != spec['base_protocol_sha256']
            or completion['source_git'] != run['repository']
            or completion['source_hash'] != run['source'] or completion['dataset'] != run['dataset']
            or completion['gpu'] is not None or completion['python'][:2] != [3, 13]
            or run['actual_transitions'] != 100000
            or run['optimizer_steps'] != dict.fromkeys(('actor', 'critic', 'entropy'), 99000)
            or run['development_evaluated'] is not False or run['test_evaluated'] is not False
            or any(run.get(k) != v for k, v in inputs.items())):
        raise ValueError('original complete native training metadata required')
    history = json.loads((folder / 'curve.json').read_text())
    selection = select_training_checkpoint(history)
    if digest(selection) != digest(run['selection']):
        raise ValueError('retained native training selection is inconsistent')
    stocks = set(dataset['manifest']['evaluation_symbols']) - {'SPY', 'QQQ', 'IWM', 'DIA'}
    holding = {key:baselines(episode, SimulationConfig()) for key,episode in view['episodes'].items()}
    checked, reconstructed, frozen_bundles = [], [], {}
    try:
        with inference_only(), preserved_rng(), torch.no_grad():
            for steps in [0] + base['selection']['checkpoints']:
                path = folder / ('initial.zip' if steps == 0 else f'checkpoint-{steps}.zip')
                model = SAC.load(path, device='cpu')
                checked.append(check_model(model, role=role, steps=steps))
                if steps == 0:
                    continue
                results = json.loads((folder / f'training-{steps}.json').read_text())
                if set(results) != set(view['episodes']):
                    raise ValueError('exact original26 native training reports required')
                for key, result in results.items():
                    if (result['split'] != 'train' or result['symbol'] != view['episodes'][key]['symbol']
                            or [row['session'] for row in result['rows']] !=
                            [bar['session'] for bar in view['episodes'][key]['bars'][1:]]):
                        raise ValueError('complete chronological native training paths required')
                    check_economic_report(result, view['episodes'][key], holding[key])
                saved = history[len(reconstructed)]
                point = curve_record(results, reconstructed, steps, 'training-window', saved['elapsed_seconds'],
                                     {'optimizer_steps': dict.fromkeys(('actor', 'critic', 'entropy'), steps-1000)})
                point['metrics'] = group_metrics(results, stocks)
                point['holding_path_sha256'] = digest({k:r['baselines']['buy_hold'] for k,r in results.items()})
                if digest(point) != digest(saved):
                    raise ValueError('raw native reports do not reproduce saved selection curve')
                reconstructed.append(point)
            if checked[0]['policy_sha256'] == checked[-1]['policy_sha256']:
                raise ValueError('retained trained weights must differ from initialization')
            model.load_replay_buffer(folder / 'replay.pkl')
            replay = model.replay_buffer
            if replay.size() != 100000 or not replay.full or replay.pos != 0:
                raise ValueError('complete original replay buffer required')
            shapes = {'observations':(100000,1,22), 'next_observations':(100000,1,22),
                      'actions':(100000,1,4), 'rewards':(100000,1), 'dones':(100000,1), 'timeouts':(100000,1)}
            for name in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
                if getattr(replay, name).shape != shapes[name] or not np.isfinite(getattr(replay, name)).all():
                    raise ValueError('finite complete original replay arrays required')
            for name in ('observations', 'next_observations'):
                tags = getattr(replay, name)[..., -1]
                if not np.isin(tags, [0] if role == 'masked_shared' else [-1, 1]).all():
                    raise ValueError('original replay task tags mismatch')
            replay_sha = fingerprint({name:getattr(replay, name) for name in
                ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')})
            for phase in ('selected', 'final'):
                paths = list((folder / f'frozen-{phase}').iterdir())
                if len(paths) != 1:
                    raise ValueError('one original frozen bundle per phase required')
                bundle = paths[0]
                _checked_files(bundle, ('model.zip', 'manifest.json', 'requirements.lock'))
                manifest = json.loads((bundle / 'manifest.json').read_text())
                steps = selection[phase]['steps']
                if (digest(manifest) != bundle.name or manifest['model_sha256'] != file_sha(bundle/'model.zip')
                        or file_sha(bundle/'model.zip') != file_sha(folder/f'checkpoint-{steps}.zip')
                        or manifest['metadata']['steps'] != steps or manifest['metadata']['phase'] != phase):
                    raise ValueError('original frozen weights/selection identity mismatch')
                actual_role, _, _, execution = _metadata(manifest, feature_schema=dataset['schema'],
                    feed=dataset['manifest']['provenance']['capabilities']['feed'])
                evidence = manifest['metadata']['task_contract']['type_manifest']
                if (digest(manifest['normalizer']) != digest(dataset['normalizer'])
                        or execution != {'version':1, 'training_exposure_mode':'unrestricted',
                                         'evaluation_exposure_mode':'unit_magnitude'}
                        or actual_role != role or validate_type_manifest(dataset, evidence) != run['type_manifest_sha256']):
                    raise ValueError('original frozen type/preprocessing provenance mismatch')
                frozen_bundles[phase] = bundle.name
        visits = run['sampling']['year_counts']['per_symbol']
        if (set(visits) != {e['symbol'] for e in dataset['episodes'].values() if e['split']=='train'}
                or any(v['starts'] < 1 or v['transitions'] < 1 for v in visits.values())):
            raise ValueError('actual complete111-symbol training coverage required')
        final_cell = {'steps':100000, 'period':'training-window', 'phase':'final', 'seed':11,
            'protocol_sha256':spec['base_protocol_sha256'],
            'holding_path_sha256':reconstructed[-1]['holding_path_sha256'], 'metrics':reconstructed[-1]['metrics']}
        if digest(final_cell) != digest(json.loads((folder/'training-cell.json').read_text())):
            raise ValueError('retained native final cell is inconsistent')
        return {'role':role, 'session':origin['session'], 'archive_sha256':origin['archive_sha256'],
            'bound_files':len(bound), 'bound_inventory_sha256':digest(bound), 'checkpoints':checked,
            'raw_training_curves_reconstructed':len(reconstructed), 'selected_step':selection['selected']['steps'],
            'frozen_bundles':frozen_bundles, 'replay_sha256':replay_sha, 'visited_training_symbols':len(visits),
            'native_backend_record_present':'execution_backend' in run,
            'native_inference_replayed':False, 'native_acceptance_changed':False,
            'physical_consistency_only':True, 'performance_promotion':False, 'training_performed':False}
    finally:
        if (file_sha(archive) != origin['archive_sha256']
                or any(file_sha(folder/name) != sha for name,sha in bound.items())):
            raise ValueError('original control files mutated during physical verification')
