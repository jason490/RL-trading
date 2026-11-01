"""Bounded task-smoke download validation and physical saved-state inspection."""
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

import numpy as np
import torch
from stable_baselines3 import SAC

from rl_stocks.core.contracts import canonical
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.inference.task_bundle import FrozenTaskPolicy
from rl_stocks.risk.exposure import filter_exposure
from rl_stocks.runtime.colab.jobs import validate_plan
from rl_stocks.training.task_optimizer import TaskOptimizerAudit


FILES = {'initial.zip', 'checkpoint-512.zip', 'checkpoint-1024.zip', 'replay.pkl',
         'input.json', 'run.json', 'curve.json', 'curve.csv', 'curve.html',
         'final-training.html', 'final-training.json'}


def verify_directory(folder, plan):
    folder = Path(folder)
    run = json.loads((folder / 'run.json').read_text())
    role = plan['task_smoke']['task_role']
    if (run.get('status') != 'complete' or run.get('task_role') != role
            or run.get('dataset') != plan['dataset_contract']['identity']
            or run.get('type_manifest_sha256') != plan['type_manifest_sha256']
            or run.get('protocol_sha256') != plan['protocol_sha256']
            or run.get('actual_transitions') != 1024
            or run.get('optimizer_steps') != dict.fromkeys(('actor', 'critic', 'entropy'), 960)
            or run.get('development_evaluated') is not False or run.get('test_evaluated') is not False
            or run.get('smoke_only') is not True):
        raise RuntimeError('complete bound training-only task smoke required')
    curve = json.loads((folder / 'curve.json').read_text())
    if [point['steps'] for point in curve] != [512, 1024]:
        raise RuntimeError('actual post-update task checkpoint curve required')
    with torch.no_grad():
        models = [SAC.load(folder / name, device='cpu') for name in
                  ('initial.zip', 'checkpoint-512.zip', 'checkpoint-1024.zip')]
        for model, steps in zip(models, (0, 512, 1024)):
            if model.policy.task_role != role:
                raise RuntimeError('saved task policy role mismatch')
            if any(not torch.isfinite(value).all() for value in model.policy.state_dict().values()):
                raise RuntimeError('finite downloaded task policy required')
            audit = TaskOptimizerAudit(model, maximum=960)
            try:
                audit.counts = dict.fromkeys(audit.counts, max(0, steps - 64))
                audit.verify(model, steps, 64)
            finally:
                audit.close()
        initial, _, final = models
        if fingerprint(initial.policy.state_dict()) == fingerprint(final.policy.state_dict()):
            raise RuntimeError('actual learned weights required')
        final.load_replay_buffer(folder / 'replay.pkl')
        replay = final.replay_buffer
        if replay.size() != 1024 or replay.pos != 1024 or replay.full:
            raise RuntimeError('complete1024-transition task replay required')
        for name in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
            if not np.isfinite(getattr(replay, name)[:1024]).all():
                raise RuntimeError('finite learned replay required')
        tags = replay.observations[:1024, :, -1]
        if set(np.unique(tags)) != ({0} if role == 'masked_shared' else {-1, 1}):
            raise RuntimeError('saved replay task encoding mismatch')
        bundles = list((folder / 'frozen').iterdir())
        if len(bundles) != 1:
            raise RuntimeError('one frozen task bundle required')
        manifest = json.loads((bundles[0] / 'manifest.json').read_text())
        contract = manifest['metadata']['task_contract']
        if (contract['type_manifest_sha256'] != plan['type_manifest_sha256']
                or contract['dataset_sha256'] != plan['dataset_contract']['identity']
                or contract['role'] != role):
            raise RuntimeError('frozen task study binding mismatch')
        frozen = FrozenTaskPolicy(bundles[0], feature_schema=manifest['feature_schema'],
            feed=manifest['data_manifest']['provenance']['capabilities']['feed'])
        if fingerprint(final.policy.state_dict()) != fingerprint(frozen._model.policy.state_dict()):
            raise RuntimeError('frozen model differs from final learned checkpoint')
        records = frozen.manifest['metadata']['task_contract']['type_manifest']['records']
        for record in records:
            x = np.linspace(-1, 1, 21, dtype=np.float32)
            tag = 0 if role == 'masked_shared' else (1 if record['kind'] == 'etf' else -1)
            expected = filter_exposure(final.predict(np.append(x, tag), deterministic=True)[0], 'unit_magnitude')
            actual = frozen.predict(x, symbol=record['symbol'], session=record['known_from'])
            if not np.array_equal(expected, actual):
                raise RuntimeError('downloaded learned frozen prediction mismatch')
    return run


def validate_task_smoke_archive(path, plan, dataset_id):
    validate_plan(plan)
    if plan['version'] != 44 or dataset_id != plan['dataset_contract']['identity']:
        raise RuntimeError('bound task smoke archive required')
    prefix = 'artifacts/task-smoke/'
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        files = {member.name for member in members if member.isfile()}
        required = {'artifacts/completion.json'} | {prefix + name for name in FILES}
        bundle_files = {name for name in files if re.fullmatch(
            prefix + r'frozen/[a-f0-9]{64}/(model.zip|manifest.json|checksums.json|requirements.lock)', name)}
        allowed = required | bundle_files | {'artifacts/bootstrap-log.txt'}
        if (len(names) != len(set(names)) or not required <= files or len(bundle_files) != 4
                or files - allowed or sum(member.size for member in members) > 256 * 1024 * 1024
                or len(members) > 40 or any(member.size < 0 for member in members)):
            raise RuntimeError('bounded complete task smoke archive required')
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                    or not (member.isfile() or member.isdir())
                    or member.isdir() and not any(f.startswith(member.name.rstrip('/') + '/') for f in allowed)):
                raise RuntimeError('safe task smoke archive members required')
        with tempfile.TemporaryDirectory(prefix='task-smoke-verify-') as temporary:
            archive.extractall(temporary, filter='data')
            root = Path(temporary) / 'artifacts'
            completion = json.loads((root / 'completion.json').read_text())
            git = completion['source_git']
            if (canonical(completion['plan']) != canonical(plan) or completion['dataset'] != dataset_id
                    or completion['gpu'] is not None or completion['python'][:2] != [3, 13]
                    or git.get('dirty') != '' or not re.fullmatch(r'[a-f0-9]{40}', git.get('revision', ''))
                    or not re.fullmatch(r'[a-f0-9]{64}', completion.get('source_hash', ''))):
                raise RuntimeError('clean default-kernel CPU task completion required')
            run = verify_directory(root / 'task-smoke', plan)
            if run['repository'] != git or run['source'] != completion['source_hash']:
                raise RuntimeError('task source provenance mismatch')
            return [dict(algorithm='task_sac_smoke', seed=11, actual_steps=1024,
                         status='complete', task_role=run['task_role'], physical_state_verified=True)]
