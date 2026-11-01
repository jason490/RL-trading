"""Inspect bounded owned task-profile downloads without local optimization."""
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

import torch
from stable_baselines3 import SAC

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.runtime.colab.jobs import validate_plan
from rl_stocks.training.task_optimizer import TaskOptimizerAudit
from rl_stocks.training.task_profile_worker import PROFILE_STEPS, PROFILE_UPDATES


FILES = {'initial.zip', 'profile-final.zip', 'input.json', 'profile.json'}


def verify_profile_directory(folder, plan, role, completion):
    folder = Path(folder)
    record = json.loads((folder / 'profile.json').read_text())
    initial_metadata = json.loads((folder / 'input.json').read_text())
    expected = dict(task_role=role, protocol_sha256=plan['protocol_sha256'],
        dataset=plan['dataset_contract']['identity'], type_manifest_sha256=plan['type_manifest_sha256'],
        repository=completion['source_git'], source=completion['source_hash'], profile_only=True,
        seed=11, development_evaluated=False, test_evaluated=False, performance_promotion=False)
    if (canonical(initial_metadata) != canonical(expected)
            or any(record.get(k) != v for k, v in expected.items())
            or record.get('status') != 'complete' or record.get('actual_transitions') != PROFILE_STEPS
            or record.get('optimizer_steps') != dict.fromkeys(('actor', 'critic', 'entropy'), PROFILE_UPDATES)
            or record.get('performance_scores_exported') is not False
            or not re.fullmatch(r'[a-f0-9]{64}', record.get('holding_path_sha256', ''))):
        raise RuntimeError('complete bound training-only profile required')
    times = [record.get(k) for k in ('learning_seconds', 'evaluation_seconds', 'inclusive_elapsed_seconds')]
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in times)
            or sum(times[:2]) > times[2] or times[2] >= plan['allocation_deadline_seconds']):
        raise RuntimeError('finite measured profile component times required')
    states = []
    with torch.no_grad():
        for name, steps in [('initial.zip', 0), ('profile-final.zip', PROFILE_STEPS)]:
            model = SAC.load(folder / name, device='cpu')
            if (model.policy.task_role != role or model.batch_size != 256
                    or model.learning_starts != 1000 or model.buffer_size != 100000
                    or model.observation_space.shape != (22,)):
                raise RuntimeError('physical research-geometry task profile required')
            if any(not torch.isfinite(v).all() for v in model.policy.state_dict().values()):
                raise RuntimeError('finite physical profile parameters required')
            audit = TaskOptimizerAudit(model, maximum=PROFILE_UPDATES)
            try:
                audit.counts = dict.fromkeys(audit.counts, max(0, steps - 1000))
                audit.verify(model, steps, 1000)
            finally:
                audit.close()
            states.append(fingerprint(model.policy.state_dict()))
    if states[0] == states[1]:
        raise RuntimeError('actual learned profile weights required')
    return record


def validate_task_profile_archive(path, plan, dataset_id):
    validate_plan(plan)
    if plan['version'] != 45 or dataset_id != plan['dataset_contract']['identity']:
        raise RuntimeError('bound task profile archive required')
    required = {'artifacts/completion.json'} | {
        f'artifacts/task-profile-{role}/{name}' for role in plan['task_profile_roles'] for name in FILES}
    allowed = required | {'artifacts/bootstrap-log.txt'}
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        files = {m.name for m in members if m.isfile()}
        if (len(names) != len(set(names)) or not required <= files or files - allowed
                or len(members) > 24 or sum(m.size for m in members) > 64 * 1024 * 1024):
            raise RuntimeError('complete bounded task profile archive required')
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name or member.size < 0
                    or not (member.isfile() or member.isdir())
                    or member.isdir() and not any(f.startswith(member.name.rstrip('/') + '/') for f in allowed)):
                raise RuntimeError('safe profile archive members required')
        with tempfile.TemporaryDirectory(prefix='task-profile-verify-') as temporary:
            archive.extractall(temporary, filter='data')
            root = Path(temporary) / 'artifacts'
            completion = json.loads((root / 'completion.json').read_text())
            git = completion['source_git']
            python = completion['python']
            if (canonical(completion['plan']) != canonical(plan) or completion['dataset'] != dataset_id
                    or completion['gpu'] is not None or len(python) != 3
                    or any(type(v) is not int for v in python) or python[:2] != [3, 13]
                    or git.get('dirty') != '' or not re.fullmatch(r'[a-f0-9]{40}', git.get('revision', ''))
                    or not re.fullmatch(r'[a-f0-9]{64}', completion.get('source_hash', ''))):
                raise RuntimeError('clean default-kernel CPU profile completion required')
            records = [verify_profile_directory(root / f'task-profile-{role}', plan, role, completion)
                       for role in plan['task_profile_roles']]
            if len({r['holding_path_sha256'] for r in records}) != 1:
                raise RuntimeError('same training-window holding paths required')
            return [dict(algorithm='task_sac_profile', seed=11, actual_steps=PROFILE_STEPS,
                         task_role=r['task_role'], bundle_id=digest(r), status='complete',
                         physical_state_verified=True, learning_seconds=r['learning_seconds'],
                         evaluation_seconds=r['evaluation_seconds']) for r in records]
