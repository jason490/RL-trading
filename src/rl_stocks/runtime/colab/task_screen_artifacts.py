"""Allowlisted full-cell transport; expensive inference verification runs separately."""
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

from rl_stocks.core.contracts import canonical
from rl_stocks.runtime.colab.jobs import validate_plan


ROOT_FILES = {'input.json', 'run.json', 'training-cell.json', 'initial.zip', 'replay.pkl',
              'curve.json', 'curve.csv', 'curve.html', 'selected-training.html',
              'selected-training.json', 'final-training.html', 'final-training.json'}
ROOT_FILES |= {f'checkpoint-{step}.zip' for step in range(10000, 100001, 10000)}
ROOT_FILES |= {f'training-{step}.json' for step in range(10000, 100001, 10000)}


def extract_screen_transport(path, plan, dataset_id, destination):
    """Validate transport before extraction; does not accept learned performance."""
    validate_plan(plan)
    if plan['version'] != 46 or dataset_id != plan['dataset_contract']['identity']:
        raise ValueError('bound task screen transport required')
    prefix = 'artifacts/task-screen/'
    required = {'artifacts/completion.json'} | {prefix + name for name in ROOT_FILES}
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        files = {member.name for member in members if member.isfile()}
        bundles = {name for name in files if re.fullmatch(prefix +
            r'frozen-(selected|final)/[a-f0-9]{64}/(model.zip|manifest.json|checksums.json|requirements.lock)', name)}
        allowed = required | bundles | {'artifacts/bootstrap-log.txt'}
        if (len(members) != len({m.name for m in members}) or len(members) > 64
                or not required <= files or files - allowed or len(bundles) != 8
                or sum(m.size for m in members) > 1024 * 1024 * 1024):
            raise RuntimeError('complete bounded task screen transport required')
        for phase in ('selected', 'final'):
            if len({name.split('/')[-2] for name in bundles if '/frozen-' + phase + '/' in name}) != 1:
                raise RuntimeError('one selected/final frozen bundle required')
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name or member.size < 0
                    or not (member.isfile() or member.isdir())
                    or member.isdir() and not any(f.startswith(member.name.rstrip('/') + '/') for f in allowed)):
                raise RuntimeError('safe task screen transport members required')
        stream = archive.extractfile('artifacts/completion.json')
        completion = json.load(stream)
        git = completion['source_git']
        python = completion['python']
        if (canonical(completion['plan']) != canonical(plan) or completion['dataset'] != dataset_id
                or completion['gpu'] is not None or len(python) != 3
                or any(type(v) is not int for v in python) or python[:2] != [3, 13]
                or git.get('dirty') != '' or not re.fullmatch(r'[a-f0-9]{40}', git.get('revision', ''))
                or not re.fullmatch(r'[a-f0-9]{64}', completion.get('source_hash', ''))):
            raise RuntimeError('clean default-kernel CPU task screen completion required')
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination, filter='data')
    return completion
