"""Bind already physically verified cells to retained artifacts before fitted gates."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.task_gates import training_gate


MASKED_RECEIPT_SHA256 = 'dc19b256f760ca1cbc848423332323d025a473515e24f80c7599cce4608b3749'


def bind_verified_cell(archive_path, proof, role):
    """Cheap transport/proof binding after independent saved-model replay, not a replacement for it."""
    if (proof.get('task_role') != role or proof.get('replayed_checkpoints') != 10
            or proof.get('performance_promotion') is not False):
        raise ValueError('complete independent ten-checkpoint proof required')
    with tarfile.open(archive_path) as archive:
        def read(name):
            member = archive.getmember('artifacts/task-screen/' + name)
            if not member.isfile() or not 0 < member.size <= 2 * 1024 * 1024:
                raise ValueError('bounded verified cell metadata required')
            return json.load(archive.extractfile(member))
        cell, run = read('training-cell.json'), read('run.json')
        if (digest(cell) != digest(proof['training_cell']) or run.get('status') != 'complete'
                or run.get('task_role') != role or run.get('actual_transitions') != 100000
                or run.get('development_evaluated') is not False or run.get('test_evaluated') is not False
                or run['selection']['selected']['steps'] != proof['selected_step']):
            raise ValueError('independently verified retained gate cell mismatch')
        for phase, identity in proof['frozen_bundles'].items():
            archive.getmember(f'artifacts/task-screen/frozen-{phase}/{identity}/manifest.json')
    return cell


def accept_stopped_screen(root):
    root = Path(root)
    ledger = json.loads((root / '.runtime/usage-ledger.json').read_text())
    if any(a['status'] != 'stopped' for a in ledger['attempts']):
        raise ValueError('all owned allocations must be stopped before family admission')
    path = root / 'results/task-conditioning-2026-10-31/masked-screen-acceptance.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != MASKED_RECEIPT_SHA256:
        raise ValueError('pinned exact masked acceptance required')
    receipt = json.loads(path.read_text())
    archive = root / receipt['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt['archive_sha256']:
        raise ValueError('accepted masked archive changed')
    cells = {'masked_shared': bind_verified_cell(archive, receipt['exact_physical_proof'], 'masked_shared')}
    sources = {'masked_shared': dict(session=receipt['session'], archive_sha256=receipt['archive_sha256'])}
    state = json.loads((root / '.runtime/task-screen-fleet-suffix-v1.json').read_text())
    attempts = [a for a in ledger['attempts'] if a['session'] == state['fleet']]
    if state.get('status') != 'complete' or state.get('cleanup_verified') is not True or len(attempts) != 1:
        raise ValueError('complete verified and stopped two-role suffix required')
    children = attempts[0]['children']
    for role in ('tagged_shared', 'tagged_routed'):
        accepted = state['completed']['task-screen-' + role.replace('_', '-')]
        session = accepted['session']
        if not re.fullmatch(r'rl-stocks-cpu-[a-f0-9]{12}', session):
            raise ValueError('safe owned task session required')
        child = children[session]
        if child.get('status') != 'stopped' or child.get('training_verified') is not True:
            raise ValueError('physical suffix verification and cleanup required')
        folder = root / 'artifacts/colab' / session
        verified = json.loads((folder / 'verification.json').read_text())
        if len(verified) != 1 or verified[0]['bundle_id'] != accepted['runs'][0]['bundle_id']:
            raise ValueError('accepted suffix proof identity mismatch')
        proof = verified[0]['cell']
        cells[role] = bind_verified_cell(folder / 'artifacts.tar.gz', proof, role)
        sources[role] = dict(session=session,
            archive_sha256=hashlib.sha256((folder / 'artifacts.tar.gz').read_bytes()).hexdigest())
    gate = training_gate(cells)
    return dict(gate=gate, artifacts=sources, project_units=ledger['consumed'],
                cleanup_verified=True, retired_test_access=False, performance_promotion=False)
