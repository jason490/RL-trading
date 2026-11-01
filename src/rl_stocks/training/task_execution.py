"""Explicit measured subsequent-allocation review; original duration failure retained."""
import hashlib
import json
from pathlib import Path
import re

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.jobs import task_profile_plan
from rl_stocks.runtime.colab.task_profile_artifacts import validate_task_profile_archive
from rl_stocks.training.task_conditioning import PROTOCOL_SHA256
from rl_stocks.training.task_profile_sources import verify_profile_sources
from rl_stocks.training.task_projection import project_screen
from rl_stocks.training.task_runtime import shared_budget, verify_all_smokes


REVIEW_SHA256 = '45e290c79d56c2914738ff8c7da407e01d9098b50b7a4b66a74ba3f1d9e58247'


def review_spec():
    path = Path(__file__).resolve().parents[3] / 'configs/experiments/task_screen_execution_review.json'
    value = json.loads(path.read_text())
    if digest(value) != REVIEW_SHA256 or value['protocol_sha256'] != PROTOCOL_SHA256:
        raise ValueError('fixed execution-only task review required')
    return value


def verify_execution(root, ledger):
    root = Path(root)
    spec = review_spec()
    limit, start = shared_budget(ledger)
    if any(a['status'] != 'stopped' for a in ledger.value['attempts']):
        raise ValueError('all original owned sessions must be stopped before screen admission')
    verify_all_smokes(root, ledger)
    path = root / 'results/task-conditioning-2026-10-31/research-profile-acceptance.json'
    receipt = json.loads(path.read_text())
    if (hashlib.sha256(path.read_bytes()).hexdigest() != spec['profile_receipt_file_sha256']
            or digest(receipt) != spec['profile_acceptance_sha256']
            or receipt['study_start_units'] != start or receipt['projection']['execution_admitted'] is not False):
        raise ValueError('original accepted profile, baseline and failed duration projection required')
    runtime = ledger.path.resolve().parent
    claim = json.loads((runtime / 'task-conditioning-profile-allocation-v1.json').read_text())
    state = json.loads((runtime / 'task-profile-fleet-v1.json').read_text())
    session = receipt['session']
    attempts = [a for a in ledger.value['attempts'] if a['session'] == state['fleet']]
    if (state.get('status') != 'complete' or state.get('cleanup_verified') is not True
            or claim.get('session') != session or claim.get('protocol_sha256') != PROTOCOL_SHA256
            or len(attempts) != 1 or set(attempts[0]['children']) != {session}
            or attempts[0]['children'][session]['status'] != 'stopped'):
        raise ValueError('exact completed profile ownership and cleanup required')
    archive = root / receipt['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt['archive_sha256']:
        raise ValueError('accepted profile archive changed')
    plan = task_profile_plan()
    proof = validate_task_profile_archive(archive, plan, plan['dataset_contract']['identity'])
    if digest(proof) != digest(receipt['physical_acceptance']):
        raise ValueError('actual physical profile acceptance required')
    sources = verify_profile_sources(root, receipt['completion']['source_git']['revision'])
    if sources != receipt['compute_sources']:
        raise ValueError('profiled compute dependency inventory changed')
    remaining = limit - (ledger.value['consumed'] - start)
    projection = project_screen(receipt['profiles'], remaining_study_units=remaining,
                                aggregate_rate=receipt['observed_rate'])
    if projection['roles'] != receipt['projection']['roles']:
        raise ValueError('original measured duration calculation changed')
    for role, duration in spec['allocation_seconds'].items():
        measured = projection['roles'][role]
        if measured['projected_allocation_seconds'] > duration or not measured['within_verification_limit']:
            raise ValueError('measured operation exceeds reviewed execution limit')
        if (runtime / f'task-screen-{role}-allocation-v1.json').exists():
            raise ValueError('screen role already claimed; no automatic retry')
    planned_units = sum(spec['allocation_seconds'].values()) * (receipt['observed_rate'] + .01) / 3600
    if not projection['within_allowance'] or planned_units >= remaining:
        raise ValueError('reviewed complete allocation plan exceeds original study allowance')
    return dict(review_sha256=REVIEW_SHA256, profile_acceptance_sha256=spec['profile_acceptance_sha256'],
                planned_units=planned_units, remaining_study_units=remaining, source_files=len(sources))


def claim_screen(ledger, session, role):
    spec = review_spec()
    shared_budget(ledger)
    if role not in spec['allocation_seconds'] or not isinstance(session, str) or not re.fullmatch(r'rl-stocks-cpu-[a-f0-9]{12}', session):
        raise ValueError('registered owned CPU screen role required')
    path = ledger.path.resolve().parent / f'task-screen-{role}-allocation-v1.json'
    if path.exists():
        raise RuntimeError('task screen already claimed; no automatic retry')
    write_once(path, canonical(dict(session=session, task_role=role, review_sha256=REVIEW_SHA256,
        allocation_deadline_seconds=spec['allocation_seconds'][role], teardown_reserve_seconds=360,
        automatic_retry_allowed=False)))
