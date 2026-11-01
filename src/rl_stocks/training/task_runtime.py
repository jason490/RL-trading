"""Persistent study allowance and non-repeating task-smoke allocation claims."""
import json
import math
import re

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.training.task_conditioning import PROTOCOL_SHA256, protocol
from rl_stocks.rl.sac.task_policy import check_role


def shared_budget(ledger, requested_limit=None, requested_start=None):
    from rl_stocks.runtime.colab.budget import BudgetStop
    spec = protocol()['execution']
    maximum = spec['max_study_incremental_units']
    limit = maximum if requested_limit is None else requested_limit
    consumed = ledger.value['consumed']
    if (type(limit) not in (int, float) or not math.isfinite(limit)
            or not 0 < limit <= maximum):
        raise ValueError('task study allowance must be positive and at most2 units')
    if (ledger.value['budget_stopped'] or ledger.value.get('minimum_account_balance') != 150
            or ledger.account_teardown_balance < 170):
        raise BudgetStop('task study requires original project stop and account protections')
    path = ledger.path.resolve().parent / 'task-conditioning-budget-v1.json'
    if not path.exists():
        if requested_start is not None and requested_start != consumed:
            raise ValueError('new task baseline must equal cumulative consumed units')
        write_once(path, canonical(dict(protocol_sha256=PROTOCOL_SHA256,
            start_units=consumed, minimum_account_balance=150)))
    value = json.loads(path.read_text())
    start = value['start_units']
    if (value.get('protocol_sha256') != PROTOCOL_SHA256
            or value.get('minimum_account_balance') != 150
            or type(start) not in (int, float) or not math.isfinite(start)
            or not 0 <= start <= consumed
            or requested_start is not None and requested_start != start):
        raise ValueError('original task study baseline must be preserved')
    if consumed - start >= limit:
        raise BudgetStop('task study allowance exhausted; no automatic retry')
    return float(limit), float(start)


def claim_smoke(ledger, session, task_role):
    role = check_role(task_role)
    shared_budget(ledger)
    if not isinstance(session, str) or not re.fullmatch(r'rl-stocks-cpu-[a-f0-9]{12}', session):
        raise ValueError('owned CPU task session required')
    path = ledger.path.resolve().parent / f'task-conditioning-smoke-{role}-allocation-v1.json'
    if path.exists():
        raise RuntimeError('task role smoke already claimed; no automatic retry')
    write_once(path, canonical(dict(protocol_sha256=PROTOCOL_SHA256, session=session,
        task_role=role, allocation_deadline_seconds=7200, teardown_reserve_seconds=360,
        automatic_retry_allowed=False)))


def validate_smoke_roles(root, ledger, roles):
    """Admit full initial matrix or the reviewed unallocated suffix, never a retry."""
    from pathlib import Path
    import hashlib
    from rl_stocks.runtime.colab.jobs import task_smoke_plan
    from rl_stocks.runtime.colab.task_smoke_artifacts import validate_task_smoke_archive
    expected = {'masked_shared', 'tagged_shared', 'tagged_routed'}
    if len(roles) == 3 and set(roles) == expected:
        return
    if len(roles) != 2 or set(roles) != expected - {'masked_shared'}:
        raise ValueError('complete initial task matrix or reviewed unallocated suffix required')
    root = Path(root)
    receipt_path = root / 'results/task-conditioning-2026-10-31/masked-smoke-acceptance.json'
    if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != '9d3c9082f21e36a351d26ff25ebae794b3091b77731b273f7cb3ab8a183ba14e':
        raise ValueError('exact retained masked acceptance review required')
    receipt = json.loads(receipt_path.read_text())
    runtime = ledger.path.resolve().parent
    old = json.loads((runtime / 'task-smoke-fleet-v1.json').read_text())
    claim = json.loads((runtime / 'task-conditioning-smoke-masked_shared-allocation-v1.json').read_text())
    session = receipt['session']
    attempts = [a for a in ledger.value['attempts'] if a['session'] == receipt['original_fleet']]
    if (len(attempts) != 1 or attempts[0]['status'] != 'stopped'
            or set(attempts[0]['children']) != {session}
            or attempts[0]['children'][session]['status'] != 'stopped'
            or old.get('fleet') != receipt['original_fleet'] or old.get('status') != 'failed'
            or old.get('cleanup_verified') is not True or claim.get('session') != session
            or claim.get('task_role') != 'masked_shared' or claim.get('protocol_sha256') != PROTOCOL_SHA256
            or any((runtime / f'task-conditioning-smoke-{role}-allocation-v1.json').exists() for role in roles)):
        raise ValueError('preserved original failure, stopped ownership and unallocated suffix required')
    archive = root / receipt['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt['archive_sha256']:
        raise ValueError('original accepted archive changed')
    plan = task_smoke_plan('masked_shared')
    proof = validate_task_smoke_archive(archive, plan, plan['dataset_contract']['identity'])
    if proof != receipt['physical_acceptance']:
        raise ValueError('original physical masked acceptance required')
