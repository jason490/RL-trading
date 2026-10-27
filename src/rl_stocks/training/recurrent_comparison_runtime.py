"""Persistent original-ledger comparison baseline and single profile claim."""
import json
import math

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.training.recurrent_comparison import protocol,PROTOCOL_SHA256


def shared_profile_budget(ledger,requested_limit=None,requested_start=None):
    from rl_stocks.runtime.colab.budget import BudgetStop
    runtime=ledger.path.resolve().parent
    if (runtime/'recurrent-comparison-profile-allocation.json').exists():
        raise RuntimeError('comparison profile already claimed; no automatic retry')
    spec=protocol()['execution']
    limit=spec['profile_maximum_incremental_units'] if requested_limit is None else requested_limit
    if type(limit) not in (int,float) or not math.isfinite(limit) or not 0<limit<=.1:
        raise ValueError('profile allowance must be positive and at most0.1 units')
    path=runtime/'recurrent-comparison-budget.json'
    if not path.exists():
        write_once(path,canonical(dict(protocol_sha256=PROTOCOL_SHA256,
            start_units=ledger.value['consumed'],minimum_account_balance=150,
            max_study_incremental_units=spec['max_study_incremental_units'])))
    value=json.loads(path.read_text())
    start=value['start_units']
    if (value['protocol_sha256']!=PROTOCOL_SHA256 or value['minimum_account_balance']!=150
            or value['max_study_incremental_units']!=2
            or type(start) not in (int,float) or not math.isfinite(start)
            or not 0<=start<=ledger.value['consumed']
            or requested_start is not None and requested_start!=start):
        raise ValueError('original comparison baseline and account floor must be preserved')
    if ledger.value['consumed']-start>=limit:
        raise BudgetStop('comparison profiling allowance exhausted')
    return float(limit),float(start)


def claim_profile(ledger,session):
    if type(session) is not str or not session.startswith('rl-stocks-cpu-'):
        raise ValueError('owned CPU profile session required')
    path=ledger.path.resolve().parent/'recurrent-comparison-profile-allocation.json'
    if path.exists():
        raise RuntimeError('profile already claimed; no automatic retry')
    write_once(path,canonical(dict(protocol_sha256=PROTOCOL_SHA256,session=session,
        allocation_deadline_seconds=900,teardown_reserve_seconds=420,automatic_retry_allowed=False)))
