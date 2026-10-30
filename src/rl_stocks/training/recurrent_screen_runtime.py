"""Original-ledger screen allowance and immutable per-role/seed allocation claims."""
import math

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.training.recurrent_execution_review import verify_original_records
from rl_stocks.training.recurrent_comparison import protocol
from rl_stocks.training.recurrent_screen_spec import EXECUTION_SHA256,specification


def shared_screen_budget(ledger,requested_limit=None,requested_start=None):
    from rl_stocks.runtime.colab.budget import BudgetStop
    baseline=verify_original_records(ledger)
    if ledger.value['budget_stopped']:
        raise BudgetStop('existing stop requires explicit review; screen never resumes automatically')
    claims=ledger.path.resolve().parent/'recurrent-comparison-screen-claims'
    if any((claims/(arm+'-11.json')).exists() for arm in protocol()['arms']):
        raise RuntimeError('screen allocation already claimed; no automatic retry or partial continuation')
    limit=2. if requested_limit is None else requested_limit
    if type(limit) not in (int,float) or not math.isfinite(limit) or not 0<limit<=2:
        raise ValueError('unchanged original at-most-two-unit study allowance required')
    start=baseline['start_units']
    if requested_start is not None and requested_start!=start:
        raise ValueError('original study baseline must include both profiles')
    if ledger.value['consumed']-start>=limit:
        raise BudgetStop('original shared study allowance exhausted')
    return float(limit),float(start)


def claim_screen_cell(ledger,session,arm):
    if arm not in protocol()['arms'] or type(session) is not str or not session.startswith('rl-stocks-cpu-'):
        raise ValueError('registered screen role and owned CPU session required')
    if ledger.value['budget_stopped']:
        raise RuntimeError('stopped ledger cannot claim a screen cell')
    path=ledger.path.resolve().parent/'recurrent-comparison-screen-claims'/(arm+'-11.json')
    if path.exists():
        raise RuntimeError('role/seed already claimed; no automatic retry')
    write_once(path,canonical(dict(execution_sha256=EXECUTION_SHA256,session=session,arm=arm,seed=11,
        allocation_deadline_seconds=specification()['allocation_seconds'][arm],
        teardown_reserve_seconds=360,transitions=100000,automatic_retry_allowed=False)))
