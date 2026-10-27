"""Explicit reviewed continuation of one stopped engineering profile, never an automatic retry."""
import hashlib
import json
from pathlib import Path

from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.files import write_once
from rl_stocks.training.recurrent_comparison import PROTOCOL_SHA256

REVIEW_SHA256='fc4ff33df54fc28641056231822b0026381f3d281a5077060a7c80ab192de491'
ROOT=Path(__file__).resolve().parents[3]


def review_spec():
    value=json.loads((ROOT/'configs/experiments/recurrent_sac_execution_review.json').read_text())
    if digest(value)!=REVIEW_SHA256 or value['research_protocol_sha256']!=PROTOCOL_SHA256:
        raise ValueError('fixed execution review required')
    return value


def verify_original_records(ledger):
    spec=review_spec()
    runtime=ledger.path.resolve().parent
    baseline=json.loads((runtime/'recurrent-comparison-budget.json').read_text())
    if (baseline['protocol_sha256']!=PROTOCOL_SHA256
            or baseline['start_units']!=spec['original_study_start_units']
            or baseline['max_study_incremental_units']!=2 or baseline['minimum_account_balance']!=150
            or ledger.value['consumed']<baseline['start_units']
            or ledger.value.get('minimum_account_balance')!=150):
        raise ValueError('original study baseline, charges and account floor required')
    for filename,key in [('recurrent-comparison-profile-allocation.json','original_claim_sha256'),
                         ('crossq-integrity-stop.json','original_crossq_stop_sha256')]:
        if hashlib.sha256((runtime/filename).read_bytes()).hexdigest()!=spec[key]:
            raise ValueError('original immutable claims and integrity stop must remain unchanged')
    return baseline


def authorize_profile_review(ledger,usage,authorization):
    """Call only after explicit user authorization; no allocation occurs here."""
    spec=review_spec()
    verify_original_records(ledger)
    destination=ledger.path.resolve().parent/'recurrent-comparison-profile-review-authorization.json'
    if destination.exists():
        raise RuntimeError('profile review already recorded; no repeated authorization')
    if ledger.active['session']!=spec['original_stopped_fleet']:
        raise ValueError('exact original stopped fleet required')
    from rl_stocks.runtime.colab.budget import BudgetStop
    if not isinstance(authorization,str) or not authorization.strip():
        raise BudgetStop('explicit reviewed authorization required')
    if not ledger.value['budget_stopped']:
        raise BudgetStop('exact stopped budget review required')
    if ledger.value['consumed']-spec['original_study_start_units']>=.1:
        raise BudgetStop('shared profile allowance exhausted before review')
    ledger_reference='recurrent-profile-review:'+REVIEW_SHA256+' user-authorization:'+authorization.strip()
    record=dict(review_sha256=REVIEW_SHA256,research_protocol_sha256=PROTOCOL_SHA256,
                scope=spec['scope'],ledger_authorization=ledger_reference)
    # Fail closed: a record alone cannot admit tasks; a matching persisted ledger
    # review is also required. Publication failure cannot clear the budget stop.
    write_once(destination,canonical(record))
    review=ledger.review_resume(usage,ledger_reference,spec['original_stopped_fleet'],150)
    return dict(**record,resumption_review=review)


def shared_reviewed_profile_budget(ledger,requested_limit=None,requested_start=None):
    from rl_stocks.runtime.colab.budget import BudgetStop
    spec=review_spec()
    baseline=verify_original_records(ledger)
    runtime=ledger.path.resolve().parent
    if (runtime/'recurrent-comparison-corrected-profile-allocation.json').exists():
        raise RuntimeError('corrected profile already claimed; no automatic retry')
    authorization=runtime/'recurrent-comparison-profile-review-authorization.json'
    if ledger.value['budget_stopped'] or not authorization.exists():
        raise BudgetStop('explicit reviewed user authorization required before corrected profile')
    record=json.loads(authorization.read_text())
    prefix='recurrent-profile-review:'+REVIEW_SHA256+' user-authorization:'
    reference=record.get('ledger_authorization')
    if (not isinstance(reference,str) or not reference.startswith(prefix) or not reference[len(prefix):].strip()
            or record['review_sha256']!=REVIEW_SHA256 or record['research_protocol_sha256']!=PROTOCOL_SHA256
            or record['scope']!=spec['scope']
            or not any(r.get('authorization')==record.get('ledger_authorization')
                       and r.get('stopped_session')==spec['original_stopped_fleet']
                       and r.get('minimum_account_balance')==150
                       for r in ledger.value.get('resumption_reviews',[]))):
        raise ValueError('exact persisted original-ledger resumption review required')
    start=baseline['start_units']
    if requested_limit not in (None,.1) or requested_start not in (None,start):
        raise ValueError('unchanged shared0.1-unit profile allowance and original baseline required')
    if ledger.value['consumed']-start>=.1:
        raise BudgetStop('shared original-plus-corrected profile allowance exhausted')
    return .1,float(start)


def claim_corrected_profile(ledger,session):
    shared_reviewed_profile_budget(ledger)
    if type(session) is not str or not session.startswith('rl-stocks-cpu-'):
        raise ValueError('owned CPU corrected profile session required')
    write_once(ledger.path.resolve().parent/'recurrent-comparison-corrected-profile-allocation.json',
               canonical(dict(review_sha256=REVIEW_SHA256,session=session,
                    allocation_deadline_seconds=1200,teardown_reserve_seconds=420,
                    automatic_retry_allowed=False,original_stopped_fleet=review_spec()['original_stopped_fleet'])))
