"""Smoke-backed CPU deadline projection; no allocations or training at import."""
import hashlib
import json
import math
from pathlib import Path

from rl_stocks.core.contracts import canonical
from rl_stocks.training.distributional_critics import PROTOCOL_SHA256, protocol

REFERENCE_SAC_SECONDS = 702.047255822
REFERENCE_METADATA_SHA256 = '7e1c78ec5ad50e2a45669b0212fc9732b4fdbe08d8fd20cb4bdc6904c8c54d75'
SAFETY_MULTIPLIER = 2.0
ACCEPTANCE_FILE = '.runtime/distributional-smoke-acceptance.json'


def projection(durations):
    arms = set(protocol()['arms'])
    if (not isinstance(durations, dict) or set(durations) != arms
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in durations.values())):
        raise ValueError('positive actual smoke durations for all critic arms required')
    ratio = max(1.0, max(durations.values()) / durations['sac_native'])
    seconds = SAFETY_MULTIPLIER * REFERENCE_SAC_SECONDS * ratio + 1200
    return dict(reference_native_sac_seconds=REFERENCE_SAC_SECONDS,
                reference_metadata_sha256=REFERENCE_METADATA_SHA256, smoke_duration_ratio=ratio,
                safety_multiplier=SAFETY_MULTIPLIER, reserve_seconds=1200, projected_seconds=seconds,
                allocation_deadline_seconds=7200, screen_allocation_allowed=seconds <= 7200,
                heuristic_not_guarantee=True)


def _acceptance(value):
    if (value.get('protocol_sha256') != PROTOCOL_SHA256 or value.get('passed') is not True
            or value.get('cleanup_verified') is not True or value.get('initialization_checks', {}).get('sampling_rng_identical') is not True
            or set(value.get('records', {})) != set(protocol()['arms'])
            or any(r.get('actual_saved_models_restored') is not True for r in value['records'].values())):
        raise ValueError('actual restored and owned-cleaned critic smokes required')
    return {arm: record['duration_seconds'] for arm, record in value['records'].items()}


def make_runtime_profile(root):
    path = Path(root) / ACCEPTANCE_FILE
    value = json.loads(path.read_text())
    durations = _acceptance(value)
    result = dict(version=1, protocol_sha256=PROTOCOL_SHA256, smoke_acceptance_file=ACCEPTANCE_FILE,
                  smoke_acceptance_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  durations_seconds=durations, projection=projection(durations))
    validate_runtime_profile(result, root)
    return result


def validate_runtime_profile(profile, root=None):
    if (not isinstance(profile, dict) or type(profile.get('version')) is not int or profile['version'] != 1
            or profile.get('protocol_sha256') != PROTOCOL_SHA256
            or profile.get('smoke_acceptance_file') != ACCEPTANCE_FILE
            or not isinstance(profile.get('smoke_acceptance_sha256'), str)
            or len(profile['smoke_acceptance_sha256']) != 64):
        raise ValueError('same-protocol actual critic smoke/runtime acceptance required')
    expected = projection(profile['durations_seconds'])
    if canonical(profile.get('projection')) != canonical(expected) or not expected['screen_allocation_allowed']:
        raise ValueError('critic screen projection exceeds fixed owned deadline or differs')
    if root is not None:
        path = Path(root) / ACCEPTANCE_FILE
        if hashlib.sha256(path.read_bytes()).hexdigest() != profile['smoke_acceptance_sha256']:
            raise ValueError('actual critic smoke acceptance checksum mismatch')
        if _acceptance(json.loads(path.read_text())) != profile['durations_seconds']:
            raise ValueError('actual critic smoke durations mismatch')
    return profile


def shared_study_budget(ledger, requested_limit=None, requested_start=None):
    """Keep the same two-unit baseline across smoke/screen controllers and retries."""
    from rl_stocks.core.files import write_once
    from rl_stocks.runtime.colab.budget import BudgetStop
    limit = 2.0 if requested_limit is None else requested_limit
    if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit <= 2:
        raise ValueError('critic study allowance must be positive and at most2 units')
    path = ledger.path.resolve().parent / 'distributional-study-budget.json'
    if not path.exists():
        value = dict(schema='distributional-study-budget-v1', protocol_sha256=PROTOCOL_SHA256,
                     start_units=ledger.value['consumed'], minimum_account_balance=150)
        write_once(path, canonical(value))
    value = json.loads(path.read_text())
    start = value['start_units']
    if (value['schema'] != 'distributional-study-budget-v1' or value['protocol_sha256'] != PROTOCOL_SHA256
            or value['minimum_account_balance'] != 150 or type(start) not in (int, float)
            or not math.isfinite(start) or not 0 <= start <= ledger.value['consumed']
            or requested_start is not None and requested_start != start):
        raise ValueError('original critic-study budget baseline must be preserved')
    if ledger.value['consumed'] - start >= limit:
        raise BudgetStop('critic study allowance exhausted; no automatic restart')
    return float(limit), float(start)
