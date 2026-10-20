"""Pinned TD3 execution-state research contract; no fitting at import."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.rl.registry.factory import resolved_parameters

PROTOCOL_SHA256 = 'c7cfeb51b279e910b71a9d80522486a570fb8e3be156b639e856c5f02a9ab32a'


def protocol():
    p = json.loads((Path(__file__).resolve().parents[3] / 'configs/experiments/td3_execution_state.json').read_text())
    if digest(p) != PROTOCOL_SHA256:
        raise ValueError('TD3 state protocol changed')
    actual = resolved_parameters('td3')
    actual['net_arch'] = actual.pop('policy_kwargs')['net_arch']
    if actual != p['parameters']:
        raise ValueError('TD3 parameters changed')
    return p


def validate_payload(data):
    p = protocol()
    if digest(data) != p['dataset']:
        raise ValueError('TD3 state dataset identity mismatch')
    if (data['manifest']['included_splits'] != ['train', 'validation']
            or data['manifest']['unverified_override'] is not False
            or any(e['split'] not in ('train', 'validation')
                   or any(b['session'] > '2023-12-31' for b in e['bars'])
                   for e in data['episodes'].values())):
        raise ValueError('TD3 state dataset contract mismatch')
    return data
