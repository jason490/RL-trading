"""Fixed off-policy execution-state protocol; no fitting at import."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.rl.registry.factory import resolved_parameters

PROTOCOL_SHA256='06a840db82ec0570c8b0466716a0343cfb93957c8cc2a69f5d9880c7152459ce'


def protocol():
    p=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/sac_execution_state.json').read_text())
    if digest(p)!=PROTOCOL_SHA256:raise ValueError('SAC state protocol changed')
    actual=resolved_parameters('sac');arch=actual.pop('policy_kwargs')['net_arch'];actual['net_arch']=arch
    if actual!=p['parameters']:raise ValueError('SAC parameters changed')
    return p


def validate_payload(data):
    p=protocol()
    if digest(data)!=p['dataset']:raise ValueError('SAC state dataset identity mismatch')
    if (data['manifest']['included_splits']!=['train','validation']
        or data['manifest']['unverified_override'] is not False
        or any(e['split'] not in ('train','validation') or any(b['session']>'2023-12-31' for b in e['bars']) for e in data['episodes'].values())):
        raise ValueError('SAC state dataset contract mismatch')
    return data
