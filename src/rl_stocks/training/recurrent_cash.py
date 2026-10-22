"""Fixed cash-availability hypothesis; builders never allocate or fit."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest

PROTOCOL_SHA256 = '109ccfb4e68b8d83dc62102967dc949ab7cec2df76b9cf801be105eea6130cc0'


def protocol():
    spec=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/recurrent_cash.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:
        raise ValueError('cash-choice protocol changed')
    return spec


def arm_spec(arm):
    spec=protocol()
    if arm not in spec['arms']:
        raise ValueError('fixed cash-choice arm required')
    return dict(spec['arm_contracts'][arm])


def overrides(arm, smoke=False):
    contract=arm_spec(arm)
    if type(smoke) is not bool:
        raise ValueError('explicit boolean smoke required')
    spec=protocol()
    params=dict(spec['parameters']);params.pop('net_arch')
    policy=dict(net_arch=list(spec['parameters']['net_arch']),**spec['recurrent_parameters'])
    if smoke:
        for key in ('n_steps','batch_size','n_epochs'):
            params[key]=spec['smoke_parameters'][key]
        policy.update(net_arch=list(spec['smoke_parameters']['net_arch']),
                      lstm_hidden_size=spec['smoke_parameters']['lstm_hidden_size'])
    if contract['distribution']=='direction_protection':
        policy['cash_enabled']=contract['cash_enabled']
    return dict(params,policy_kwargs=policy)
