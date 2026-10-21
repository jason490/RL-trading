"""Fixed SAC discount-horizon comparison, with unchanged Adam/temperature."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.training.sac_regularization import optimizer_record,validate_payload
from rl_stocks.training.sac_regularization import validate_optimizer_record,validate_final_optimizer_record

PROTOCOL_SHA256='21844c6542cc6bc324add985bfcd5fb7617a9c55dda2c59e5ebaa820846fcaf7'


def protocol():
    p=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/sac_discount.json').read_text())
    if digest(p)!=PROTOCOL_SHA256:raise ValueError('discount protocol changed')
    from rl_stocks.training.sac_regularization import protocol as base
    b=base()
    if p['parameters']!=b['parameters'] or p['dataset']!=b['dataset']:raise ValueError('discount base contract changed')
    return p


def overrides(arm,smoke=False):
    from rl_stocks.training.sac_regularization import overrides as base
    if arm not in protocol()['arms']:raise ValueError('fixed discount arm required')
    return {**base('control',smoke),'gamma':protocol()['discount'][arm]}


def model_record(model):
    return dict(gamma=model.gamma,num_timesteps=model.num_timesteps,optimizer=optimizer_record(model))


def validate_record(record,arm,initial=False):
    if arm not in protocol()['arms'] or record['gamma']!=protocol()['discount'][arm]:
        raise ValueError('actual discount factor mismatch')
    steps=record['num_timesteps']
    if type(steps) is not int or (steps!=0 if initial else steps<=0):raise ValueError('actual saved step count mismatch')
    if initial:validate_optimizer_record(record['optimizer'],'control')
    else:validate_final_optimizer_record(record['optimizer'],'control')
