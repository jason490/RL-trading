"""One fixed actor/critic Adam L2 study; entropy optimizer stays unchanged."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.training.sac_temperature import initialization_record

PROTOCOL_SHA256 = 'b9b0b310684628c5994b08075769924223577a2265db268b345bc147f9a0388d'


def protocol():
    p=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/sac_regularization.json').read_text())
    if digest(p)!=PROTOCOL_SHA256:raise ValueError('regularization protocol changed')
    from rl_stocks.training.diversity import protocol as base
    b=base()
    if p['parameters']!=b['parameters'] or p['dataset']!=b['datasets']['expanded']:
        raise ValueError('unchanged SAC parameters/expanded data required')
    return p


def overrides(arm,smoke=False):
    p=protocol()
    if arm not in p['arms'] or type(smoke) is not bool:raise ValueError('fixed arm/smoke required')
    return {'policy_kwargs':{'net_arch':[32,32] if smoke else [64,64],
                            'optimizer_kwargs':{'weight_decay':p['weight_decay'][arm]}}}


def optimizer_record(model):
    records={}
    for name,optimizer in [('actor',model.actor.optimizer),('critic',model.critic.optimizer),
                            ('entropy',model.ent_coef_optimizer)]:
        records[name]={'class':type(optimizer).__module__+'.'+type(optimizer).__name__,
                       'weight_decay':[g['weight_decay'] for g in optimizer.param_groups]}
    return {'initialization':initialization_record(model),'optimizers':records}


def validate_optimizer_record(record,arm):
    from rl_stocks.training.diversity import validate_initialization
    p=protocol()
    if arm not in p['arms']:raise ValueError('fixed arm required')
    validate_initialization(record['initialization'])
    for name in ('actor','critic','entropy'):
        item=record['optimizers'][name];expected=0. if name=='entropy' else p['weight_decay'][arm]
        if item!={'class':'torch.optim.adam.Adam','weight_decay':[expected]}:
            raise ValueError('actual Adam regularization mismatch')


def validate_payload(data):
    from rl_stocks.training.diversity import validate_payload as validate
    return validate(data,'expanded')


def validate_final_optimizer_record(record,arm):
    # Alpha changes during training; only optimizer configuration is fixed here.
    import math
    alpha=record['initialization']['alpha']
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('positive finite learned alpha required')
    probe=dict(record)
    probe['initialization']={**record['initialization'],'alpha':1.0}
    validate_optimizer_record(probe,arm)
