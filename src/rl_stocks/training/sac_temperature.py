"""Fixed initial-temperature study and read-only initialization fingerprints."""
import hashlib,json,math
from pathlib import Path
from rl_stocks.core.contracts import digest

PROTOCOL_SHA256='c86473966ae00aed2e9410bbb528a01d664b1e1d6e02d26dd7afbc7188db8e70'


def protocol():
    p=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/sac_temperature.json').read_text())
    if digest(p)!=PROTOCOL_SHA256:raise ValueError('temperature protocol changed')
    from rl_stocks.rl.registry.factory import resolved_parameters
    actual=resolved_parameters('sac');actual.pop('ent_coef');actual['net_arch']=actual.pop('policy_kwargs')['net_arch']
    if actual!=p['parameters']:raise ValueError('temperature base parameters changed')
    return p


def validate_payload(data):
    if digest(data)!=protocol()['dataset']:raise ValueError('temperature dataset identity mismatch')
    if data['manifest']['included_splits']!=['train','validation'] or any(e['split'] not in ('train','validation') or any(b['session']>'2023-12-31' for b in e['bars']) for e in data['episodes'].values()):raise ValueError('temperature data contract mismatch')
    return data


def initialization_record(model):
    fingerprints={}
    for name in ('actor','critic','critic_target'):
        h=hashlib.sha256()
        for key,value in sorted(getattr(model,name).state_dict().items()):
            h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
        fingerprints[name]=h.hexdigest()
    return dict(alpha=float(model.log_ent_coef.detach().exp().cpu().item()),target_entropy=float(model.target_entropy),network_sha256=fingerprints)


def validate_initialization(record,arm):
    expected=protocol()['initial_alpha'][arm]
    hashes=record.get('network_sha256',{})
    if (not math.isclose(record.get('alpha',float('nan')),expected,rel_tol=1e-6,abs_tol=1e-10)
        or record.get('target_entropy')!=-4. or set(hashes)!={'actor','critic','critic_target'}
        or any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes.values())):
        raise ValueError('temperature initialization mismatch')
