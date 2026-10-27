"""Fixed full-development package comparison; no allocation or training at import."""
import json
import math
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.training.recurrent_loop import LoopConfig

PROTOCOL_SHA256 = '58351c3c62f4e65fb59c49e29ab87edb3226ca17445b155d8268a885a6b9c3eb'
ROOT = Path(__file__).resolve().parents[3]


def protocol():
    value=json.loads((ROOT/'configs/experiments/recurrent_sac_comparison.json').read_text())
    if digest(value)!=PROTOCOL_SHA256:
        raise ValueError('fixed recurrent SAC comparison protocol required')
    if (value['transitions_per_cell']!=100000 or value['evaluation_every']!=10000
            or value['confirmation_seeds']!=[11,23,37] or value['checkpoint_phase']!='post_update'):
        raise ValueError('fixed full-panel transition/seed/checkpoint contracts required')
    LoopConfig(**value['arm_contracts']['gru_sac']['loop'])
    return value


def candidate_config(seed):
    from dataclasses import replace
    spec=protocol()
    if type(seed) is not int or seed not in spec['confirmation_seeds']:
        raise ValueError('registered comparison seed required')
    return replace(LoopConfig(**spec['arm_contracts']['gru_sac']['loop']),seed=seed)


def higher_p95(values, *, count, discard=0):
    if (type(values) is not list or len(values)!=count
            or any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in values)):
        raise ValueError('complete finite positive timing samples required')
    retained=sorted(values[discard:])
    return retained[math.ceil(.95*(len(retained)-1))]


def profile_admission(profile):
    """Pure conservative projection; cannot authorize tasks without physical proof."""
    spec=protocol()
    contract=spec['profiling']
    if (profile.get('protocol_sha256')!=PROTOCOL_SHA256 or profile.get('dataset')!=spec['dataset']
            or profile.get('device')!='cpu' or profile.get('default_python_family')!=[3,13]
            or profile.get('candidate_updates')!=64 or profile.get('native_sac_updates')!=64
            or profile.get('recurrent_ppo_transitions')!=3000 or profile.get('recurrent_ppo_adam_steps')!=600):
        raise ValueError('actual fixed Colab profile identity/counts required')
    collect=higher_p95(profile['candidate_collection_seconds'],count=512)
    refresh=higher_p95(profile['actor_refresh512_seconds'],count=64)
    update=higher_p95(profile['candidate_sample_update_seconds'],count=64,discard=8)
    native=higher_p95(profile['native_sac_update_seconds'],count=64)
    ppo=higher_p95(profile['recurrent_ppo_rollout1000_seconds'],count=3)
    for values in (profile['validation_seconds'],profile['checkpoint_io_seconds']):
        if (type(values) is not list or len(values)!=3
                or any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in values)):
            raise ValueError('three finite actual validation/checkpoint timings required')
    validation=max(profile['validation_seconds'])
    checkpoint=max(profile['checkpoint_io_seconds'])
    common=10*validation+11*checkpoint
    work=dict(gru_sac=100000*(collect+refresh)+99000*update+common,
              native_sac=100000*collect+99000*native+common,
              recurrent_ppo_reference=100*ppo+common)
    projected={arm:1.5*seconds+600 for arm,seconds in work.items()}
    required={arm:seconds+contract['allocation_setup_reserve_seconds']+
              contract['allocation_teardown_reserve_seconds'] for arm,seconds in projected.items()}
    passes={arm:seconds<=contract['allocation_deadline_seconds'] for arm,seconds in required.items()}
    return dict(profile_projection_passed=all(passes.values()),arm_passes=passes,
                projected_work_seconds=projected,required_allocation_seconds=required,
                performance_interpretation_allowed=False,physical_profile_acceptance_still_required=True,
                original_project_ceiling=150,minimum_account_balance=150)
