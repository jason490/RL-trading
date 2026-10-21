"""Validated remote job plans, with smoke acceptance before research training."""
from pathlib import Path


def bootstrap_environment(parent, root):
    root = Path(root)
    content = root.parent
    environment = dict(parent, UV_CACHE_DIR=str(content/'uv-cache'),
                       MPLCONFIGDIR=str(content/'matplotlib'), MPLBACKEND='Agg',
                       PYTHONPATH=str(root/'src'),
                       RL_CHECKPOINT_ARCHIVE=str(content/'research-checkpoint.tar.gz'))
    environment.pop('PYTHONHOME', None)
    environment.pop('VIRTUAL_ENV', None)
    return environment


def bootstrap_commands(uv, root, kernel_python):
    root = Path(root)
    python = str(root / '.venv/bin/python')
    return [
        [uv, 'venv', '--python', kernel_python, str(root / '.venv')],
        [uv, 'pip', 'sync', '--python', python, str(root / 'requirements.lock')],
        [uv, 'pip', 'install', '--python', python, '--no-deps', '-e', str(root)],
        [python, str(root / 'scripts/remote_train.py')],
    ]


def job_plan(research_algorithm=None, accelerator="A100"):
    if research_algorithm not in (None, "ppo", "sac", "td3"):
        raise ValueError("unknown research algorithm")
    if accelerator not in ("CPU", "T4", "A100"):
        raise ValueError("unsupported runtime")
    jobs = [
        {"algorithm": "ppo", "device": "cpu", "smoke": True, "steps": 512},
        {"algorithm": "ppo", "device": "cuda", "smoke": True, "steps": 512},
        {"algorithm": "sac", "device": "cuda", "smoke": True, "steps": 1024},
        {"algorithm": "td3", "device": "cuda", "smoke": True, "steps": 1024},
    ]
    if accelerator == "CPU":
        jobs = [{"algorithm": algorithm, "device": "cpu", "smoke": True, "steps": steps}
                for algorithm, steps in (("ppo",512),("sac",1024),("td3",1024))]
    if research_algorithm:
        jobs.append({"algorithm": research_algorithm, "device": "cpu" if accelerator=="CPU" else "cuda",
                     "smoke": False, "steps": 50000})
    return {"version": 2, "seed": 11, "accelerator": accelerator, "jobs": jobs}


def study_plan(algorithm=None, accelerator="A100", seeds=None):
    """One dataset, matched algorithms/seeds; budget watchdog owns the allocation."""
    if algorithm not in (None,"ppo","sac","td3"):
        raise ValueError("unsupported study algorithm")
    if accelerator not in ("CPU","T4","A100"):
        raise ValueError("unsupported runtime")
    selected=(11,23,37) if seeds is None else tuple(seeds)
    if (not selected or selected!=tuple(s for s in (11,23,37) if s in selected)
            or seeds is not None and algorithm is None):
        raise ValueError("study seeds require one algorithm and an ordered unique subset of 11/23/37")
    smoke = [{**job, "seed": 11} for job in job_plan(accelerator="CPU" if accelerator=="CPU" else "T4")["jobs"]] if algorithm is None else []
    research = [{"algorithm": algorithm, "device": "cpu" if accelerator=="CPU" else "cuda", "smoke": False,
                 "steps": 50000, "seed": seed}
                for algorithm in ((algorithm,) if algorithm else ("ppo", "sac", "td3")) for seed in selected]
    plan={"version": 3, "accelerator": accelerator, "jobs": smoke+research}
    if algorithm is not None:plan["study_algorithm"]=algorithm
    if seeds is not None:plan["study_seeds"]=list(selected)
    return plan


def validate_plan(plan):
    if plan.get('version')==21:
        try:expected=discount_plan(**plan['discount_study'])
        except (KeyError,TypeError):raise ValueError('malformed discount plan') from None
        if plan!=expected:raise ValueError('unsupported discount plan')
    elif plan.get('version')==20:
        try:expected=regularization_plan(**plan['regularization_study'])
        except (KeyError,TypeError):raise ValueError('malformed regularization plan') from None
        if plan!=expected:raise ValueError('unsupported regularization plan')
    elif plan.get('version')==19:
        try:expected=diversity_plan(**plan['diversity_study'])
        except (KeyError,TypeError):raise ValueError('malformed diversity plan') from None
        if plan!=expected:raise ValueError('unsupported diversity plan')
    elif plan.get('version')==18:
        try:expected=td3_state_plan(**plan['td3_state_study'])
        except (KeyError,TypeError):raise ValueError('malformed TD3 state plan') from None
        if plan!=expected:raise ValueError('unsupported TD3 state plan')
    elif plan.get('version')==17:
        try:expected=temperature_plan(**plan['temperature_study'])
        except (KeyError,TypeError):raise ValueError('malformed temperature plan') from None
        if plan!=expected:raise ValueError('unsupported temperature plan')
    elif plan.get('version')==16:
        try:expected=sac_state_plan(**plan['sac_state_study'])
        except (KeyError,TypeError):raise ValueError('malformed SAC state plan') from None
        if plan!=expected:raise ValueError('unsupported SAC state plan')
    elif plan.get('version')==15:
        try:expected=history_plan(**plan['history_study'])
        except (KeyError,TypeError):raise ValueError('malformed historical study plan') from None
        if plan!=expected:raise ValueError('unsupported historical study plan')
    elif plan.get('version')==14:
        if plan!=forest_plan():raise ValueError('unsupported forest-signal plan')
    elif plan.get('version')==13:
        if plan!=signal_plan():raise ValueError('unsupported supervised-signal plan')
    elif plan.get('version')==12:
        try:expected=context_plan(**plan['context_study'])
        except (KeyError,TypeError):raise ValueError('malformed trend-context study plan') from None
        if plan!=expected:raise ValueError('unsupported trend-context study plan')
    elif plan.get('version')==11:
        try:expected=reward_plan(**plan['reward_study'])
        except (KeyError,TypeError):raise ValueError('malformed reward study plan') from None
        if plan!=expected:raise ValueError('unsupported reward study plan')
    elif plan.get('version')==10:
        try:expected=architecture_plan(**plan['architecture_study'])
        except (KeyError,TypeError):raise ValueError('malformed policy-architecture study plan') from None
        if plan!=expected:raise ValueError('unsupported policy-architecture study plan')
    elif plan.get('version')==9:
        try:expected=observation_plan(**plan['observation_study'])
        except (KeyError,TypeError):raise ValueError('malformed execution-observation study plan') from None
        if plan!=expected:raise ValueError('unsupported execution-observation study plan')
    elif plan.get('version')==8:
        try:expected=sampling_plan(**plan['sampling_study'])
        except (KeyError,TypeError):raise ValueError('malformed causal-sampling study plan') from None
        if plan!=expected:raise ValueError('unsupported causal-sampling study plan')
    elif plan.get('version')==7:
        try:expected=initialization_plan(**plan['initialization_study'])
        except (KeyError,TypeError):raise ValueError('malformed initialization study plan') from None
        if plan!=expected:raise ValueError('unsupported initialization study plan')
    elif plan.get('version')==6:
        try:expected=single_smoke_plan(plan['smoke_algorithm'])
        except (KeyError,TypeError):raise ValueError('malformed single smoke plan') from None
        if plan!=expected:raise ValueError('unsupported single smoke plan')
    elif plan.get('version')==5:
        try:expected=stability_plan(**plan['stability_study'])
        except (KeyError,TypeError):raise ValueError('malformed stability study plan') from None
        if plan!=expected:raise ValueError('unsupported stability study plan')
    elif plan.get('version')==4:
        try:expected=learning_plan(**plan['learning_study'])
        except (KeyError,TypeError):raise ValueError('malformed learning study plan') from None
        if plan!=expected:raise ValueError('unsupported learning study plan')
    elif plan.get("version") == 3:
        if plan != study_plan(plan.get("study_algorithm"),plan.get("accelerator"),plan.get("study_seeds")):
            raise ValueError("unsupported matched study plan")
    elif plan.get("version") == 2:
        jobs = plan["jobs"]
        research = jobs[-1]["algorithm"] if not jobs[-1]["smoke"] else None
        if plan != job_plan(research, plan["accelerator"]):
            raise ValueError("unsupported remote job plan")
    else:
        raise ValueError("unsupported plan version")
    return plan


def learning_plan(seed=11,steps=100000,reward_scale=1.):
    """One bounded CPU SAC cell; learning budgets are deliberately finite."""
    if (type(seed) is not int or type(steps) is not int or type(reward_scale) not in (int,float)
            or seed not in (11,23,37) or steps not in (50000,100000,250000) or reward_scale not in (1.,10.,100.)):
        raise ValueError('unsupported learning-study seed, steps or reward scale')
    milestones=[s for s in (50000,100000,250000) if s<=steps]
    specification={'seed':seed,'steps':steps,'reward_scale':float(reward_scale)}
    job={'algorithm':'sac','device':'cpu','smoke':False,'seed':seed,'steps':steps,
         'evaluation_every':10000,'compact_checkpoints':True,'milestones':milestones,'reward_scale':float(reward_scale)}
    return {'version':4,'accelerator':'CPU','learning_study':specification,'jobs':[job]}


def stability_plan(algorithm,seed):
    """A fixed 100k CPU cell for the separate PPO/SAC stability comparison."""
    if algorithm not in ('ppo','sac','recurrent_ppo') or type(seed) is not int or seed not in (11,23,37):
        raise ValueError('unsupported stability algorithm or seed')
    job={'algorithm':algorithm,'device':'cpu','smoke':False,'seed':seed,'steps':100000,
         'evaluation_every':10000,'compact_checkpoints':True,'milestones':[100000],'reward_scale':1.}
    return {'version':5,'accelerator':'CPU','stability_study':{'algorithm':algorithm,'seed':seed},'jobs':[job]}


def single_smoke_plan(algorithm):
    if algorithm not in ('ppo','sac','td3','recurrent_ppo'):raise ValueError('unsupported smoke algorithm')
    steps=512 if algorithm in ('ppo','recurrent_ppo') else 1024
    return {'version':6,'accelerator':'CPU','smoke_algorithm':algorithm,
        'jobs':[{'algorithm':algorithm,'device':'cpu','smoke':True,'seed':11,'steps':steps}]}


def initialization_plan(arm,seed=11,smoke=False):
    from rl_stocks.training.initialization import initialization_spec
    initialization_spec(arm,'ppo')
    if type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported initialization seed or smoke mode')
    steps=512 if smoke else 50000
    return {'version':7,'accelerator':'CPU',
        'initialization_study':{'arm':arm,'seed':seed,'smoke':smoke},
        'jobs':[{'algorithm':'ppo','device':'cpu','smoke':smoke,'seed':seed,'steps':steps,
            'evaluation_every':256 if smoke else 10000,'compact_checkpoints':True,
            'milestones':[steps],'reward_scale':1.,'initialization':arm}]}


SAMPLING_ARMS=('small_uniform','small_balanced','large_uniform','large_balanced')


def sampling_plan(arm,seed=11,smoke=False):
    if arm not in SAMPLING_ARMS or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported causal-sampling arm, seed or smoke mode')
    steps=512 if smoke else 50000
    mode='ready_uniform' if arm.endswith('uniform') else 'downtrend_balanced'
    return {'version':8,'accelerator':'CPU','sampling_study':{'arm':arm,'seed':seed,'smoke':smoke},
        'jobs':[{'algorithm':'ppo','device':'cpu','smoke':smoke,'seed':seed,'steps':steps,
            'evaluation_every':256 if smoke else 10000,'compact_checkpoints':True,
            'milestones':[steps],'reward_scale':1.,'initialization':'wide','start_sampling':mode}]}


def observation_plan(profile,seed=11,smoke=False):
    from rl_stocks.environment.observations.state import PROFILES
    if profile not in PROFILES or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported execution-observation profile, seed or smoke mode')
    steps=512 if smoke else 50000
    return {'version':9,'accelerator':'CPU',
        'observation_study':{'profile':profile,'seed':seed,'smoke':smoke},
        'jobs':[{'algorithm':'ppo','device':'cpu','smoke':smoke,'seed':seed,'steps':steps,
            'evaluation_every':256 if smoke else 10000,'compact_checkpoints':True,
            'milestones':[steps],'reward_scale':1.,'initialization':'wide',
            'start_sampling':'legacy_uniform','observation_profile':profile}]}


def architecture_plan(variant,seed=11,smoke=False):
    from rl_stocks.rl.ppo.architecture import VARIANTS
    if variant not in VARIANTS or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported architecture variant, seed or smoke mode')
    steps=512 if smoke else 50000
    return {'version':10,'accelerator':'CPU',
        'architecture_study':{'variant':variant,'seed':seed,'smoke':smoke},
        'jobs':[{'algorithm':'ppo','device':'cpu','smoke':smoke,'seed':seed,'steps':steps,
            'evaluation_every':256 if smoke else 10000,'compact_checkpoints':True,
            'milestones':[steps],'reward_scale':1.,'initialization':'wide',
            'start_sampling':'legacy_uniform','observation_profile':'execution_full','policy_variant':variant}]}


def reward_plan(kind, seed=11, smoke=False):
    from rl_stocks.environment.rewards.equity import validate_reward_kind
    validate_reward_kind(kind)
    if type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed != 11:
        raise ValueError('unsupported reward study seed or smoke mode')
    steps = 512 if smoke else 50000
    return {'version':11, 'accelerator':'CPU',
        'reward_study':{'kind':kind,'seed':seed,'smoke':smoke},
        'jobs':[{'algorithm':'ppo','device':'cpu','smoke':smoke,'seed':seed,'steps':steps,
            'evaluation_every':256 if smoke else 10000,'compact_checkpoints':True,
            'milestones':[steps],'reward_scale':1.,'initialization':'wide',
            'start_sampling':'legacy_uniform','observation_profile':'execution_full',
            'policy_variant':'native','reward_kind':kind}]}


CONTEXT_DATASETS = {
    'matched_baseline':'fcde438c6423ec15f8d42fd6d7afe08deb0e434653bead517757d42be1ce4546',
    'masked':'d603bad1d698d53ac9aadf7890c84c055a4b4b7e51eff6033ab0270420f22ae3',
    'full':'830eba077a761a2de72b3935f9cf91b1b2694ecc51a888943f4d0cb30d124eed'}


def context_plan(arm, seed=11, smoke=False):
    if arm not in CONTEXT_DATASETS or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported trend-context arm, seed or smoke mode')
    plan=reward_plan('net_log_equity',seed,smoke)
    plan.pop('reward_study')
    plan.update(version=12,context_study={'arm':arm,'seed':seed,'smoke':smoke},
        dataset_contract={'identity':CONTEXT_DATASETS[arm],'mode':arm,
            'market_features':7 if arm=='matched_baseline' else 10,
            'schema':'daily-indicators-v1-trend-context-v1-'+arm})
    return plan


def validate_context_payload(data, plan):
    from rl_stocks.features.registry.assemble import NAMES as BASE
    from rl_stocks.features.trend.context import NAMES as EXTRA
    contract=plan['dataset_contract'];mode=contract['mode']
    names=list(BASE)+(list(EXTRA) if mode!='matched_baseline' else [])
    if (data['names']!=names or len(names)!=contract['market_features']
        or data['schema']!=contract['schema']
        or data['manifest'].get('trend_context',{}).get('mode')!=mode
        or data['manifest'].get('included_splits')!=['train','validation']
        or data['manifest'].get('unverified_override') is not False
        or any(e['split'] not in ('train','validation') or any(b['session']>'2023-12-31' for b in e['bars']) for e in data['episodes'].values())):
        raise ValueError('trend-context dataset contract mismatch')


def signal_plan():
    from rl_stocks.training.supervised.spec import protocol,PROTOCOL_SHA256,SIGNAL_DATASET
    protocol()
    return {'version':13,'accelerator':'CPU','protocol_sha256':PROTOCOL_SHA256,
        'dataset_contract':{'identity':SIGNAL_DATASET,'kind':'completed-training-signal-v1'},
        'jobs':[{'algorithm':'signal_feasibility','device':'cpu','smoke':False,'seed':11,'steps':12002}]}


def forest_plan():
    from rl_stocks.training.supervised.forest_spec import protocol,PROTOCOL_SHA256
    from rl_stocks.training.supervised.spec import SIGNAL_DATASET
    protocol()
    return {'version':14,'accelerator':'CPU','protocol_sha256':PROTOCOL_SHA256,
        'dataset_contract':{'identity':SIGNAL_DATASET,'kind':'completed-training-signal-v1'},
        'jobs':[{'algorithm':'forest_signal_feasibility','device':'cpu','smoke':False,'seed':11,'steps':770}]}


def history_plan(arm,seed=11,smoke=False):
    from rl_stocks.training.history.spec import protocol,PROTOCOL_SHA256
    p=protocol()
    if arm not in p['arms'] or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported historical arm, seed or smoke mode')
    plan=reward_plan('net_log_equity',seed,smoke);plan.pop('reward_study')
    plan.update(version=15,history_study=dict(arm=arm,seed=seed,smoke=smoke),protocol_sha256=PROTOCOL_SHA256,
        dataset_contract=dict(identity=p['datasets'][arm],arm=arm))
    plan['jobs'][0]['start_sampling']='full_window_uniform'
    return plan


def sac_state_plan(arm,seed=11,smoke=False):
    from rl_stocks.training.sac_state import protocol,PROTOCOL_SHA256
    p=protocol()
    if arm not in p['arms'] or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported SAC state arm, seed or smoke mode')
    steps=p['smoke_steps'] if smoke else p['steps_per_cell']
    return dict(version=16,accelerator='CPU',sac_state_study=dict(arm=arm,seed=seed,smoke=smoke),
        protocol_sha256=PROTOCOL_SHA256,dataset_contract=dict(identity=p['dataset']),
        jobs=[dict(algorithm='sac',device='cpu',smoke=smoke,seed=seed,steps=steps,
            evaluation_every=p['smoke_evaluation_every'] if smoke else p['evaluation_every'],compact_checkpoints=True,
            milestones=[steps],reward_scale=1.,initialization='standard',start_sampling='legacy_uniform',
            observation_profile=arm,policy_variant='native',reward_kind='net_log_equity',study_protocol_sha256=PROTOCOL_SHA256)])


def temperature_plan(arm,seed=11,smoke=False):
    from rl_stocks.training.sac_temperature import protocol,PROTOCOL_SHA256
    p=protocol()
    if arm not in p['arms'] or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported temperature arm, seed or smoke mode')
    plan=sac_state_plan('execution_full',seed,smoke);plan.pop('sac_state_study')
    plan.update(version=17,temperature_study=dict(arm=arm,seed=seed,smoke=smoke),protocol_sha256=PROTOCOL_SHA256)
    plan['jobs'][0].update(study_protocol_sha256=PROTOCOL_SHA256,overrides={'ent_coef':p['entropy_coefficient'][arm]})
    return plan


def td3_state_plan(arm,seed=11,smoke=False):
    from rl_stocks.training.td3_state import protocol,PROTOCOL_SHA256
    p=protocol()
    if arm not in p['arms'] or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported TD3 state arm, seed or smoke mode')
    plan=sac_state_plan(arm,seed,smoke);plan.pop('sac_state_study')
    plan.update(version=18,td3_state_study=dict(arm=arm,seed=seed,smoke=smoke),protocol_sha256=PROTOCOL_SHA256)
    plan['jobs'][0].update(algorithm='td3',study_protocol_sha256=PROTOCOL_SHA256)
    return plan


def diversity_plan(arm,seed=11,smoke=False):
    from rl_stocks.training.diversity import protocol,PROTOCOL_SHA256
    p=protocol()
    if arm not in p['arms'] or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported diversity arm, seed or smoke mode')
    plan=sac_state_plan('execution_full',seed,smoke);plan.pop('sac_state_study')
    steps=p['smoke_steps'] if smoke else p['steps_per_cell']
    plan.update(version=19,diversity_study=dict(arm=arm,seed=seed,smoke=smoke),protocol_sha256=PROTOCOL_SHA256,dataset_contract=dict(identity=p['datasets'][arm]))
    plan['jobs'][0].update(steps=steps,milestones=[steps],start_sampling=p['start_sampling'],study_protocol_sha256=PROTOCOL_SHA256)
    return plan


def regularization_plan(arm,seed=11,smoke=False):
    from rl_stocks.training.sac_regularization import protocol,PROTOCOL_SHA256,overrides
    p=protocol()
    if arm not in p['arms'] or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported regularization arm/seed/smoke')
    plan=diversity_plan('expanded',seed,smoke);plan.pop('diversity_study')
    plan.update(version=20,regularization_study=dict(arm=arm,seed=seed,smoke=smoke),protocol_sha256=PROTOCOL_SHA256)
    plan['jobs'][0].update(study_protocol_sha256=PROTOCOL_SHA256,overrides=overrides(arm,smoke))
    return plan


def discount_plan(arm,seed=11,smoke=False):
    from rl_stocks.training.sac_discount import protocol,PROTOCOL_SHA256,overrides
    p=protocol()
    if arm not in p['arms'] or type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('unsupported discount arm/seed/smoke')
    plan=regularization_plan('control',seed,smoke);plan.pop('regularization_study')
    plan.update(version=21,discount_study=dict(arm=arm,seed=seed,smoke=smoke),protocol_sha256=PROTOCOL_SHA256,
                account_contract=dict(minimum_balance=150,teardown_balance=170))
    plan['jobs'][0].update(study_protocol_sha256=PROTOCOL_SHA256,overrides=overrides(arm,smoke))
    return plan
