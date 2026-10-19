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
    if plan.get('version')==7:
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
