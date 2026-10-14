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
    if research_algorithm and accelerator != "A100":
        raise ValueError("full research training requires A100")
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
        jobs.append({"algorithm": research_algorithm, "device": "cuda",
                     "smoke": False, "steps": 50000})
    return {"version": 2, "seed": 11, "accelerator": accelerator, "jobs": jobs}


def study_plan():
    """One dataset, matched algorithms/seeds; budget watchdog owns the allocation."""
    smoke = [{**job, "seed": 11} for job in job_plan(accelerator="T4")["jobs"]]
    research = [{"algorithm": algorithm, "device": "cuda", "smoke": False,
                 "steps": 50000, "seed": seed}
                for algorithm in ("ppo", "sac", "td3") for seed in (11, 23, 37)]
    return {"version": 3, "accelerator": "A100", "jobs": smoke+research}


def validate_plan(plan):
    if plan.get("version") == 3:
        if plan != study_plan():
            raise ValueError("unsupported matched study plan")
    elif plan.get("version") == 2:
        jobs = plan["jobs"]
        research = jobs[-1]["algorithm"] if not jobs[-1]["smoke"] else None
        if plan != job_plan(research, plan["accelerator"]):
            raise ValueError("unsupported remote job plan")
    else:
        raise ValueError("unsupported plan version")
    return plan
