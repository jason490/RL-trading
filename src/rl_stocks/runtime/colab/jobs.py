"""Validated remote job plans, with smoke acceptance before research training."""
from pathlib import Path


def bootstrap_commands(uv, root):
    root = Path(root)
    python = str(root / '.venv/bin/python')
    return [
        [uv, 'python', 'install', '--no-bin', '3.12.14'],
        [uv, 'venv', '--managed-python', '--python', '3.12.14', str(root / '.venv')],
        [uv, 'pip', 'sync', '--python', python, str(root / 'requirements.lock')],
        [uv, 'pip', 'install', '--python', python, '--no-deps', '-e', str(root)],
        [python, str(root / 'scripts/remote_train.py')],
    ]


def job_plan(research_algorithm=None):
    if research_algorithm not in (None, "ppo", "sac", "td3"):
        raise ValueError("unknown research algorithm")
    jobs = [
        {"algorithm": "ppo", "device": "cpu", "smoke": True, "steps": 512},
        {"algorithm": "ppo", "device": "cuda", "smoke": True, "steps": 512},
        {"algorithm": "sac", "device": "cuda", "smoke": True, "steps": 1024},
        {"algorithm": "td3", "device": "cuda", "smoke": True, "steps": 1024},
    ]
    if research_algorithm:
        jobs.append({"algorithm": research_algorithm, "device": "cuda",
                     "smoke": False, "steps": 50000})
    return {"version": 1, "seed": 11, "jobs": jobs}
