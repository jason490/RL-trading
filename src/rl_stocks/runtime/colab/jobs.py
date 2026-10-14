"""Validated remote job plans, with smoke acceptance before research training."""

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
