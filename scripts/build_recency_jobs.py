"""Prepare fixed history tasks only after required smoke/runtime acceptance."""
import argparse,json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import recency_plan
from rl_stocks.training.training_history import protocol,make_runtime_profile,PROTOCOL_SHA256


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int,default=11);p.add_argument('--screen-summary',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();spec=protocol()
    if a.seed!=11 and not a.smoke:
        if a.screen_summary is None:p.error('confirmation requires passing fixed screen summary')
        result=json.loads(a.screen_summary.read_text())
        if result.get('protocol_sha256')!=PROTOCOL_SHA256 or result.get('screen',{}).get('passed') is not True:
            p.error('same-protocol passing screen required')
    profile=None if a.smoke else make_runtime_profile(Path.cwd())
    tasks=[dict(id=f'recency-{arm}-{a.seed}'+('-smoke' if a.smoke else ''),dataset='data/datasets/'+spec['datasets'][arm],
        plan=recency_plan(arm,a.seed,a.smoke,profile)) for arm in spec['arms']]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(tasks,indent=2)+'\n')
    if profile is not None:a.output.with_suffix('.runtime-profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    print('Prepared two CPU tasks; no allocation.')
