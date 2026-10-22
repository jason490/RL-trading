"""Build two fixed CPU update-guard tasks; no allocation."""
import argparse,json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import guard_plan
from rl_stocks.training.recurrent_guard import protocol

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=11);p.add_argument('--output',type=Path,required=True);a=p.parse_args();spec=protocol()
    tasks=[dict(id=f'guard-{arm}-{a.seed}'+('-smoke' if a.smoke else ''),dataset='data/datasets/'+spec['dataset'],plan=guard_plan(arm,a.seed,a.smoke)) for arm in spec['arms']]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(tasks,indent=2)+'\n');print('Prepared2 CPU tasks; no allocation.')
