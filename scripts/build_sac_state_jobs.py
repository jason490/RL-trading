"""Prepare fixed SAC state tasks without allocating Colab."""
import argparse,json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import sac_state_plan
from rl_stocks.training.sac_state import protocol

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=11);p.add_argument('--output',type=Path,required=True);a=p.parse_args();spec=protocol()
    tasks=[dict(id=f'sac-state-{arm}-{a.seed}'+('-smoke' if a.smoke else ''),dataset='data/datasets/'+spec['dataset'],plan=sac_state_plan(arm,a.seed,a.smoke)) for arm in spec['arms']]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(tasks,indent=2)+'\n');print('Prepared3 CPU tasks; no allocation.')
