"""Build two fixed CPU SAC diversity tasks; no allocation."""
import argparse,json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import diversity_plan
from rl_stocks.training.diversity import protocol

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=11);p.add_argument('--output',type=Path,required=True);a=p.parse_args();spec=protocol()
    tasks=[dict(id=f'diversity-{arm}-{a.seed}'+('-smoke' if a.smoke else ''),dataset='data/datasets/'+spec['datasets'][arm],plan=diversity_plan(arm,a.seed,a.smoke)) for arm in spec['arms']]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(tasks,indent=2)+'\n');print('Prepared2 CPU tasks; no allocation.')
