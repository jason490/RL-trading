"""Build fixed historical augmentation tasks; never allocate on construction."""
import argparse,json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import history_plan
from rl_stocks.training.history.spec import protocol

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=11);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    tasks=[dict(id=f'history-{arm}-{a.seed}'+('-smoke' if a.smoke else ''),dataset='data/datasets/'+identity,plan=history_plan(arm,a.seed,a.smoke)) for arm,identity in protocol()['datasets'].items()]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(tasks,indent=2)+'\n');print('Prepared',len(tasks),'CPU tasks; no allocation.')
