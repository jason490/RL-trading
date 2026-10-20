"""Prepare one bounded CPU feasibility task without allocating a runtime."""
import argparse,json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import signal_plan,forest_plan
from rl_stocks.training.supervised.spec import SIGNAL_DATASET

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--forest',action='store_true');a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps([{'id':'forest-feasibility' if a.forest else 'signal-feasibility','dataset':'data/signals/'+SIGNAL_DATASET,'plan':forest_plan() if a.forest else signal_plan()}],indent=2)+'\n')
    print('Prepared1 CPU task with6 final fits; no runtime allocated.')
