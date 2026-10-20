"""Prepare one bounded CPU feasibility task without allocating a runtime."""
import argparse,json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import signal_plan
from rl_stocks.training.supervised.spec import SIGNAL_DATASET

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps([{'id':'signal-feasibility','dataset':'data/signals/'+SIGNAL_DATASET,'plan':signal_plan()}],indent=2)+'\n')
    print('Prepared1 CPU task:6 final MLP fits, no runtime allocated.')
