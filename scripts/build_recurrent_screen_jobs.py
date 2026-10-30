"""Prepare the fixed three-role seed11 CPU screen, without allocation."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.jobs import recurrent_screen_plan
from rl_stocks.training.recurrent_comparison import protocol

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tasks=[dict(id='recurrent-screen-'+arm+'-11',dataset='data/datasets/'+protocol()['dataset'],
                plan=recurrent_screen_plan(arm)) for arm in protocol()['arms']]
    write_once(args.output,(json.dumps(tasks,indent=2)+'\n').encode())
    print('Prepared three100k seed11 CPU screen cells. No allocation; accepted-profile/budget preflight required.')
