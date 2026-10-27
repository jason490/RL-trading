"""Prepare one registered CPU profile task; never allocates a session."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.jobs import recurrent_comparison_profile_plan

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=recurrent_comparison_profile_plan()
    task=dict(id='recurrent-comparison-profile',
              dataset='data/datasets/'+plan['dataset_contract']['identity'],plan=plan)
    write_once(args.output,(json.dumps([task],indent=2)+'\n').encode())
    print('Prepared fixed Colab CPU profile task; no allocation.')
