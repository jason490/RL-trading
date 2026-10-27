"""Prepare the single reviewed profile task; authorization and allocation stay separate."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.jobs import recurrent_reviewed_profile_plan

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=recurrent_reviewed_profile_plan()
    task=dict(id='recurrent-corrected-profile',dataset='data/datasets/'+plan['dataset_contract']['identity'],plan=plan)
    write_once(args.output,(json.dumps([task],indent=2)+'\n').encode())
    print('Prepared one corrected CPU profile task. Explicit review authorization still required; no allocation.')
