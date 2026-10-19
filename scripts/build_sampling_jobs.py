"""Prepare bounded sampler-study tasks; confirmation requires a passing screen."""
import argparse
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.runtime.colab.jobs import SAMPLING_ARMS,sampling_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/experiments/causal_sampling.json'))
    p.add_argument('--phase',choices=('smoke','screen','confirm'),required=True)
    p.add_argument('--screen',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text())
    if protocol['arms']!=list(SAMPLING_ARMS) or protocol['steps_per_cell']!=50000 or protocol['initialization']!='wide':
        raise ValueError('unsupported fixed sampler study')
    if a.phase=='confirm':
        if not a.screen:raise ValueError('passing screen required')
        screen=json.loads(a.screen.read_text())
        if screen['protocol_sha256']!=digest(protocol) or not screen['screen']['passed']:
            raise ValueError('matching passing screen required')
        arms=(protocol['reference_arm'],screen['screen']['promoted_candidate']);seeds=(23,37)
    else:
        if a.screen:raise ValueError('screen evidence applies only to confirmation')
        arms=SAMPLING_ARMS;seeds=(11,)
    tasks=[{'id':f'{arm}-{seed}','dataset':'data/datasets/'+protocol['arm_config'][arm]['dataset'],
            'plan':sampling_plan(arm,seed,a.phase=='smoke')} for seed in seeds for arm in arms]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(tasks,indent=2)+'\n')
    print('Prepared',len(tasks),a.phase,'tasks; no sessions allocated.')


if __name__=='__main__':main()
