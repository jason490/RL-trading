"""Build fixed CPU execution-observation tasks; no runtime allocation."""
import argparse
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.runtime.colab.jobs import observation_plan,architecture_plan,reward_plan,context_plan,CONTEXT_DATASETS
from rl_stocks.rl.ppo.architecture import VARIANTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('configs/experiments/execution_observations.json'))
    parser.add_argument('--phase',choices=('smoke','screen','confirm'),required=True)
    parser.add_argument('--screen',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    architecture=protocol.get('study_kind')=='policy_architecture'
    reward=protocol.get('study_kind')=='reward_objective'
    context=protocol.get('study_kind')=='trend_context'
    expected=list(CONTEXT_DATASETS) if context else ['net_log_equity','net_initial_equity_change'] if reward else list(VARIANTS) if architecture else ['legacy','execution_masked','execution_full']
    if protocol['arms']!=expected or protocol['steps_per_cell']!=50000:
        raise ValueError('unsupported execution-observation protocol')
    if context and {a:c['dataset'] for a,c in protocol['arm_config'].items()}!=CONTEXT_DATASETS:
        raise ValueError('fixed context dataset identities required')
    seeds=(11,)
    if args.phase=='confirm':
        if not args.screen:raise ValueError('passing screen required')
        screen=json.loads(args.screen.read_text())
        if screen['protocol_sha256']!=digest(protocol) or not screen['screen']['passed']:
            raise ValueError('matching passing screen required')
        seeds=(23,37)
    elif args.screen:raise ValueError('unexpected screen input')
    tasks=[{'id':f'{profile}-{seed}','dataset':'data/datasets/'+(protocol['arm_config'][profile]['dataset'] if context else protocol['dataset']),
            'plan':(context_plan if context else reward_plan if reward else architecture_plan if architecture else observation_plan)(profile,seed,args.phase=='smoke')}
           for seed in seeds for profile in protocol['arms']]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(tasks,indent=2)+'\n')
    print(f'Prepared {len(tasks)} {args.phase} tasks; no sessions allocated.')


if __name__=='__main__':main()
