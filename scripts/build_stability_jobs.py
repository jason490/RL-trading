"""Create reviewed CPU pool inputs for smoke or fixed algorithm comparisons."""
import argparse
import json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import single_smoke_plan,stability_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('configs/experiments/algorithm_stability.json'))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=json.loads(args.protocol.read_text())
    if p['steps']!=100000 or p['device']!='cpu' or p['reward_scale']!=1.:
        raise ValueError('unsupported fixed stability protocol')
    tasks=[]
    for seed in ([11] if args.smoke else p['seeds']):
        for algorithm in p['algorithms']:
            tasks.append({'id':algorithm+'-'+str(seed),'dataset':'data/datasets/'+p['dataset'],
                'plan':single_smoke_plan(algorithm) if args.smoke else stability_plan(algorithm,seed)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(tasks,indent=2)+'\n')
    print('Prepared',len(tasks),'CPU tasks; no sessions allocated.')


if __name__=='__main__':main()
