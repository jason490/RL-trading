import argparse
import json
from pathlib import Path
from rl_stocks.reporting.stability import report_stability
from rl_stocks.reporting.study import verified_sources


def main():
    parser=argparse.ArgumentParser(description='Report fixed CPU algorithm stability cells; validation only.')
    parser.add_argument('roots',type=Path,nargs='+')
    parser.add_argument('--protocol',type=Path,default=Path('configs/experiments/algorithm_stability.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runs=[json.loads(p.read_text()) for root in args.roots for p in (root/'runs').glob('*/metadata.json')]
    value=report_stability(runs,json.loads(args.protocol.read_text()),verified_sources(runs,Path.cwd()),args.output)
    print('Completed cells:',value['completed_cells'],'complete:',value['complete'])


if __name__=='__main__':main()
