"""Compare completed input families against the fixed 250k CPU baseline."""
import argparse,json
from pathlib import Path
from rl_stocks.reporting.indicators import report_indicators
from rl_stocks.reporting.study import verified_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('roots',nargs='+',help='Retrieved artifact roots with runs/metadata.json')
    parser.add_argument('--protocol',default='configs/experiments/indicator_followup.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();runs=[]
    for root in map(Path,args.roots):
        for p in (root/'runs').glob('*/metadata.json'):
            r=json.loads(p.read_text())
            if r.get('status')=='complete' and not r.get('smoke'):runs.append(r)
    proofs=verified_sources(runs,Path.cwd())
    value=report_indicators(runs,json.loads(Path(args.protocol).read_text()),proofs,args.output)
    print('Received',value['received_cells'],'of',value['expected_cells'],'cells; complete:',value['complete'])

if __name__=='__main__':main()
