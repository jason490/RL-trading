"""Build compact validation learning graphs from completed research cells."""
import argparse,json
from pathlib import Path
from rl_stocks.reporting.learning import report_learning


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('roots',nargs='+')
    parser.add_argument('--protocol',default='configs/experiments/learning_study.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--families',nargs='+',choices=('baseline','combined'))
    args=parser.parse_args()
    runs=[]
    for root in map(Path,args.roots):
        for path in (root/'runs').glob('*/metadata.json'):
            record=json.loads(path.read_text())
            if record.get('status')=='complete':runs.append(record)
    value=report_learning(runs,json.loads(Path(args.protocol).read_text()),args.output,args.families)
    print('Completed cells:',value['completed_cells'],'stage complete:',value['complete'],
          'validation improvement gate:',value['improvement_gate_passed'])


if __name__=='__main__':main()
