"""Audit existing training coverage before changing the stock universe."""
import argparse,csv,json
from pathlib import Path
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.reporting.regimes import regime_coverage


def main():
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--universe',default='configs/universes/sector_research.json');p.add_argument('--output',required=True);a=p.parse_args()
    v=regime_coverage(read_dataset(a.dataset),json.loads(Path(a.universe).read_text()));v['dataset']=Path(a.dataset).name
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(v,indent=2)+'\n')
    with out.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(v['records'][0]));writer.writeheader();writer.writerows(v['records'])
    print('Training symbols with declining trailing windows:',v['symbols_with_declining_both_windows'],'of',len(v['records']))
    print('Historical membership and delisting returns remain unverified.')

if __name__=='__main__':main()
