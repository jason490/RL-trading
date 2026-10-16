"""Diagnose deterioration using already retrieved validation outputs only."""
import argparse
import hashlib
import json
from pathlib import Path
from rl_stocks.reporting.deterioration import diagnose_pair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();records=[]
    for root in args.roots:
        for path in sorted((root/'runs').glob('*/metadata.json')):
            paths=[path,path.parent/'selected-validation.json',path.parent/'validation.json']
            values=[json.loads(p.read_text()) for p in paths]
            record=diagnose_pair(*values)
            record['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            records.append(record)
    if not records:raise ValueError('no retrieved runs found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'split':'validation','records':records},indent=2)+'\n')
    print('Diagnosed',len(records),'saved validation pairs; no training or test evaluation.')


if __name__=='__main__':main()
