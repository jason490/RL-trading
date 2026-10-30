"""Bounded local subprocess entry: physical inference-only verification and retrieval."""
import argparse
import json
from pathlib import Path
import tarfile

from rl_stocks.core.files import write_once
from rl_stocks.core.contracts import canonical
from rl_stocks.runtime.colab.recurrent_screen_artifacts import validate_screen_archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--dataset-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--retrieved',type=Path,required=True)
    args=parser.parse_args()
    proof=validate_screen_archive(args.archive,json.loads(args.plan.read_text()),args.dataset_id)
    if args.retrieved.exists():
        raise FileExistsError('fresh retrieved scientific directory required')
    args.retrieved.mkdir(parents=True)
    with tarfile.open(args.archive) as archive:
        archive.extractall(args.retrieved,filter='data')
    write_once(args.output,canonical(proof))
    print('Complete physical screen cell verified; no training performed locally.')


if __name__=='__main__':
    main()
