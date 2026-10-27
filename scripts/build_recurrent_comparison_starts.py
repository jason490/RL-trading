"""Create or independently accept unfitted comparison starts; never trains."""
import argparse
import json
from pathlib import Path

from rl_stocks.training.recurrent_comparison_starts import build, accept

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('destination',type=Path)
parser.add_argument('--accept-only',action='store_true')
args=parser.parse_args()
if not args.accept_only:
    build(args.destination)
print(json.dumps(accept(args.destination),sort_keys=True))
