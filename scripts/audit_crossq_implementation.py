"""Persist fixed untrained implementation evidence; no provider access or training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from rl_stocks.core.files import write_once
from rl_stocks.core.provenance import repository_state, source_identity
from rl_stocks.training.crossq_inspection import ROOT, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('audit output exists; do not overwrite prior evidence')
    ledger = ROOT / '.runtime/usage-ledger.json'
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    source = repository_state(ROOT)
    if source['dirty']:
        raise ValueError('commit implementation before recording actual audit evidence')
    result = audit()
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
    result.update(source_revision=source['revision'], source_hash=source_identity(),
                  python=sys.version.split()[0], ledger_unchanged=True,
                  ledger_sha256=before, project_consumed=json.loads(ledger.read_text())['consumed'])
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print('Untrained SAC/CrossQ geometry and serialization accepted; no training, data or Colab allocation.')


if __name__ == '__main__':
    main()
