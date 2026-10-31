"""Read-only cached type audit; never backdate retrospective classifications."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.task_conditioning import audit_type_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new audit output required')
    protocol_path = Path('configs/experiments/task_conditioning.json')
    protocol = json.loads(protocol_path.read_text())
    source = Path('data/datasets') / protocol['dataset']
    protected = {str(p): checksum(p) for p in
                 (source / 'dataset.json', protocol_path, Path('.runtime/usage-ledger.json'))}
    data = read_dataset(source)
    result = audit_type_evidence(data)
    if (result['dataset_identity'] != protocol['dataset']
            or result['normalizer_sha256'] != protocol['normalizer_sha256']
            or result['stocks'] != 107 or result['etfs'] != 4):
        raise ValueError('fixed source identity/group counts changed')
    if not all(checksum(Path(path)) == value for path, value in protected.items()):
        raise ValueError('protected source changed')
    result['protected_sha256'] = protected
    result['protected_sources_unchanged'] = True
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print('Audited107 stocks/4 ETFs; historical type evidence not yet admitted')


if __name__ == '__main__':
    main()
