"""Build the fixed actual cached smoke view; no model, provider or allocation."""
import argparse
import hashlib
import json
from pathlib import Path

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.training.recurrent_environment_spec import build_view, protocol, PROTOCOL_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', required=True, type=Path)
    args = parser.parse_args()
    if args.proof.exists():
        raise FileExistsError('actual view proof already exists')
    root = Path(__file__).resolve().parents[1]
    source_path = root/'data/datasets'/protocol()['source_dataset']
    protected = [source_path/'dataset.json', root/'.runtime/usage-ledger.json',
                 root/'.runtime/crossq-integrity-stop.json', root/'.runtime/recurrent-sac-smoke-allocation.json']
    def hashes():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    before = hashes()
    source = read_dataset(source_path)
    view = build_view(source)
    identity = digest(view)
    output = root/'data/datasets'/identity/'dataset.json'
    write_once(output, canonical(view))
    assert before == hashes()
    proof = dict(schema='recurrent-environment-view-preparation-v1', protocol_sha256=PROTOCOL_SHA256,
                 source_dataset=source_path.name, input_identity=identity,
                 normalizer_sha256=digest(view['normalizer']),
                 original_preprocessing_unchanged=view['normalizer'] == source['normalizer'],
                 exact_source_prefixes=True,
                 episode_rows={k: dict(symbol=ep['symbol'], split=ep['split'], rows=len(ep['bars']),
                                       first=ep['bars'][0]['session'], last=ep['bars'][-1]['session'])
                               for k, ep in view['episodes'].items()},
                 payload_bytes=output.stat().st_size, protected_sha256=before,
                 protected_files_unchanged=True, optimizer_steps=0, colab_allocations=0,
                 provider_requests=0, retired_test_access=False, performance_evaluation=False,
                 implementation_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                     ['src/rl_stocks/training/recurrent_environment_spec.py', 'scripts/build_recurrent_environment_view.py']})
    write_once(args.proof, canonical(proof))
    print(json.dumps({k: proof[k] for k in ['input_identity','normalizer_sha256','payload_bytes','protected_files_unchanged']},indent=2))


if __name__ == '__main__':
    main()
