"""One reviewed setup-corrected market replay, retaining the original allowance."""
import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import zipfile

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.runtime.colab.canonical_market_probe import (
    CanonicalMarketJob, PROTOCOL_SHA256, file_sha, prepare,
)

REVIEW_SHA256 = '6c45c5db914a5bfdb7ff4bafd29bca4e4a2914ed3d56a637bea8a7cbc66c5ef6'
CONTROLLER = 'src/rl_stocks/runtime/colab/canonical_market_probe.py'
ADDITIONS = {'src/rl_stocks/runtime/colab/inference_setup.py',
             'src/rl_stocks/runtime/colab/canonical_market_review.py'}


def review(root):
    value = json.loads((Path(root) / 'configs/experiments/canonical_market_probe_review.json').read_text())
    if digest(value) != REVIEW_SHA256 or value['base_protocol_sha256'] != PROTOCOL_SHA256:
        raise ValueError('fixed reviewed market execution contract required')
    return value


def _worker_contract(source):
    tree = ast.parse(source)
    names = {'run_worker', 'protocol', 'file_sha'}
    functions = {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in names}
    if set(functions) != names:
        raise ValueError('complete worker functions required for preflight reuse')
    imports = [ast.dump(n, include_attributes=False) for n in tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    constants = [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'PROTOCOL_SHA256' for t in n.targets)]
    if len(constants) != 1:
        raise ValueError('exact worker protocol constant required')
    return functions, imports, constants


def equivalent_sources(root, archive):
    """Prove the actual locally replayed numerical engine has not changed."""
    root = Path(root)
    with zipfile.ZipFile(archive) as stream:
        original = {name: stream.read(name) for name in stream.namelist()
                    if name.startswith('src/') and name.endswith('.py')}
    current = {str(p.relative_to(root)): p.read_bytes() for p in (root / 'src').rglob('*.py')}
    if (set(original)-set(current) or set(current)-set(original)-ADDITIONS
            or CONTROLLER not in original or CONTROLLER not in current):
        raise ValueError('unexpected numerical source additions/deletions')
    for name, content in original.items():
        if name == CONTROLLER:
            if _worker_contract(content) != _worker_contract(current[name]):
                raise ValueError('worker arithmetic or runtime contract changed; new local preflight required')
        elif current[name] != content:
            raise ValueError('numerical engine source changed; new local preflight required')
    return {'byte_identical_modules': len(original)-1,
            'worker_ast_contract_identical': True, 'added_modules': sorted(set(current)-set(original)),
            'original_inventory_sha256': digest({k: hashlib.sha256(v).hexdigest()
                                                for k, v in original.items()})}


def admit(root, ledger):
    root = Path(root)
    spec = review(root)
    failure_path = root / 'results/reference-inference-2026-11-02/market-probe-failure.json'
    if file_sha(failure_path) != spec['failure_receipt_sha256']:
        raise ValueError('original failed attempt receipt changed')
    failure = json.loads(failure_path.read_text())
    original_claim = json.loads((root / '.runtime/canonical-market-probe-v1-claim.json').read_text())
    if (failure['session'] != spec['failed_session'] or not failure['cleanup_verified']
            or original_claim['session'] != spec['failed_session']):
        raise ValueError('original failed claim and verified cleanup required')
    attempts = ledger.value['attempts']
    originals = [a for a in attempts if a['session'] == spec['failed_session']]
    if (len(originals) != 1 or originals[0]['status'] != 'stopped'
            or originals[0]['base_consumed'] != spec['shared_baseline_units']
            or originals[0]['charged'] < failure['attempt_charged_units']
            or any(a['status'] != 'stopped' for a in attempts)
            or ledger.value['budget_stopped']):
        raise BudgetStop('original charges and stopped owned state required; no budget-stop bypass')
    if ledger.value['consumed'] - spec['shared_baseline_units'] >= spec['maximum_total_engineering_units']:
        raise BudgetStop('original cumulative engineering allowance exhausted')
    original = root / '.runtime/canonical-market-probe-input-v1'
    if (file_sha(original / 'payload.zip') != spec['original_payload_sha256']
            or file_sha(original / 'local-preflight-acceptance.json') != spec['original_local_preflight_sha256']
            or file_sha(root / 'src/rl_stocks/runtime/colab/inference_setup.py') != spec['corrected_setup_module_sha256']):
        raise ValueError('original preflight or reviewed setup identity changed')
    return equivalent_sources(root, original / 'payload.zip')


class ReviewedCanonicalMarketJob(CanonicalMarketJob):
    kind = 'canonical-market-reviewed-execution-v1'

    def validate_input(self, inputs, expected):
        super().validate_input(inputs, expected)
        proof = admit(self.root, self.ledger)
        if inputs.get('review_sha256') != REVIEW_SHA256 or inputs.get('preflight_reuse') != proof:
            raise ValueError('review-bound source equivalence proof required')

    def guard_allowance(self, projected_seconds):
        super().guard_allowance(projected_seconds)
        spec = review(self.root)
        active = self.ledger.active
        projected = self.ledger.value['consumed'] - spec['shared_baseline_units'] + (
            self.ledger.clock()-active['last_time']+projected_seconds)*(active['rate']+.01)/3600
        if projected >= spec['maximum_total_engineering_units']:
            self.ledger.value['budget_stopped'] = True
            active['stop_reason'] = 'original shared market-engineering allowance exhausted'
            self.ledger.save()
            raise BudgetStop(active['stop_reason'])


def main():
    from rl_stocks.runtime.colab.budget import Ledger
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--input', default='.runtime/canonical-market-probe-reviewed-input-v1/input.json')
    parser.add_argument('--cli-python')
    args = parser.parse_args()
    root = Path.cwd()
    if args.prepare:
        proof = admit(root, Ledger(root / '.runtime/usage-ledger.json'))
        target = prepare(root, Path(args.input).parent)
        inputs = json.loads(target.read_text())
        inputs.update(review_sha256=REVIEW_SHA256, preflight_reuse=proof)
        target.write_bytes(canonical(inputs))
        print(target)
        return
    if not args.cli_python:
        parser.error('--cli-python required for allocation')
    inputs = json.loads(Path(args.input).read_text())
    expected = json.loads((root / 'artifacts/canonical-market-replay-v1/result.json').read_text())
    with (root / '.runtime/allocation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ReviewedCanonicalMarketJob(root, args.cli_python, 'CPU').run_probe(
            inputs, expected, root / '.runtime/canonical-market-probe-reviewed-v1-claim.json')


if __name__ == '__main__':
    main()
