"""Compare immutable research commits without importing or training either source."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from rl_stocks.core.files import write_once
from rl_stocks.training.crossq_spec import PROTOCOL_SHA256, protocol


SCOPES = ('src/rl_stocks/core', 'src/rl_stocks/environment', 'src/rl_stocks/features',
          'src/rl_stocks/risk', 'src/rl_stocks/evaluation', 'src/rl_stocks/rl/sac',
          'src/rl_stocks/data/generation/datasets.py', 'src/rl_stocks/rl/registry',
          'src/rl_stocks/training/runner.py', 'src/rl_stocks/training/curves.py',
          'src/rl_stocks/training/checkpoint_state.py', 'requirements.lock')


def files_at(root, revision):
    raw = subprocess.check_output(['git', 'archive', revision, *SCOPES], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers()
                if m.isfile() and (m.name.endswith('.py') or m.name == 'requirements.lock')}


def function(source, name, parent=None):
    tree = ast.parse(source)
    if parent:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent)
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def dump(value):
    return ast.dump(value, include_attributes=False)


def calls(node, name):
    return [dump(n) for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def audit(root):
    root = Path(root)
    historical = protocol()['historical_control_reproduction']['source_revision']
    proof = json.loads((root / 'results/crossq-integrity-2026-10-27/cleanup.json').read_text())['source_proof']
    current = proof['source_revision']
    left, right = files_at(root, historical), files_at(root, current)
    common = sorted(left.keys() & right.keys())
    records = {name: dict(historical_sha256=hashlib.sha256(left[name]).hexdigest(),
                         current_sha256=hashlib.sha256(right[name]).hexdigest(),
                         bytes_equal=left[name] == right[name]) for name in common}
    factory = 'src/rl_stocks/rl/registry/factory.py'
    runner = 'src/rl_stocks/training/runner.py'
    train_left, train_right = function(left[runner], 'train'), function(right[runner], 'train')
    evidence = dict(
        factory_create_ast_equal=dump(function(left[factory], 'create')) == dump(function(right[factory], 'create')),
        callback_on_step_ast_equal=dump(function(left[runner], '_on_step', 'ValidationCheckpoint')) ==
                                   dump(function(right[runner], '_on_step', 'ValidationCheckpoint')),
        environment_constructor_calls_equal=calls(train_left, 'TradingEnv') == calls(train_right, 'TradingEnv'),
        environment_constructor_call_counts=[len(calls(train_left, 'TradingEnv')), len(calls(train_right, 'TradingEnv'))],
        intraop_thread_calls_equal=calls(train_left, 'torch.set_num_threads') == calls(train_right, 'torch.set_num_threads'),
        historical_intraop_thread_calls=calls(train_left, 'torch.set_num_threads'),
        current_intraop_thread_calls=calls(train_right, 'torch.set_num_threads'),
        seed_calls_equal=all(calls(train_left, name) == calls(train_right, name)
                             for name in ('random.seed', 'np.random.seed', 'torch.manual_seed')),
    )
    return dict(schema='crossq-scoped-source-audit-v1', protocol_sha256=PROTOCOL_SHA256,
                historical_revision=historical, screen_revision=current, scopes=list(SCOPES),
                common_files=records, added_files=sorted(right.keys() - left.keys()),
                removed_files=sorted(left.keys() - right.keys()),
                changed_common_files=[name for name in common if not records[name]['bytes_equal']],
                unchanged_common_file_count=sum(r['bytes_equal'] for r in records.values()),
                static_call_evidence=evidence,
                limits='Scoped byte/AST comparison only; runner/registry changes remain. Does not prove runtime/backend or full semantic equivalence.',
                root_cause_established=False, training_performed=False, optimizer_steps=0,
                allocation_performed=False, retry_authorized=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(Path.cwd())
    write_once(args.output, (json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({k: result[k] for k in ('unchanged_common_file_count', 'changed_common_files', 'static_call_evidence')}))
