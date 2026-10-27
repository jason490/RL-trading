"""Read only retained task-owned replay artifacts to localize SAC reproduction drift."""
import argparse
import hashlib
import json
import io
import zipfile
from pathlib import Path
import pickle
import tarfile

import numpy as np

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.crossq_spec import PROTOCOL_SHA256, protocol


def field_difference(left, right):
    if (left.shape != right.shape or left.dtype != np.float32 or right.dtype != np.float32
            or left.ndim < 2 or len(left) == 0 or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError('same nonempty finite float32 filled-prefix geometry required')
    a, b = left.reshape(len(left), -1), right.reshape(len(right), -1)
    changed = a.view(np.uint32) != b.view(np.uint32)
    rows = np.flatnonzero(changed.any(axis=1))
    result = dict(compared_rows=len(a), differing_rows=len(rows),
                  first_different_zero_based_row=int(rows[0]) if len(rows) else None,
                  common_leading_rows=int(rows[0]) if len(rows) else len(a))
    if len(rows):
        i = int(rows[0])
        bits_a, bits_b = a[i].view(np.uint32), b[i].view(np.uint32)
        ordered_a = np.where(bits_a & 0x80000000, ~bits_a, bits_a | 0x80000000).astype(np.int64)
        ordered_b = np.where(bits_b & 0x80000000, ~bits_b, bits_b | 0x80000000).astype(np.int64)
        result.update(first_row_maximum_absolute_difference=float(np.max(np.abs(a[i].astype(float) - b[i]))),
                      first_row_maximum_float32_ulp=int(np.max(np.abs(ordered_a - ordered_b))),
                      first_row_different_components=np.flatnonzero(changed[i]).tolist())
    return result


def optimizer_group_comparison(original_model, current_model):
    import torch

    result = {}
    with zipfile.ZipFile(io.BytesIO(original_model)) as original, zipfile.ZipFile(io.BytesIO(current_model)) as current:
        for name in ('actor.optimizer.pth', 'critic.optimizer.pth', 'ent_coef_optimizer.pth'):
            left = torch.load(io.BytesIO(original.read(name)), map_location='cpu', weights_only=True)['param_groups']
            right = torch.load(io.BytesIO(current.read(name)), map_location='cpu', weights_only=True)['param_groups']
            normalized = []
            for groups in (left, right):
                value = json.loads(json.dumps(groups))
                for group in value:
                    if type(group['weight_decay']) not in (int, float):
                        raise ValueError('native real-valued Adam weight decay required')
                    group['weight_decay'] = float(group['weight_decay'])
                normalized.append(value)
            result[name] = dict(
                serialized_python_container_equality=left == right,
                effective_json_values_equal=canonical(normalized[0]) == canonical(normalized[1]),
                original_groups=left, current_groups=right,
                original_value_types=[{k:type(v).__name__ for k,v in g.items()} for g in left],
                current_value_types=[{k:type(v).__name__ for k,v in g.items()} for g in right],
            )
    return result


def diagnose(checkpoint, root):
    root, checkpoint = Path(root), Path(checkpoint)
    spec = protocol()
    history = spec['historical_control_reproduction']
    state = json.loads((root / history['source_state']).read_text())
    original_root = Path(state['completed'][history['source_task']]['root'])
    paths = list((original_root / 'runs').glob('*/metadata.json'))
    if len(paths) != 1 or checksum(paths[0]) != history['metadata_sha256']:
        raise ValueError('pinned original SAC metadata required')
    original = json.loads(paths[0].read_text())
    old_replay = paths[0].parent / 'replay.pkl'
    old_model = paths[0].parent / 'latest.zip'
    protected = {str(p): checksum(p) for p in (checkpoint, paths[0], old_replay, old_model)}
    with tarfile.open(checkpoint) as archive:
        names = archive.getnames()
        records = [n for n in names if n.endswith('/metadata.json')]
        if len(records) != 1:
            raise ValueError('one owned SAC checkpoint required')
        current = json.load(archive.extractfile(records[0]))
        prefix = records[0].rsplit('/', 1)[0] + '/'
        curve = [json.loads(line) for line in archive.extractfile(prefix + 'learning-curve.jsonl').read().splitlines()]
        raw = archive.extractfile(prefix + 'checkpoint-replay.pkl').read()
        current_model = archive.extractfile(prefix + 'latest-checkpoint.zip').read()
        replay = pickle.loads(raw)  # Only previously retrieved task-owned transport.
    if (current['crossq_arm'] != 'sac' or current['seed'] != 11 or current['smoke'] is not False
            or current['study_protocol_sha256'] != PROTOCOL_SHA256):
        raise ValueError('registered seed11 full SAC checkpoint required')
    optimizer_groups = optimizer_group_comparison(old_model.read_bytes(), current_model)
    initial = current['crossq_initialization']
    original_initial = original['critic_initialization']
    checks = dict(policy_tensors_exact=initial['policy_tensor_sha256'] == original_initial['policy_tensor_sha256'],
                  original_rng_streams_exact=initial['rng'] == original_initial['rng'])
    for field in ('dataset', 'simulation', 'reward_config', 'observation', 'start_sampling', 'action_execution', 'versions'):
        checks[field + '_exact'] = canonical(current[field]) == canonical(original[field])
    with old_replay.open('rb') as stream:
        old = pickle.load(stream)  # Original owned, retained control evidence.
    n = replay.pos
    if type(n) is not int or replay.full or old.full or not 0 < n <= old.pos:
        raise ValueError('bounded nonwrapped actual filled replay prefixes required')
    fields = {field: field_difference(getattr(old, field)[:n], getattr(replay, field)[:n])
              for field in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')}
    expected = history['all_checkpoint_scores'][0]
    first = curve[0]
    if (first['steps'], first['phase']) != (expected['steps'], expected['phase']):
        raise ValueError('fixed first historical checkpoint identity required')
    discrepancy = 100 * abs(first['score'] - expected['score'])
    if any(checksum(p) != value for p, value in protected.items()):
        raise ValueError('retained diagnostic sources changed')
    return dict(schema='crossq-reproduction-diagnosis-v1', protocol_sha256=PROTOCOL_SHA256,
                first_checkpoint=dict(steps=first['steps'], phase=first['phase'], actual_score=first['score'],
                    historical_score=expected['score'], absolute_discrepancy_pp=discrepancy,
                    maximum_allowed_pp=history['maximum_checkpoint_mean_return_discrepancy_pp'],
                    passed=discrepancy <= history['maximum_checkpoint_mean_return_discrepancy_pp']),
                initial_and_contract_checks=checks, replay_comparison=fields,
                saved_optimizer_group_comparison=optimizer_groups,
                current_checkpoint_replay_sha256=hashlib.sha256(raw).hexdigest(),
                protected_source_sha256=protected, source_files_unchanged=True,
                root_cause_established=False, candidate_performance_evaluated=False,
                interpretation_valid=False, automatic_retry_allowed=False,
                training_performed=False, optimizer_steps=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.checkpoint, Path.cwd())
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps(result['first_checkpoint']))
