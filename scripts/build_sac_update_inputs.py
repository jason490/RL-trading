"""Extract matched training-only prefixes from retained owned replay; never learn."""
import argparse
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.crossq_assets import initialization_spec
from rl_stocks.training.crossq_spec import parameters
from rl_stocks.training.sac_update_input import PROTOCOL_SHA256, FIELDS, protocol, validate_payload


def build(root, output):
    root, output = Path(root), Path(output)
    spec = protocol()
    state = json.loads((root / spec['historical_state']).read_text())
    retained = Path(state['completed'][spec['historical_task']]['root'])
    paths = list((retained / 'runs').glob('*/metadata.json'))
    if len(paths) != 1 or checksum(paths[0]) != spec['historical_metadata_sha256']:
        raise ValueError('pinned original complete SAC metadata required')
    metadata = json.loads(paths[0].read_text())
    old_replay = paths[0].parent / 'replay.pkl'
    checkpoint = root / spec['screen_diagnostic_checkpoint']
    if checksum(checkpoint) != spec['screen_diagnostic_checkpoint_sha256']:
        raise ValueError('pinned task-owned diagnostic checkpoint required')
    protected = {str(p):checksum(p) for p in (paths[0], old_replay, checkpoint)}
    with old_replay.open('rb') as stream:
        original = pickle.load(stream)  # Previously retrieved and verified task-owned evidence only.
    with tarfile.open(checkpoint) as archive:
        name = next(n for n in archive.getnames() if n.endswith('/checkpoint-replay.pkl'))
        current = pickle.loads(archive.extractfile(name).read())
    n = spec['replay_rows']
    if original.full or current.full or min(original.pos, current.pos) <= n:
        raise ValueError('fixed nonwrapped filled training prefixes required')
    fields = {}
    for name, shape in FIELDS.items():
        left, right = getattr(original, name)[:n], getattr(current, name)[:n]
        if (left.dtype != np.float32 or right.dtype != np.float32 or left.shape != shape
                or right.shape != shape or not np.array_equal(left.view(np.uint32), right.view(np.uint32))):
            raise ValueError('all retained pre-update prefix fields must agree bitwise')
        fields[name] = dict(dtype='float32', values=left.tolist(), sha256=fingerprint(left))
    followup = original.observations[n, 0]
    if not np.array_equal(followup.view(np.uint32), current.observations[n, 0].view(np.uint32)):
        raise ValueError('follow-up observation must agree before first action divergence')
    payload = dict(schema='sac-first-update-input-v1', protocol_sha256=PROTOCOL_SHA256, rows=n,
        historical_metadata_sha256=spec['historical_metadata_sha256'],
        diagnostic_checkpoint_sha256=spec['screen_diagnostic_checkpoint_sha256'],
        common_prefix_bitwise_verified=True, arrays=fields,
        postupdate_observation=followup.tolist(), postupdate_observation_sha256=fingerprint(followup),
        arm_parameters=dict(original_defaults=metadata['resolved_parameters'],
                            explicit_serialized_defaults=json.loads(json.dumps(parameters('sac', False)))),
        shared_initialization=initialization_spec('sac', 11, False),
        source_kind='retained_training_replay_only_no_new_environment_transitions')
    validate_payload(payload)
    identity = digest(payload)
    write_once(output / identity / 'dataset.json', canonical(payload))
    if any(checksum(p) != value for p, value in protected.items()):
        raise ValueError('retained diagnostic source changed')
    return dict(protocol_sha256=PROTOCOL_SHA256, input_identity=identity, rows=n,
                filled_field_sha256={k:v['sha256'] for k,v in fields.items()},
                protected_source_sha256=protected, source_files_unchanged=True,
                shared_initialization=payload['shared_initialization'],
                training_performed=False, optimizer_steps=0, allocation_performed=False,
                performance_evaluation=False, original_integrity_stop_cleared=False,
                implementation_sha256={str(p.relative_to(root)):checksum(p) for p in
                    (Path(__file__).resolve(), root/'src/rl_stocks/training/sac_update_input.py')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    args = parser.parse_args()
    proof = build(Path.cwd(), args.output)
    write_once(args.proof, (json.dumps(proof, indent=2) + '\n').encode())
    print(json.dumps(proof))
