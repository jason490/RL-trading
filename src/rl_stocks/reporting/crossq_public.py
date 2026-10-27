"""Mandatory public CrossQ provenance and observed evidence; no replay contents."""
import json
from pathlib import Path

from rl_stocks.core.contracts import canonical
from rl_stocks.training.crossq_assets import initialization_entry, initialization_spec
from rl_stocks.training.crossq_forward_audit import validate_audit
from rl_stocks.training.crossq_state import validate_record
from rl_stocks.training.shared_initialization import read_initialization


def public_evidence(run, metadata):
    run = Path(run)
    arm, seed, smoke = metadata['crossq_arm'], metadata['seed'], metadata['smoke']
    if arm not in ('sac', 'crossq') or metadata['algorithm'] != arm or type(smoke) is not bool:
        raise ValueError('fixed CrossQ public arm/algorithm/phase required')
    proof = initialization_spec(arm, seed, smoke)
    if canonical(metadata['crossq_shared_initialization']) != canonical(proof):
        raise ValueError('public CrossQ initializer provenance mismatch')
    initializer = run / 'crossq-initialization.npz'
    entry, _ = initialization_entry(arm, seed, smoke)
    read_initialization(initializer.read_bytes(), entry)
    validate_record(metadata['crossq_initialization'], arm, smoke, initial=True)
    validate_record(metadata['crossq_final'], arm, smoke)
    validate_record(metadata['crossq_selected'], arm, smoke, phase=metadata['crossq_selected_phase'])
    if (metadata['crossq_final']['seed'] != seed or metadata['crossq_selected']['seed'] != seed
            or metadata['crossq_initialization']['seed'] != seed
            or metadata['crossq_final']['num_timesteps'] != metadata['actual_steps']
            or metadata['crossq_selected']['num_timesteps'] != metadata['selected_step']):
        raise ValueError('public CrossQ seed/checkpoint identity mismatch')
    replay = run / 'crossq-replay-state.json'
    if canonical(json.loads(replay.read_text())) != canonical(metadata['crossq_replay']):
        raise ValueError('public final replay hash evidence mismatch')
    if metadata['crossq_replay']['filled_transitions'] != metadata['actual_steps']:
        raise ValueError('public final replay transition mismatch')
    paths = [initializer, replay]
    if arm == 'crossq':
        audit = run / 'crossq-normalization-audit.json'
        value = json.loads(audit.read_text())
        if canonical(value) != canonical(metadata['crossq_forward_audit']):
            raise ValueError('public observed normalization sidecar mismatch')
        validate_audit(value, metadata['actual_steps'], 'post_update', smoke,
                       metadata['crossq_final']['normalization'])
        if (metadata['crossq_geometry'] != metadata['crossq_selected']['geometry']
                or metadata['crossq_normalization'] != metadata['crossq_selected']['normalization']):
            raise ValueError('public selected normalization binding mismatch')
        paths.append(audit)
    elif any(k in metadata for k in ('crossq_geometry', 'crossq_normalization', 'crossq_forward_audit')):
        raise ValueError('public SAC control must not claim CrossQ normalization')
    return paths
