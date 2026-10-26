"""Fixed CrossQ-study archive semantics; model deserialization is a separate check."""
from dataclasses import asdict
import hashlib
import json
import math

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.crossq_assets import BINDING_SHA256, initialization_entry, initialization_spec
from rl_stocks.training.crossq_spec import PROTOCOL_SHA256
from rl_stocks.training.crossq_state import validate_record
from rl_stocks.training.shared_initialization import read_initialization

DEVELOPMENT_SYMBOLS = ('SPY', 'QQQ', 'IWM', 'DIA', 'GOOGL', 'DIS', 'AMZN', 'HD', 'PG', 'KO', 'XOM', 'CVX', 'JPM', 'BAC', 'JNJ', 'UNH', 'CAT', 'HON', 'LIN', 'APD', 'AMT', 'PLD', 'AAPL', 'MSFT', 'NEE', 'DUK')
NORMALIZER_SHA256 = 'b3c8a298d66f253e4e4f5474214873d6b6aae16d1b4666633c6d83be6041469f'


def validate_crossq_artifacts(read, prefix, folder, run, plan):
    context = plan['crossq_study']
    arm, seed, smoke = context['arm'], context['seed'], context['smoke']
    job = plan['jobs'][0]
    frozen = json.loads(read(folder + 'manifest.json'))
    duration = run.get('duration_seconds')
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError('positive actual CrossQ run duration required')
    if (run.get('crossq_arm') != arm or run.get('crossq_binding_sha256') != BINDING_SHA256 or run.get('study_protocol_sha256') != PROTOCOL_SHA256
            or run.get('start_sampling') != {'mode': 'legacy_uniform'} or run.get('sampling_diagnostics') is not None
            or run.get('action_execution') != execution_contract()
            or run.get('action_distribution') != {'version': 1, 'kind': 'squashed_gaussian', 'components': 4}):
        raise RuntimeError('fixed CrossQ arm/input/execution/distribution contract mismatch')
    if canonical(run.get('simulation')) != canonical(asdict(SimulationConfig())):
        raise RuntimeError('unchanged CrossQ simulation/cost contract required')
    proof = initialization_spec(arm, seed, smoke)
    if canonical(run.get('crossq_shared_initialization')) != canonical(proof) or plan['initialization_contract'] != proof:
        raise RuntimeError('actual CrossQ initialization provenance mismatch')
    entry, _ = initialization_entry(arm, seed, smoke)
    read_initialization(read(prefix + 'crossq-initialization.npz'), entry)
    validate_record(run['crossq_initialization'], arm, smoke, initial=True)
    validate_record(run['crossq_final'], arm, smoke)
    validate_record(run['crossq_selected'], arm, smoke, phase=run['crossq_selected_phase'])
    if run['crossq_initialization']['policy_tensor_sha256'] == run['crossq_final']['policy_tensor_sha256']:
        raise RuntimeError('CrossQ model weights did not change during training')
    counts = run['training_execution']
    if (counts != json.loads(read(prefix + 'training-execution.json')) or counts['mode'] != 'unrestricted'
            or counts['transitions'] != job['steps'] or counts['sizing_changed'] != 0 or counts['protective_changed'] != 0
            or any(type(counts[k]) is not int or not 0 <= counts[k] <= job['steps']
                   for k in ('transitions', 'requested_nonzero', 'processed_nonunit', 'sizing_changed', 'protective_changed'))
            or counts['processed_nonunit'] > counts['requested_nonzero']):
        raise RuntimeError('actual unrestricted CrossQ execution counts mismatch')
    for filename in ('selected-validation.json', 'validation.json'):
        reports = json.loads(read(prefix + filename))
        if (len(reports) != 26 or {r['symbol'] for r in reports.values()} != set(DEVELOPMENT_SYMBOLS)
                or any(r['split'] != 'validation' for r in reports.values())):
            raise RuntimeError('all26 matched CrossQ development symbols required')
    if (run['crossq_final']['num_timesteps'] != job['steps']
            or run['crossq_selected']['num_timesteps'] != run['selected_step']):
        raise RuntimeError('CrossQ selected/final transition identity mismatch')
    checkpoint = json.loads(read(prefix + 'checkpoint-state.json'))
    if checkpoint['selected']['phase'] != run['crossq_selected_phase']:
        raise RuntimeError('CrossQ selected checkpoint phase mismatch')
    if run['crossq_replay'] != json.loads(read(prefix + 'crossq-replay-state.json')):
        raise RuntimeError('CrossQ final replay sidecar mismatch')
    if run['crossq_replay']['filled_transitions'] != job['steps']:
        raise RuntimeError('CrossQ filled replay transition mismatch')
    if arm == 'crossq':
        from rl_stocks.training.crossq_forward_audit import validate_audit
        if (run.get('crossq_geometry') != run['crossq_selected']['geometry']
                or run.get('crossq_normalization') != run['crossq_selected']['normalization']):
            raise RuntimeError('selected CrossQ geometry/normalization mismatch')
        validate_audit(run['crossq_forward_audit'], job['steps'], 'post_update', smoke,
                       run['crossq_final']['normalization'])
        if run['crossq_forward_audit'] != json.loads(read(prefix + 'crossq-normalization-audit.json')):
            raise RuntimeError('CrossQ observed forward sidecar mismatch')
        selected_points = []
        for point in run['validation_checkpoints']:
            validate_audit(point['crossq_forward_audit'], point['steps'], point['phase'], smoke)
            if (point['steps'], point['phase']) == (run['selected_step'], run['crossq_selected_phase']):
                selected_points.append(point)
        if len(selected_points) != 1:
            raise RuntimeError('unique selected CrossQ observed checkpoint required')
        validate_audit(selected_points[0]['crossq_forward_audit'], run['selected_step'],
                       run['crossq_selected_phase'], smoke, run['crossq_selected']['normalization'])
    elif any(key in run for key in ('crossq_geometry', 'crossq_normalization', 'crossq_forward_audit')):
        raise RuntimeError('SAC control must not claim CrossQ normalization')
    if 'critic_geometry' in run:
        raise RuntimeError('unrelated TQC geometry in CrossQ study')
    if (frozen['metadata'] != {k: v for k, v in run.items() if k != 'bundle_id'}
            or frozen['names'] != list(NAMES) or frozen['feature_schema'] != 'daily-indicators-v1'
            or digest(frozen['normalizer']) != NORMALIZER_SHA256
            or frozen['data_manifest']['included_splits'] != ['train', 'validation']
            or frozen['model_sha256'] != hashlib.sha256(read(prefix + 'best.zip')).hexdigest()
            or read(prefix + 'best.zip') != read(folder + 'model.zip')):
        raise RuntimeError('CrossQ frozen bundle/model/normalization mismatch')
    read(prefix + 'latest.zip')
    read(prefix + 'replay.pkl')
