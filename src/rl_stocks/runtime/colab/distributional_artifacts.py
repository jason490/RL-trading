"""Fixed critic-study archive semantics; model deserialization is a separate check."""
from dataclasses import asdict
import hashlib
import json
import math

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.distributional_assets import initialization_entry, initialization_spec
from rl_stocks.training.distributional_critics import PROTOCOL_SHA256
from rl_stocks.training.distributional_records import validate_record
from rl_stocks.training.shared_initialization import read_initialization

DEVELOPMENT_SYMBOLS = ('SPY', 'QQQ', 'IWM', 'DIA', 'GOOGL', 'DIS', 'AMZN', 'HD', 'PG', 'KO', 'XOM', 'CVX', 'JPM', 'BAC', 'JNJ', 'UNH', 'CAT', 'HON', 'LIN', 'APD', 'AMT', 'PLD', 'AAPL', 'MSFT', 'NEE', 'DUK')
NORMALIZER_SHA256 = 'b3c8a298d66f253e4e4f5474214873d6b6aae16d1b4666633c6d83be6041469f'


def validate_critic_artifacts(read, prefix, folder, run, plan):
    context = plan['critic_study']
    arm, seed, smoke = context['arm'], context['seed'], context['smoke']
    job = plan['jobs'][0]
    frozen = json.loads(read(folder + 'manifest.json'))
    duration = run.get('duration_seconds')
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError('positive actual critic run duration required')
    if (run.get('critic_arm') != arm or run.get('study_protocol_sha256') != PROTOCOL_SHA256
            or run.get('start_sampling') != {'mode': 'legacy_uniform'} or run.get('sampling_diagnostics') is not None
            or run.get('action_execution') != execution_contract()
            or run.get('action_distribution') != {'version': 1, 'kind': 'squashed_gaussian', 'components': 4}):
        raise RuntimeError('fixed critic arm/input/execution/distribution contract mismatch')
    if canonical(run.get('simulation')) != canonical(asdict(SimulationConfig())):
        raise RuntimeError('unchanged critic simulation/cost contract required')
    proof = initialization_spec(arm, seed, smoke)
    if canonical(run.get('critic_shared_initialization')) != canonical(proof) or plan['initialization_contract'] != proof:
        raise RuntimeError('actual critic initialization provenance mismatch')
    entry, _ = initialization_entry(arm, seed, smoke)
    read_initialization(read(prefix + 'critic-initialization.npz'), entry)
    validate_record(run['critic_initialization'], arm, smoke, initial=True, seed=seed)
    validate_record(run['critic_final'], arm, smoke, seed=seed)
    validate_record(run['critic_selected'], arm, smoke, selected_steps=run['selected_step'], seed=seed)
    if run['critic_initialization']['policy_tensor_sha256'] == run['critic_final']['policy_tensor_sha256']:
        raise RuntimeError('critic model weights did not change during training')
    counts = run['training_execution']
    if (counts != json.loads(read(prefix + 'training-execution.json')) or counts['mode'] != 'unrestricted'
            or counts['transitions'] != job['steps'] or counts['sizing_changed'] != 0 or counts['protective_changed'] != 0
            or any(type(counts[k]) is not int or not 0 <= counts[k] <= job['steps']
                   for k in ('transitions', 'requested_nonzero', 'processed_nonunit', 'sizing_changed', 'protective_changed'))
            or counts['processed_nonunit'] > counts['requested_nonzero']):
        raise RuntimeError('actual unrestricted critic execution counts mismatch')
    for filename in ('selected-validation.json', 'validation.json'):
        reports = json.loads(read(prefix + filename))
        if (len(reports) != 26 or {r['symbol'] for r in reports.values()} != set(DEVELOPMENT_SYMBOLS)
                or any(r['split'] != 'validation' for r in reports.values())):
            raise RuntimeError('all26 matched critic development symbols required')
    geometry = run['critic_final']['critic_geometry']
    if run.get('critic_geometry') != geometry:
        raise RuntimeError('saved critic geometry mismatch')
    if (frozen['metadata'] != {k: v for k, v in run.items() if k != 'bundle_id'}
            or frozen['names'] != list(NAMES) or frozen['feature_schema'] != 'daily-indicators-v1'
            or digest(frozen['normalizer']) != NORMALIZER_SHA256
            or frozen['data_manifest']['included_splits'] != ['train', 'validation']
            or frozen['model_sha256'] != hashlib.sha256(read(prefix + 'best.zip')).hexdigest()
            or read(prefix + 'best.zip') != read(folder + 'model.zip')):
        raise RuntimeError('critic frozen bundle/model/normalization mismatch')
    read(prefix + 'latest.zip')
    read(prefix + 'replay.pkl')
