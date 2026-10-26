"""Restore fixed untrained tensors and original SAC sampling streams before fitting."""

from rl_stocks.core.contracts import canonical
from rl_stocks.training.crossq_assets import (
    initialization_entry, initialization_spec, portable_path, validate_assets,
)
from rl_stocks.training.crossq_mapping import _native_normalization
from rl_stocks.training.crossq_state import model_record, _validate, validate_record, _linear_arrays
from rl_stocks.training.distributional_critics import parsed_global_rng, restore_global_rng
from rl_stocks.training.shared_initialization import _untrained, load_initialization, read_initialization, tensor_fingerprint


def apply_initialization(model, arm, seed=11, smoke=False):
    entry, globals_record = initialization_entry(arm, seed, smoke)
    proof = initialization_spec(arm, seed, smoke)
    if type(model.seed) is not int or model.seed != seed:
        raise ValueError('actual model seed must match shared start')
    _untrained(model)
    fresh = model_record(model)
    _validate(fresh, arm, smoke, True, 'post_update', False)
    if arm == 'crossq':
        _native_normalization(model)
    parsed_global_rng(globals_record)
    validate_assets()
    actual = load_initialization(model, portable_path(entry['file']), entry)
    if any(actual[k] != proof[k] for k in actual):
        raise ValueError('loaded initializer identity mismatch')
    restore_global_rng(globals_record)
    model._research_initialization = proof
    validate_record(model_record(model), arm, smoke, initial=True)
    if canonical(model._research_initialization) != canonical(proof):
        raise ValueError('shared initialization changed')
    return proof


def validate_initial_comparison(records, smoke=False):
    """Bound both observed starts to pinned archives before comparing shared fields."""
    if set(records) != {'sac', 'crossq'}:
        raise ValueError('both fixed SAC/CrossQ initial records required')
    validate_assets()  # Exact ten-module mapping; native parameter names differ across algorithms.
    for arm, record in records.items():
        validate_record(record, arm, smoke, initial=True)
        entry, _ = initialization_entry(arm, record['seed'], smoke)
        arrays, _ = read_initialization(portable_path(entry['file']).read_bytes(), entry)
        if record['linear_tensor_sha256'] != tensor_fingerprint(_linear_arrays(arrays)):
            raise ValueError('actual per-arm Linear identity must match pinned mapped archive')
    if (records['sac']['seed'] != records['crossq']['seed']
            or canonical(records['sac']['rng']) != canonical(records['crossq']['rng'])):
        raise ValueError('actual shared CrossQ seed/Linear tensors/original sampling RNG must match')
    return dict(shared_linear_tensors_verified=True, sampling_rng_identical=True)
