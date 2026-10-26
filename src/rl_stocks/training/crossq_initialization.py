"""Restore fixed untrained tensors and original SAC sampling streams before fitting."""

from rl_stocks.core.contracts import canonical
from rl_stocks.training.crossq_assets import (
    initialization_entry, initialization_spec, portable_path, validate_assets,
)
from rl_stocks.training.crossq_mapping import _native_normalization
from rl_stocks.training.crossq_state import model_record, _validate, validate_record
from rl_stocks.training.distributional_critics import parsed_global_rng, restore_global_rng
from rl_stocks.training.shared_initialization import _untrained, load_initialization


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
