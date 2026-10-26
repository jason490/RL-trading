from copy import deepcopy

import numpy as np
import pytest

from rl_stocks.rl.registry.factory import create
from rl_stocks.training.crossq_inspection import SpacesOnly
from rl_stocks.training.crossq_forward_audit import NormalizationAudit, validate_audit, SOURCE_SHA256
from rl_stocks.training.crossq_spec import expected_updates


def model():
    return create('crossq', SpacesOnly(dict(observation_size=21, action_size=4)), seed=11, smoke=True)


def test_real_eval_forwards_observed_without_changing_running_state():
    candidate = model()
    audit = NormalizationAudit(candidate)
    try:
        candidate.predict(np.zeros(21, dtype=np.float32), deterministic=True)
        snapshot = audit.snapshot()
        assert sum(r['evaluation_calls'] for r in snapshot['layers'].values()) == 3
        assert not any(r['training_calls'] for r in snapshot['layers'].values())
    finally:
        audit.close()
    assert not audit.handles


def test_manual_counter_fixture_exercises_strict_warmup_classification_without_fitting():
    candidate = model()
    audit = NormalizationAudit(candidate)
    name, layer = next(iter(audit.modules.items()))
    # Invoke observer callbacks and increment a synthetic counter, never a train-mode forward.
    for _ in range(1002):
        audit._before(name, layer)
        layer.steps.add_(1)
        audit._after(name, layer)
    assert audit.snapshot()['layers'][name]['renorm_condition_calls'] == 1
    layer.steps.add_(1)
    with pytest.raises(ValueError, match='unobserved'):
        audit.snapshot()
    audit.close()


def test_observed_record_cannot_claim_renorm_from_nominal_steps_alone():
    counts = expected_updates('crossq', 4096, 'post_update', True)
    rows = {f'{branch}.layer{i}': dict(training_calls=counts[branch+'_normalization'],
            counter=counts[branch+'_normalization'], evaluation_calls=0,
            renorm_condition_calls=counts[branch+'_renorm_forward_calls'])
            for branch,total in [('actor',3),('critic',6)] for i in range(total)}
    value = dict(version=1, observed=True, source_sha256=SOURCE_SHA256, warmup_steps=1000, layers=rows)
    validate_audit(value, 4096, 'post_update', True)
    wrong = deepcopy(value)
    next(iter(wrong['layers'].values()))['renorm_condition_calls'] = 0
    with pytest.raises(ValueError):
        validate_audit(wrong, 4096, 'post_update', True)


def test_native_checkpoint_save_does_not_serialize_observer_hooks(tmp_path):
    from sb3_contrib import CrossQ
    from rl_stocks.rl.crossq.geometry import layers
    candidate = model()
    audit = NormalizationAudit(candidate)
    try:
        candidate.save(tmp_path / 'untrained.zip')
        loaded = CrossQ.load(tmp_path / 'untrained.zip', device='cpu')
        assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in layers(loaded).values())
        assert not any(row['training_calls'] for row in audit.snapshot()['layers'].values())
    finally:
        audit.close()


def test_ignored_evaluation_mutation_cannot_be_relabelled_as_passing_audit():
    candidate = model()
    audit = NormalizationAudit(candidate)
    name, layer = next(iter(audit.modules.items()))
    layer.eval()
    audit._before(name, layer)
    layer.ra_mean.add_(1)  # Explicit manual corruption fixture, no training forward.
    with pytest.raises(ValueError, match='evaluation'):
        audit._after(name, layer)
    layer.ra_mean.zero_()
    with pytest.raises(ValueError):
        audit.snapshot()
    audit.close()
