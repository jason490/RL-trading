from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.runtime.colab.jobs import critic_plan, validate_plan
from rl_stocks.runtime.colab.payloads import prepare_payload
from rl_stocks.training.distributional_critics import PROTOCOL_SHA256, protocol
from rl_stocks.training.distributional_runtime import (
    make_runtime_profile, projection, shared_study_budget, validate_runtime_profile,
)


def accepted_profile(tmp_path):
    records = {arm: dict(duration_seconds=30., actual_saved_models_restored=True) for arm in protocol()['arms']}
    path = tmp_path / '.runtime/distributional-smoke-acceptance.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(protocol_sha256=PROTOCOL_SHA256, passed=True,
        cleanup_verified=True, initialization_checks={'sampling_rng_identical': True}, records=records)))
    return make_runtime_profile(tmp_path)


@pytest.mark.parametrize('arm', list(protocol()['arms']))
def test_actual_bound_smoke_and_gated_full_plans(tmp_path, arm):
    smoke = critic_plan(arm, smoke=True)
    validate_plan(smoke)
    assert smoke['version'] == 33 and smoke['jobs'][0]['steps'] == 1024
    assert smoke['jobs'][0]['algorithm'] == protocol()['arms'][arm]['algorithm']
    with pytest.raises(ValueError):
        critic_plan(arm)
    full = critic_plan(arm, profile=accepted_profile(tmp_path))
    validate_plan(full)
    assert full['jobs'][0]['steps'] == 50000


@pytest.mark.parametrize('field', ['algorithm', 'steps', 'overrides', 'initialization', 'boolean_alias'])
def test_plan_semantic_tampering_rejected(field):
    plan = deepcopy(critic_plan('tqc_truncated', smoke=True))
    if field == 'algorithm':
        plan['jobs'][0][field] = 'sac'
    if field == 'steps':
        plan['jobs'][0][field] = 2048
    if field == 'overrides':
        plan['jobs'][0][field]['top_quantiles_to_drop_per_net'] = 1
    if field == 'boolean_alias':
        plan['jobs'][0]['overrides']['target_update_interval'] = True
    if field == 'initialization':
        plan['initialization_contract']['rng_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_runtime_projection_and_acceptance_mutations_do_not_authorize_screen(tmp_path):
    profile = accepted_profile(tmp_path)
    validate_runtime_profile(profile, tmp_path)
    bad = deepcopy(profile)
    bad['projection']['projected_seconds'] = 1
    with pytest.raises(ValueError):
        validate_runtime_profile(bad)
    assert not projection(dict(sac_native=1, tqc_untruncated=30, tqc_truncated=40))['screen_allocation_allowed']
    path = tmp_path / '.runtime/distributional-smoke-acceptance.json'
    path.write_text('{}')
    with pytest.raises(ValueError):
        validate_runtime_profile(profile, tmp_path)


def test_budget_baseline_survives_controllers_and_cannot_reset(tmp_path):
    ledger = SimpleNamespace(path=tmp_path / 'usage-ledger.json', value={'consumed': 39.0})
    assert shared_study_budget(ledger) == (2.0, 39.0)
    ledger.value['consumed'] = 39.5
    assert shared_study_budget(ledger) == (2.0, 39.0)
    with pytest.raises(ValueError):
        shared_study_budget(ledger, requested_start=39.5)
    with pytest.raises(ValueError):
        shared_study_budget(ledger, requested_limit=3)
    ledger.value['consumed'] = 41.0
    with pytest.raises(BudgetStop):
        shared_study_budget(ledger)


def test_source_payload_carries_all_bound_initializer_files(tmp_path):
    import tarfile
    root = Path(__file__).parents[1]
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'dataset.json').write_text('{}')
    # Use repository source plus isolated job staging; no account or data requests.
    plan = critic_plan('tqc_truncated', smoke=True)
    packed = prepare_payload(root, dataset, plan, 'test-critic-payload-' + tmp_path.name)
    try:
        with tarfile.open(packed) as archive:
            names = {m.name for m in archive.getmembers() if m.isfile()}
            assert 'configs/initializations/distributional_v1/manifest.json' in names
            assert sum(n.startswith('configs/initializations/distributional_v1/') and n.endswith('.npz') for n in names) == 12
    finally:
        import shutil
        shutil.rmtree(packed.parent)
