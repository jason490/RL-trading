from pathlib import Path
import zipfile

import pytest

from rl_stocks.runtime.colab.canonical_market_review import CONTROLLER, equivalent_sources, review

FIXTURE = '''
import json
PROTOCOL_SHA256 = 'fixture'
def protocol(root): return 'unchanged'
def file_sha(path): return 'unchanged'
def run_worker(root, output): return 1
def bootstrap(): return 'original'
'''


def source_fixture(tmp_path):
    controller = tmp_path / CONTROLLER
    controller.parent.mkdir(parents=True)
    controller.write_text(FIXTURE)
    engine = tmp_path / 'src/engine.py'
    engine.write_text('def simulate(): return 1\n')
    archive = tmp_path / 'original.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.write(controller, CONTROLLER)
        stream.write(engine, 'src/engine.py')
    return archive, controller, engine


def test_setup_only_change_reuses_actual_unchanged_engine(tmp_path):
    archive, controller, _ = source_fixture(tmp_path)
    controller.write_text(FIXTURE.replace("'original'", "'setup correction'"))
    proof = equivalent_sources(tmp_path, archive)
    assert proof['worker_ast_contract_identical'] is True
    assert proof['byte_identical_modules'] == 1


@pytest.mark.parametrize('change', ['worker', 'import', 'protocol', 'engine', 'added', 'deleted'])
def test_preflight_cannot_be_reused_for_changed_numerical_code(tmp_path, change):
    archive, controller, engine = source_fixture(tmp_path)
    if change == 'worker':
        controller.write_text(FIXTURE.replace('return 1', 'return 2'))
    elif change == 'import':
        controller.write_text(FIXTURE.replace('import json', 'import random'))
    elif change == 'protocol':
        controller.write_text(FIXTURE.replace("PROTOCOL_SHA256 = 'fixture'", "PROTOCOL_SHA256 = 'changed'"))
    elif change == 'engine':
        engine.write_text('def simulate(): return 2\n')
    elif change == 'added':
        (tmp_path / 'src/unreviewed.py').write_text('pass\n')
    else:
        engine.unlink()
    with pytest.raises(ValueError):
        equivalent_sources(tmp_path, archive)


def test_review_preserves_original_allowance_and_outcome_contract():
    spec = review(Path(__file__).resolve().parents[1])
    assert spec['shared_baseline_units'] == 40.121717183767466
    assert spec['maximum_total_engineering_units'] == .1
    assert spec['original_failure_and_claim_preserved'] is True
    assert spec['automatic_retry_allowed'] is False


def test_shared_allowance_keeps_prior_attempt_charges(tmp_path, monkeypatch):
    import sys
    from rl_stocks.runtime.colab import canonical_market_review
    from rl_stocks.runtime.colab.budget import BudgetStop, Usage
    from rl_stocks.runtime.colab.canonical_market_review import ReviewedCanonicalMarketJob
    baseline = 40.
    monkeypatch.setattr(canonical_market_review, 'review', lambda root:
                        {'shared_baseline_units': baseline, 'maximum_total_engineering_units': .1})
    job = ReviewedCanonicalMarketJob(tmp_path, sys.executable, 'CPU')
    job.ledger.value['consumed'] = baseline + .095
    job.ledger.begin(job.session, Usage(548., .08, 0))
    # This new attempt alone fits .1 units, but prior charges plus its shutdown
    # reserve exceed the unchanged shared allowance.
    with pytest.raises(BudgetStop, match='shared'):
        job.guard_allowance(360)
    assert job.ledger.value['budget_stopped'] is True
    assert job.ledger.value['consumed'] == baseline + .095
