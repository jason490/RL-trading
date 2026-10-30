"""Execution admission and transport fixtures; no local learner optimization."""
import io
import json
from types import SimpleNamespace
import tarfile

import pytest

from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.runtime.colab.jobs import recurrent_screen_plan, validate_plan
from rl_stocks.training import recurrent_screen_runtime as runtime
from rl_stocks.training.recurrent_comparison import protocol
from rl_stocks.training.recurrent_screen_spec import specification, verify_compute_sources, ROOT


def test_fixed_roles_deadlines_and_seed():
    for arm, seconds in specification()['allocation_seconds'].items():
        plan = recurrent_screen_plan(arm)
        validate_plan(plan)
        assert plan['allocation_deadline_seconds'] == seconds
        assert plan['jobs'][0]['steps'] == 100000
        plan['jobs'][0]['steps'] = 110000
        with pytest.raises(ValueError):
            validate_plan(plan)
    for seed in (True, 23, 37):
        with pytest.raises(ValueError):
            recurrent_screen_plan('gru_sac', seed)
    verify_compute_sources(ROOT)


def test_original_budget_and_single_claim_per_role(tmp_path, monkeypatch):
    ledger = SimpleNamespace(path=tmp_path/'usage-ledger.json',
                             value=dict(consumed=40., budget_stopped=False))
    monkeypatch.setattr(runtime, 'verify_original_records', lambda _: dict(start_units=39.5))
    assert runtime.shared_screen_budget(ledger) == (2., 39.5)
    with pytest.raises(ValueError):
        runtime.shared_screen_budget(ledger, requested_start=40.)
    with pytest.raises(ValueError):
        runtime.shared_screen_budget(ledger, requested_limit=3.)
    for arm in protocol()['arms']:
        runtime.claim_screen_cell(ledger, 'rl-stocks-cpu-fixture', arm)
        record = json.loads((tmp_path/'recurrent-comparison-screen-claims'/f'{arm}-11.json').read_text())
        assert record['transitions'] == 100000 and not record['automatic_retry_allowed']
        with pytest.raises(RuntimeError, match='already claimed'):
            runtime.claim_screen_cell(ledger, 'rl-stocks-cpu-new', arm)
    with pytest.raises(RuntimeError, match='no automatic retry'):
        runtime.shared_screen_budget(ledger)
    ledger.value['budget_stopped'] = True
    with pytest.raises(BudgetStop):
        runtime.shared_screen_budget(ledger)


def test_incomplete_and_hostile_transport_rejected_before_model_load(tmp_path, monkeypatch):
    from rl_stocks.runtime.colab import recurrent_screen_artifacts as transport
    def forbidden(*args, **kwargs):
        pytest.fail('invalid transport reached model loading')
    monkeypatch.setattr(transport, 'verify_cell', forbidden)
    for name in ('artifacts/completion.json', '../escape', 'artifacts/extra'):
        path = tmp_path/'bad.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            member = tarfile.TarInfo(name)
            member.size = 2
            archive.addfile(member, io.BytesIO(b'{}'))
        with pytest.raises(RuntimeError, match='allowlisted'):
            transport.validate_screen_archive(path, recurrent_screen_plan('gru_sac'), protocol()['dataset'])


def test_download_defers_heavy_verification_and_serializes_local_workers(tmp_path):
    from rl_stocks.runtime.colab.fleet import Fleet
    actor = SimpleNamespace(session='rl-stocks-cpu-fixture', phase='download',
                            task=dict(plan=recurrent_screen_plan('gru_sac')))
    saves = []
    fleet = Fleet.__new__(Fleet)
    fleet.root = tmp_path
    fleet.budget = SimpleNamespace(active=dict(children={actor.session: {}}),
                                   ledger=SimpleNamespace(save=lambda: saves.append('ledger')))
    fleet.save = lambda: saves.append('state')
    fleet.completed_command(actor, 0, tmp_path/'download-output.txt')
    assert actor.phase == 'await_validation'
    fleet.fatal = False
    fleet.snapshot = lambda _: None
    other = SimpleNamespace(phase='validate')
    fleet.actors = [actor, other]
    fleet.start_local_validation = lambda _: saves.append('validate')
    fleet.step(actor)
    assert 'validate' not in saves
    other.phase = 'done'
    fleet.step(actor)
    assert saves[-1] == 'validate'
