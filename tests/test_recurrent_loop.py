"""Synthetic schedule/curve fixtures; update spies never change model tensors."""
from dataclasses import replace
from copy import deepcopy
import json
from types import SimpleNamespace
import tarfile

import pytest

from test_recurrent_collector import setup
from rl_stocks.core.contracts import digest
from rl_stocks.training.checkpoint_export import export_artifacts
from rl_stocks.training.recurrent_loop import (
    LoopConfig, _CheckpointHistory, _TrainingEngine, run_colab_training, validate_dataset,
)


def config():
    return LoopConfig(transitions=12, warmup_steps=4, train_every=2, gradient_steps=1,
                       burn_in=1, unroll=1, windows_per_batch=2, replay_capacity=30,
                       episode_steps=4, evaluate_every=5, seed=11,
                       hidden_size=8, start_sampling='legacy_uniform')


def test_schedule_real_collection_and_spy_updates_final_evaluation_not_on_grid():
    cfg = config()
    env, _, _, collector = setup(warmup=4, episode_steps=4)
    learner = SimpleNamespace(updates=0, failed=False, max_updates=4)
    seen = []
    def update(batch):
        seen.append((collector.transitions, batch.insertion_ids.copy()))
        learner.updates += 1
        return dict(update=learner.updates)
    learner.update = update
    engine = _TrainingEngine(collector, learner, cfg)
    evaluations = []
    try:
        for _ in range(12):
            engine.advance()
            if engine.evaluation_due():
                evaluations.append(collector.transitions)
        assert [step for step, _ in seen] == [6, 8, 10, 12]
        assert evaluations == [5, 10, 12] and learner.updates == cfg.maximum_updates == 4
        assert all(rows.shape == (2, 2) for _, rows in seen)
        with pytest.raises(RuntimeError, match='exhausted'):
            engine.advance()
        assert collector.transitions == 12
    finally:
        env.close()


def test_update_failure_poisoned_without_collecting_again():
    env, _, _, collector = setup(warmup=4, episode_steps=4)
    learner = SimpleNamespace(updates=0, failed=False, max_updates=4)
    def fail(batch):
        raise RuntimeError('spy failure before any optimizer')
    learner.update = fail
    engine = _TrainingEngine(collector, learner, config())
    try:
        for _ in range(5):
            engine.advance()
        with pytest.raises(RuntimeError, match='spy failure'):
            engine.advance()
        assert collector.transitions == 6 and engine.failed
        with pytest.raises(RuntimeError, match='no retry'):
            engine.advance()
        assert collector.transitions == 6
    finally:
        env.close()


def test_multiple_updates_per_collection_and_revision_counts():
    cfg = replace(config(), gradient_steps=2)
    env, _, _, collector = setup(warmup=4, episode_steps=4)
    learner = SimpleNamespace(updates=0, failed=False, max_updates=8)
    def update(batch):
        learner.updates += 1
        return {}
    learner.update = update
    engine = _TrainingEngine(collector, learner, cfg)
    try:
        for _ in range(12):
            engine.advance()
        assert learner.updates == cfg.maximum_updates == 8
    finally:
        env.close()


def result(score):
    return {'SPY': dict(metrics=dict(net_return=score, max_drawdown=-.1, sharpe=.2, costs=12., turnover=3.),
                        baselines=dict(buy_hold=dict(metrics=dict(net_return=.3))))}


def test_compact_percentage_changes_tie_selection_and_separate_final(tmp_path):
    recorder = _CheckpointHistory(tmp_path, clock=lambda: 10.)
    points = [(5, .1), (10, .1), (12, .08)]
    for step, score in points:
        checkpoint = f'checkpoint-{step}'
        (tmp_path/checkpoint).mkdir()
        (tmp_path/checkpoint/'manifest.json').write_text('{}')  # completed-checkpoint boundary fixture
        recorder.record(result(score), steps=step, updates=step//2, training={}, checkpoint=checkpoint)
    rows = [json.loads(line) for line in (tmp_path/'learning-curve.jsonl').read_text().splitlines()]
    assert len(rows) == 3 and rows[0]['change_pp'] is None
    assert rows[1]['change_pp'] == 0 and rows[2]['change_pp'] == pytest.approx(-2)
    assert rows[2]['equity_change_pct_from_previous'] == pytest.approx(100*(1.08/1.1-1))
    assert rows[2]['buy_hold_return_pct'] == 30
    assert recorder.best_step == 5 and recorder.history[-1]['steps'] == 12
    assert json.loads((tmp_path/'selected.json').read_text())['checkpoint'] == 'checkpoint-5'
    with pytest.raises(ValueError, match='increasing'):
        recorder.record(result(.2), steps=12, updates=7, training={}, checkpoint='checkpoint-12')


def test_missing_checkpoint_never_becomes_a_curve_or_selection(tmp_path):
    history = _CheckpointHistory(tmp_path)
    with pytest.raises(ValueError, match='durable'):
        history.record(result(.1), steps=1, updates=0, training={}, checkpoint='missing')
    assert not history.history and not (tmp_path/'selected.json').exists()


@pytest.mark.parametrize('kwargs', [dict(transitions=4), dict(warmup_steps=1), dict(train_every=0),
                                   dict(burn_in=True), dict(replay_capacity=2), dict(unroll=5),
                                   dict(evaluate_every=20), dict(start_sampling='unknown')])
def test_invalid_schedule_rejects(kwargs):
    with pytest.raises(ValueError):
        replace(config(), **kwargs)


def test_outer_loop_rejects_locally_before_output_or_model(tmp_path):
    with pytest.raises(RuntimeError, match='Colab CPU'):
        run_colab_training({}, tmp_path/'artifacts/run', config(), exposure_mode='unit_magnitude', lineage={})
    assert not (tmp_path/'artifacts').exists()


def test_dataset_identity_chronology_and_retired_exclusion():
    from test_recurrent_sac_evaluation import fixture
    _, data = fixture()
    train = deepcopy(data['episodes']['SPY'])
    train['split'] = 'train'
    for i, bar in enumerate(train['bars']):
        bar['session'] = f'2020-01-{i+1:02d}'
    data['episodes']['training-SPY'] = train
    normalizer, training = validate_dataset(data, digest(data))
    assert len(training) == 1 and normalizer.names == tuple(data['names'])
    with pytest.raises(ValueError, match='identity'):
        validate_dataset(data, '0'*64)
    bad = deepcopy(data)
    bad['episodes']['SPY']['split'] = 'test'
    with pytest.raises(ValueError, match='retired'):
        validate_dataset(bad, digest(bad))
    bad = deepcopy(data)
    bad['episodes']['training-SPY']['bars'][0]['session'] = '2021-01-01'
    with pytest.raises(ValueError, match='chronological'):
        validate_dataset(bad, digest(bad))


def test_checkpoint_export_owns_only_artifact_tree_and_cleans_temporary(tmp_path, monkeypatch):
    root = tmp_path/'artifacts'
    root.mkdir()
    (root/'model.txt').write_text('fixture')
    (tmp_path/'outside.txt').write_text('excluded')
    target = tmp_path/'checkpoint.tar.gz'
    monkeypatch.setenv('RL_CHECKPOINT_ARCHIVE', str(target))
    export_artifacts(root)
    with tarfile.open(target) as archive:
        assert {m.name for m in archive.getmembers()} == {'artifacts', 'artifacts/model.txt'}
    assert not target.with_name(target.name+'.tmp').exists()
    monkeypatch.setenv('RL_CHECKPOINT_ARCHIVE', str(root/'recursive.tar.gz'))
    with pytest.raises(ValueError):
        export_artifacts(root)


def test_final_is_distinct_and_selected_final_visual_reports_render(tmp_path):
    from test_recurrent_sac_evaluation import fixture
    from rl_stocks.evaluation.recurrent_sac import evaluate_recurrent_sac
    networks, data = fixture()
    results = evaluate_recurrent_sac(networks, data, exposure_mode='unit_magnitude')
    history = _CheckpointHistory(tmp_path)
    for step in (4, 8):
        checkpoint = f'checkpoint-{step}'
        (tmp_path/checkpoint).mkdir()
        (tmp_path/checkpoint/'manifest.json').write_text('{}')
        history.record(results, steps=step, updates=step//2, training={}, checkpoint=checkpoint)
    history.finish(results, steps=8, updates=4)
    assert json.loads((tmp_path/'selected.json').read_text())['steps'] == 4
    assert json.loads((tmp_path/'final.json').read_text())['steps'] == 8
    assert 'Selected development checkpoint' in (tmp_path/'selected-validation.html').read_text()
    assert 'Final development checkpoint' in (tmp_path/'final-validation.html').read_text()
    with pytest.raises(ValueError, match='last completed'):
        history.finish(results, steps=9, updates=4)
