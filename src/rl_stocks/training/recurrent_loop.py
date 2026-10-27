"""Bounded fresh-run recurrent SAC loop; actual updates remain Colab-only."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.recurrent_sac import evaluate_recurrent_sac
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.evaluation.runner import selection_score
from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.training.curves import append_record, curve_record, render_curve
from rl_stocks.training.checkpoint_export import export_artifacts
from rl_stocks.training.recurrent_collector import RecurrentCollector
from rl_stocks.training.recurrent_payload import RecurrentPayloadStore
from rl_stocks.training.recurrent_replay import _integer
from rl_stocks.training.recurrent_sac_checkpoint import save_checkpoint,validate_learner
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner, require_colab_cpu


@dataclass(frozen=True)
class LoopConfig:
    transitions: int
    warmup_steps: int
    train_every: int
    gradient_steps: int
    burn_in: int
    unroll: int
    windows_per_batch: int
    replay_capacity: int
    episode_steps: int
    evaluate_every: int
    seed: int
    hidden_size: int = 64
    start_sampling: str = 'stock_etf_cycle_windows'

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != 'start_sampling':
                _integer(name, value, 0 if name in ('seed', 'burn_in') else 1)
        if (self.burn_in+self.unroll > self.episode_steps
                or self.replay_capacity < 2*(self.burn_in+self.unroll)
                or self.warmup_steps < self.burn_in+self.unroll
                or self.maximum_updates < 1 or self.evaluate_every > self.transitions
                or self.start_sampling not in ('legacy_uniform', 'stock_etf_cycle_windows')):
            raise ValueError('invalid bounded replay/episode/update/evaluation schedule')

    @property
    def maximum_updates(self):
        return ((self.transitions-self.warmup_steps)//self.train_every)*self.gradient_steps

    def updates_due(self, transitions):
        return transitions > self.warmup_steps and (transitions-self.warmup_steps) % self.train_every == 0


class _TrainingEngine:
    """Scheduling only; test spies never bypass the learner's own update guard."""
    def __init__(self, collector, learner, config):
        if (collector.transitions != 0 or learner.updates != 0 or learner.failed
                or collector.warmup_steps != config.warmup_steps
                or learner.max_updates != config.maximum_updates):
            raise ValueError('fresh matching collector/learner allowances required')
        self.collector, self.learner, self.config = collector, learner, config
        self.rng = np.random.default_rng(config.seed)
        self.failed = False
        self.last_update = {}

    def advance(self):
        if self.failed or self.collector.transitions >= self.config.transitions:
            raise RuntimeError('fresh run exhausted or failed; no retry')
        try:
            transition = self.collector.collect(policy_revision=self.learner.updates)
            step = self.collector.transitions
            if self.config.updates_due(step):
                for _ in range(self.config.gradient_steps):
                    sequences = self.collector.store.index.sample(
                        batch_size=self.config.windows_per_batch, burn_in=self.config.burn_in,
                        unroll=self.config.unroll, rng=self.rng)
                    batch = self.collector.store.gather(sequences)
                    previous = self.learner.updates
                    self.last_update = self.learner.update(batch)
                    if self.learner.updates != previous+1:
                        raise ValueError('one actual learner update required per scheduled call')
            return transition
        except BaseException:
            self.failed = True
            raise

    def evaluation_due(self):
        step = self.collector.transitions
        return step > 0 and (step % self.config.evaluate_every == 0 or step == self.config.transitions)


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    try:
        temporary.write_bytes(canonical(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class _CheckpointHistory:
    def __init__(self, output, *, clock=time.monotonic):
        self.output = Path(output)
        self.history = []
        self.best_step = None
        self.best_score = -float('inf')
        self.clock = clock
        self.started = clock()

    def record(self, results, *, steps, updates, training, checkpoint):
        # A checkpoint must be fully saved before its curve/pointer becomes visible.
        if not (self.output/checkpoint/'manifest.json').is_file():
            raise ValueError('durable completed checkpoint required before evaluation record')
        if self.history and steps <= self.history[-1]['steps']:
            raise ValueError('strictly increasing checkpoint transition counts required')
        score = selection_score(results)
        record = curve_record(results, self.history, steps, 'post_update',
                              self.clock()-self.started, training)
        record.update(learner_updates=updates, checkpoint=checkpoint)
        append_record(self.output/'learning-curve.jsonl', record)
        self.history.append(record)
        if score > self.best_score:  # ties keep the earlier declared selection
            self.best_step, self.best_score = steps, score
            _atomic_json(self.output/'selected.json', dict(steps=steps, score=score,
                                                          updates=updates, checkpoint=checkpoint))
            _atomic_json(self.output/'selected-validation.json', results)
        return record

    def finish(self, results, *, steps, updates):
        if (not self.history or self.history[-1]['steps'] != steps
                or self.history[-1]['learner_updates'] != updates):
            raise ValueError('final must match the last completed checkpoint/evaluation')
        record = self.history[-1]
        _atomic_json(self.output/'final.json', dict(steps=steps, updates=updates,
                     checkpoint=record['checkpoint'], score=record['score']))
        _atomic_json(self.output/'final-validation.json', results)
        from rl_stocks.reporting.html import report
        selected = json.loads((self.output/'selected-validation.json').read_text())
        report(selected, self.output/'selected-validation.html', 'Selected development checkpoint', self.history)
        report(results, self.output/'final-validation.html', 'Final development checkpoint', self.history)


def validate_dataset(dataset, expected_identity):
    if digest(dataset) != expected_identity:
        raise ValueError('immutable research dataset identity mismatch')
    if dataset.get('schema') != 'daily-indicators-v1' or len(dataset.get('names', [])) != 7:
        raise ValueError('baseline seven-field dataset required')
    normalizer = normalizer_from(dataset['normalizer'])
    if normalizer.names != tuple(dataset['names']):
        raise ValueError('fixed normalizer feature order required')
    training, validation = [], []
    for episode in dataset['episodes'].values():
        split = episode['split']
        if split not in ('train', 'validation'):
            raise ValueError('retired/test episodes excluded from research payload')
        low, high = ('2010-01-01', '2021-01-01') if split == 'train' else ('2021-01-01', '2024-01-01')
        sessions = [b['session'] for b in episode['bars']]
        if (len(sessions) < 2 or any(not low <= s < high for s in sessions)
                or any(b <= a for a, b in zip(sessions, sessions[1:]))):
            raise ValueError('fixed chronological training/development dates required')
        (training if split == 'train' else validation).append(episode)
    if not training or not validation:
        raise ValueError('both training and validation episodes required')
    return normalizer, training


def validate_initial_learner(learner,config):
    if type(learner) is not RecurrentSACLearner:
        raise ValueError('native unfitted recurrent SAC learner required')
    validate_learner(learner)
    if (learner.updates!=0 or learner.max_updates!=config.maximum_updates
            or learner.networks.observation_size!=21 or learner.networks.hidden_size!=config.hidden_size
            or learner.learning_rate!=.0003 or learner.gamma!=.99 or learner.tau!=.005
            or learner.log_alpha.item()!=0):
        raise ValueError('fresh matching portable recurrent initialization required')


def run_colab_training(dataset, output, config: LoopConfig, *, exposure_mode: str, lineage: dict,
                       initial_learner=None):
    require_colab_cpu()  # before dataset construction, output, RNG or updates
    if type(config) is not LoopConfig:
        raise ValueError('validated LoopConfig required')
    if initial_learner is not None:
        validate_initial_learner(initial_learner,config)
    normalizer, training = validate_dataset(dataset, lineage['dataset'])
    from rl_stocks.evaluation.exposure import EXPOSURE_MODES
    if exposure_mode not in EXPOSURE_MODES:
        raise ValueError('registered evaluation exposure mode required')
    output = Path(output)
    if output.parent.name != 'artifacts':
        raise ValueError('run must live directly inside its owned artifacts root')
    if output.exists():
        raise FileExistsError('fresh training output required; automatic continuation unsupported')
    output.mkdir(parents=True)
    env = learner = engine = None
    try:
        torch.set_num_threads(1)
        if initial_learner is None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            networks = RecurrentSACNetworks(21, config.hidden_size)
            learner = RecurrentSACLearner(networks, max_updates=config.maximum_updates)
        else:
            learner=initial_learner
            networks=learner.networks
        env = TradingEnv(training, normalizer, dataset['names'], random_start=True,
                         episode_steps=config.episode_steps, start_sampling=config.start_sampling,
                         observation_profile='execution_full')
        store = RecurrentPayloadStore(config.replay_capacity, 21)
        collector = RecurrentCollector(env, networks, store, seed=config.seed,
                                        warmup_steps=config.warmup_steps,
                                        max_episode_steps=config.episode_steps)
        engine = _TrainingEngine(collector, learner, config)
        history = _CheckpointHistory(output)
        _atomic_json(output/'run.json', dict(status='running', config=asdict(config), lineage=lineage,
                                            exposure_mode=exposure_mode, exact_resume_supported=False))
        save_checkpoint(learner, output/'initial', lineage=lineage)
        export_artifacts(output.parent)
        while collector.transitions < config.transitions:
            engine.advance()
            if engine.evaluation_due():
                step = collector.transitions
                name = f'checkpoint-{step}'
                save_checkpoint(learner, output/name, lineage=lineage)
                export_artifacts(output.parent)  # publish learned state before expensive validation
                results = evaluate_recurrent_sac(networks, dataset, exposure_mode=exposure_mode, config=env.config)
                validate_development_report(results, dataset, env.config)
                history.record(results, steps=step, updates=learner.updates,
                               training=engine.last_update, checkpoint=name)
                _atomic_json(output/'training-execution.json', env.execution_statistics())
                if env.start_sampler is not None:
                    _atomic_json(output/'training-sampling.json', env.sampling_statistics())
                render_curve(history.history, output/'learning-curve.html')
                export_artifacts(output.parent)
        if learner.updates != config.maximum_updates:
            raise ValueError('final scheduled learner update count mismatch')
        history.finish(results, steps=collector.transitions, updates=learner.updates)
        _atomic_json(output/'run.json', dict(status='complete', config=asdict(config), lineage=lineage,
                        exposure_mode=exposure_mode, exact_resume_supported=False,
                        selected_step=history.best_step, selected_score=history.best_score,
                        actual_transitions=collector.transitions, actual_updates=learner.updates,
                        final_score=history.history[-1]['score']))
        export_artifacts(output.parent)
        return history.history
    except BaseException as error:
        _atomic_json(output/'failure.json', dict(error_type=type(error).__name__, automatic_retry=False,
              actual_updates=None if learner is None else learner.updates,
              transitions=None if engine is None else engine.collector.transitions,
              partial_update_failed=None if learner is None else learner.failed))
        export_artifacts(output.parent)
        raise
    finally:
        if env is not None:
            env.close()
