"""Actual four-update-per-arm Colab smoke; never runs optimization locally."""
from copy import deepcopy
import os
import tarfile
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.core.execution_provenance import execution_backend
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.rl.recurrent_sac.prediction import FrozenRecurrentActor
from rl_stocks.training.crossq_state import model_record
from rl_stocks.training.distributional_critics import restore_global_rng
from rl_stocks.training.recurrent_sac_checkpoint import load_checkpoint, save_checkpoint
from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
from rl_stocks.training.recurrent_sac_smoke import PROTOCOL_SHA256, prepare_smoke, protocol
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.shared_initialization import tensor_fingerprint

INITIAL_RECURRENT_SHA256 = '6832b68b7141e304ecf582cf80dae567cb5318dee8df651de535b36dae438649'


def network_hash(networks):
    return tensor_fingerprint({k: v.detach().cpu().numpy() for k, v in networks.state_dict().items()})


def write(path, value):
    Path(path).write_bytes(canonical(value))


def export_checkpoint(output):
    archive_path = os.environ.get('RL_CHECKPOINT_ARCHIVE')
    if archive_path:
        archive_path = Path(archive_path)
        temporary = archive_path.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as archive:
            archive.add(Path(output).parent, arcname='artifacts')
        temporary.replace(archive_path)


class OptimizerAudit:
    """Independent native optimizer pre/post hooks bound steps and record gradients."""
    def __init__(self, optimizers, maximum, *, networks=None):
        self.maximum = maximum
        self.steps = dict.fromkeys(('entropy', 'critic', 'actor'), 0)
        self.attempts = dict(self.steps)
        self.events = []
        self.networks = networks
        self.handles = []
        for role, optimizer in optimizers.items():
            self.handles.append(optimizer.register_step_pre_hook(
                lambda opt, args, kwargs, role=role: self.before(role, opt)))
            self.handles.append(optimizer.register_step_post_hook(
                lambda opt, args, kwargs, role=role: self.after(role)))

    def before(self, role, optimizer):
        expected = ('entropy', 'critic', 'actor')[sum(self.attempts.values()) % 3]
        if role != expected or self.attempts[role] >= self.maximum:
            raise ValueError('unexpected optimizer order or extra update')
        self.attempts[role] += 1
        params = [p for group in optimizer.param_groups for p in group['params']]
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError('observed missing/nonfinite native gradients')
        event = dict(role=role, finite_gradients=True, parameter_count=len(params))
        if self.networks is not None:
            if any(p.grad is not None for p in self.networks.targets.parameters()):
                raise ValueError('target gradients forbidden')
            if role == 'actor' and any(p.grad is not None for p in self.networks.critics.parameters()):
                raise ValueError('actor phase must not accumulate critic gradients')
            memories = ([self.networks.actor.memory] if role == 'actor' else
                        [c.memory for c in self.networks.critics] if role == 'critic' else [])
            event['nonzero_memory_gradients'] = [any(bool(p.grad.abs().max() > 0)
                                                    for p in memory.parameters()) for memory in memories]
            if not all(event['nonzero_memory_gradients']):
                raise ValueError('each active recurrent memory needs an observed gradient')
        self.events.append(event)

    def after(self, role):
        self.steps[role] += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def verify(self):
        expected = dict.fromkeys(self.steps, self.maximum)
        if self.steps != expected or self.attempts != expected:
            raise ValueError('actual native optimizer counts mismatch')


def prediction_parity(learner, restored, arrays):
    a = FrozenRecurrentActor(learner.networks, observation_schema='execution-full-v1-normalized21')
    b = FrozenRecurrentActor(restored.networks, observation_schema='execution-full-v1-normalized21')
    state_a = state_b = None
    previous = None
    records = []
    for row in protocol()['prediction_rows']:
        reset = previous is None or row != previous + 1 or bool(arrays['dones'][previous, 0])
        obs = arrays['observations'][row, 0]
        action_a, state_a = a.predict(obs, observation_schema='execution-full-v1-normalized21',
                                      state=state_a, episode_start=reset)
        action_b, state_b = b.predict(obs, observation_schema='execution-full-v1-normalized21',
                                      state=state_b, episode_start=reset)
        if not np.array_equal(action_a, action_b) or not np.array_equal(state_a, state_b):
            raise ValueError('learned frozen carried/reset parity failed')
        records.append(dict(row=row, reset=reset, action=action_a.tolist(),
                            state_sha256=tensor_fingerprint({'state': state_a})))
        previous = row
    return records


def execute_colab(payload, output):
    require_colab_cpu()  # reject before construction/output/optimizer mutation
    spec = protocol()
    output = Path(output)
    if output.exists():
        raise FileExistsError('smoke outputs are immutable; no automatic retry')
    prepared = None
    audits = []
    output.mkdir(parents=True)
    try:
        with preserved_rng():
            torch.set_num_threads(1)
            prepared = prepare_smoke(payload)
            learner, control = prepared.learner, prepared.control
            if network_hash(learner.networks) != INITIAL_RECURRENT_SHA256:
                raise ValueError('actual untrained recurrent initializer mismatch')
            lineage = dict(source='recurrent-sac-smoke-'+PROTOCOL_SHA256,
                           dataset=spec['input_identity'], observation_schema='execution-full-v1-normalized21')
            write(output/'input.json', payload)
            save_checkpoint(learner, output/'recurrent-initial', lineage=lineage)
            control.save(output/'native-initial.zip')
            export_checkpoint(output)
            initial_alpha = learner.log_alpha.detach().clone()
            initial_actor = deepcopy(learner.networks.actor.state_dict())
            initial_critics = deepcopy(learner.networks.critics.state_dict())
            audit = OptimizerAudit(learner.optimizers, 4, networks=learner.networks)
            audits.append(audit)
            recurrent_records = []
            restore_global_rng(prepared.recurrent_rng)
            for batch in prepared.batches:
                old_targets = [p.detach().clone() for p in learner.networks.targets.parameters()]
                record = learner.update(batch)
                for old, current, actual in zip(old_targets, learner.networks.critics.parameters(),
                                                 learner.networks.targets.parameters(), strict=True):
                    expected = old.mul(1 - learner.tau).add(current.detach(), alpha=learner.tau)
                    if not torch.equal(expected, actual):
                        raise ValueError('actual target Polyak arithmetic mismatch')
                record['polyak_exact'] = True
                recurrent_records.append(record)
                write(output/'recurrent-progress.json', recurrent_records)
            audit.verify()
            if (torch.equal(initial_alpha, learner.log_alpha)
                    or not any(not torch.equal(v, learner.networks.actor.state_dict()[k])
                               for k, v in initial_actor.items())
                    or not any(not torch.equal(v, learner.networks.critics.state_dict()[k])
                               for k, v in initial_critics.items())):
                raise ValueError('actual actor/critic/entropy state changes required')
            save_checkpoint(learner, output/'recurrent-final', lineage=lineage)
            restored, _ = load_checkpoint(output/'recurrent-final', expected_lineage=lineage)
            if (network_hash(learner.networks) != network_hash(restored.networks)
                    or not torch.equal(learner.log_alpha, restored.log_alpha)
                    or any(fingerprint(opt.state_dict()) != fingerprint(restored.optimizers[role].state_dict())
                           for role, opt in learner.optimizers.items())):
                raise ValueError('actual recurrent tensors/raw entropy/Adam did not restore exactly')
            parity = prediction_parity(learner, restored, prepared.arrays)
            write(output/'recurrent-predictions.json', parity)
            write(output/'recurrent-audit.json', dict(events=audit.events, steps=audit.steps,
                                                      records=recurrent_records))
            audit.close()
            export_checkpoint(output)
            native_optimizers = dict(entropy=control.ent_coef_optimizer,
                                      critic=control.critic.optimizer, actor=control.actor.optimizer)
            native_audit = OptimizerAudit(native_optimizers, 4)
            audits.append(native_audit)
            samples = []
            original_sample = control.replay_buffer.sample
            def sample(batch_size, env=None):
                if batch_size != 128 or env is not None or len(samples) >= 4:
                    raise ValueError('exactly four matched128-row native samples required')
                sample_index = len(samples)
                samples.append(sample_index)
                return prepared.native_sample(sample_index)
            control.replay_buffer.sample = sample
            try:
                restore_global_rng(prepared.control_rng)
                for _ in range(4):
                    control.train(gradient_steps=1, batch_size=128)
            finally:
                control.replay_buffer.sample = original_sample
            native_audit.verify()
            if control._n_updates != 4 or len(samples) != 4:
                raise ValueError('actual native update/sample counts mismatch')
            native_final = model_record(control)
            if any(set(opt['per_parameter_steps'].values()) != {4}
                   or opt['state_count'] != opt['parameter_tensors']
                   for opt in native_final['optimizers'].values()):
                raise ValueError('actual native per-parameter Adam counts mismatch')
            control.save(output/'native-final.zip')
            from stable_baselines3 import SAC
            native_restored = SAC.load(output/'native-final.zip', device='cpu')
            from rl_stocks.training.crossq_restoration import _same_restored, _raw_entropy
            if not _same_restored(model_record(native_restored), native_final):
                raise ValueError('native saved tensors/moments did not restore')
            _raw_entropy(native_restored, output/'native-final.zip')
            native_actions = []
            for row in spec['prediction_rows']:
                action, _ = control.predict(prepared.arrays['observations'][row, 0], deterministic=True)
                replayed, _ = native_restored.predict(prepared.arrays['observations'][row, 0], deterministic=True)
                if not np.array_equal(action, replayed):
                    raise ValueError('native learned prediction restoration failed')
                native_actions.append(dict(row=row, action=action.tolist()))
            write(output/'native-audit.json', dict(events=native_audit.events, steps=native_audit.steps,
                    final_state=native_final, predictions=native_actions))
            write(output/'loss-rows.json', [rows.tolist() for rows in prepared.loss_rows])
            summary = dict(protocol_sha256=PROTOCOL_SHA256, input_identity=spec['input_identity'],
                           model_updates=8, adam_steps=24, recurrent_updates=learner.updates,
                           native_updates=control._n_updates, prediction_count=len(parity),
                           backend=execution_backend('cpu'), performance_evaluation=False,
                           environment_transitions=0, crossq_stop_cleared=False)
            write(output/'summary.json', summary)
            export_checkpoint(output)
            return summary
    except BaseException as error:
        write(output/'failure.json', dict(error_type=type(error).__name__,
              audits=[dict(steps=a.steps, attempts=a.attempts, events=a.events) for a in audits],
              performance_evaluation=False, automatic_retry=False))
        export_checkpoint(output)
        raise
    finally:
        for audit in audits:
            audit.close()
        if prepared is not None:
            prepared.control.get_env().close()
