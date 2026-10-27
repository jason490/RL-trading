"""Two-update Colab numerical worker; preparation and observation are independently testable."""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.logger import Logger
import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.execution_provenance import execution_backend
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.training.crossq_assets import initialization_entry, initialization_spec, portable_path
from rl_stocks.training.crossq_inspection import SpacesOnly
from rl_stocks.training.crossq_state import model_record
from rl_stocks.training.distributional_critics import capture_global_rng, restore_global_rng, rng_record
from rl_stocks.training.sac_update_input import PROTOCOL_SHA256, protocol, validate_payload
from rl_stocks.training.shared_initialization import load_initialization, tensor_fingerprint

INPUT_ID = '7f50968c0272bda3d5b5a75861e5ea25d2f4e7d328a1b014af7de219a0d5a8a7'


@contextmanager
def preserved_rng():
    previous = capture_global_rng()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            restore_global_rng(previous)


def prepare(payload, arm):
    arrays = validate_payload(payload)
    spec = protocol()
    if digest(payload) != INPUT_ID or arm not in spec['arms']:
        raise ValueError('exact bound numerical input and diagnostic condition required')
    model = SAC('MlpPolicy', SpacesOnly(spec), seed=11, device='cpu', verbose=0,
                **deepcopy(payload['arm_parameters'][arm]))
    entry, globals_record = initialization_entry('sac', 11, False)
    load_initialization(model, portable_path(entry['file']), entry)
    restore_global_rng(globals_record)
    model._research_initialization = initialization_spec('sac', 11, False)
    initial = model_record(model)
    if (initial['policy_tensor_sha256'] != entry['policy_tensor_sha256']
            or initial['rng'] != entry['global_rng_sha256']
            or initial['alpha'] != 1 or initial['updates'] != 0 or any(o['state_count'] for o in initial['optimizers'].values())):
        raise ValueError('actual copied original tensors/RNG and empty native optimizers required')
    for name, value in arrays.items():
        np.copyto(getattr(model.replay_buffer, name)[:spec['replay_rows']], value)
    model.replay_buffer.pos = spec['replay_rows']
    model.num_timesteps = spec['replay_rows']
    model._current_progress_remaining = spec['progress_remaining']
    model.set_logger(Logger(folder=None, output_formats=[]))
    return model, initial


class SingleUpdateAudit:
    """Fail before any second replay sample or second step of an optimizer."""
    def __init__(self, model):
        if type(model) is not SAC or model._n_updates != 0:
            raise ValueError('fresh native SAC diagnostic required')
        self.model = model
        self.steps = {name: 0 for name in ('actor', 'critic', 'entropy')}
        self.attempts = dict(self.steps)
        self.step_order = []
        self.batch = None
        self.batch_rng = None
        self.handles = []
        self.closed = False
        self.buffer = model.replay_buffer
        if 'sample' in self.buffer.__dict__:
            raise ValueError('unmodified native replay sampling required')
        self.original_sample = self.buffer.sample
        optimizers = dict(actor=model.actor.optimizer, critic=model.critic.optimizer, entropy=model.ent_coef_optimizer)
        if any(o.state for o in optimizers.values()):
            raise ValueError('empty native optimizers before diagnostic update required')
        for name, optimizer in optimizers.items():
            self.handles.append(optimizer.register_step_pre_hook(
                lambda optimizer, args, kwargs, name=name: self._before_step(name)))
            self.handles.append(optimizer.register_step_post_hook(
                lambda optimizer, args, kwargs, name=name: self._after_step(name)))
        self.buffer.sample = self._sample

    def _before_step(self, name):
        if self.closed or self.attempts[name] != 0:
            raise ValueError('diagnostic permits exactly one step per native optimizer')
        self.attempts[name] += 1
        self.step_order.append(name)

    def _after_step(self, name):
        if self.closed or self.attempts[name] != 1 or self.steps[name] != 0:
            raise ValueError('one completed step must follow its single native attempt')
        self.steps[name] += 1

    def _sample(self, *args, **kwargs):
        if self.closed or self.batch is not None:
            raise ValueError('diagnostic permits exactly one native replay batch')
        size = args[0] if args else kwargs.get('batch_size')
        if type(size) is not int or size != 256:
            raise ValueError('fixed diagnostic replay batch256 required')
        self.batch_rng = rng_record()
        sample = self.original_sample(*args, **kwargs)
        expected = dict(observations=(256, 21), next_observations=(256, 21),
                        actions=(256, 4), rewards=(256, 1), dones=(256, 1))
        fields = sample._asdict()
        if set(fields) != {*expected, 'discounts'} or fields['discounts'] is not None:
            raise ValueError('native one-step replay with no n-step discounts required')
        self.batch = {k:fields[k].detach().cpu().numpy().copy() for k in expected}
        if any(a.shape != expected[k] or a.dtype != np.float32 or not np.isfinite(a).all()
               for k, a in self.batch.items()):
            raise ValueError('finite actual float32 sampled replay geometry required')
        return sample

    def record(self):
        return dict(optimizer_steps=dict(self.steps), optimizer_step_attempts=dict(self.attempts),
                    step_order=list(self.step_order),
                    replay_batches=int(self.batch is not None), one_step_discounts_none=self.batch is not None, batch_rng=self.batch_rng,
                    sampled_field_sha256=None if self.batch is None else {k:fingerprint(v) for k,v in self.batch.items()})

    def close(self):
        if not self.closed:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            del self.buffer.__dict__['sample']
            self.closed = True


def state_arrays(model):
    result = {'policy.' + k:v.detach().cpu().numpy().copy() for k,v in model.policy.state_dict().items()}
    result['raw_log_entropy'] = model.log_ent_coef.detach().cpu().numpy().copy()
    for branch, optimizer, named in (
        ('actor', model.actor.optimizer, list(model.actor.named_parameters())),
        ('critic', model.critic.optimizer, list(model.critic.named_parameters())),
        ('entropy', model.ent_coef_optimizer, [('log_ent_coef', model.log_ent_coef)]),
    ):
        for name, parameter in named:
            for field, tensor in optimizer.state.get(parameter, {}).items():
                if field not in ('step', 'exp_avg', 'exp_avg_sq') or not isinstance(tensor, torch.Tensor):
                    raise ValueError('native Adam tensor state required')
                result[f'adam.{branch}.{name}.{field}'] = tensor.detach().cpu().numpy().copy()
    if any(not np.isfinite(a).all() for a in result.values()):
        raise ValueError('finite actual numerical state required')
    return result


def stochastic_action(model, observation):
    observation = np.asarray(observation, dtype=np.float32).copy()
    action = model.predict(observation, deterministic=False)[0]
    return model.policy.scale_action(action).reshape(4).astype(np.float32)


def require_colab():
    if (sys.version_info[:2] != (3, 13) or not os.environ.get('COLAB_RELEASE_TAG')
            or Path.cwd().resolve() != Path('/content/rl-stocks')
            or torch.cuda.is_available()):
        raise RuntimeError('owned default-kernel Colab CPU diagnostic required; local updates forbidden')


def execute_colab(payload, output):
    require_colab()  # Before construction, output mutation or any native update.
    validate_payload(payload)
    if digest(payload) != INPUT_ID:
        raise ValueError('bound real numerical diagnostic input required')
    spec = protocol()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'input.json').write_bytes(canonical(payload))
    records, states = {}, {}
    with preserved_rng():
        torch.set_num_threads(1)  # Same explicit intra-op setting in both historical runners.
        backend = execution_backend('cpu')
        for arm in spec['arms']:
            folder = output / arm
            folder.mkdir()
            model, initial = prepare(payload, arm)
            try:
                torch_states = dict(initial=torch.get_rng_state().numpy().copy())
                before_action = stochastic_action(model, payload['arrays']['observations']['values'][1000][0])
                before_rng = rng_record()
                torch_states['before_update'] = torch.get_rng_state().numpy().copy()
                reference = records.get(spec['arms'][0])
                if reference is not None and (
                        reference['initial']['policy_tensor_sha256'] != initial['policy_tensor_sha256']
                        or reference['before_action_sha256'] != fingerprint(before_action)
                        or reference['before_rng'] != before_rng):
                    raise ValueError('matched starts/actions/RNG required before second condition update')
                audit = SingleUpdateAudit(model)
                try:
                    model.train(gradient_steps=1, batch_size=256)  # Only reachable after require_colab.
                    after_update_rng = rng_record()
                    torch_states['after_update'] = torch.get_rng_state().numpy().copy()
                    final = model_record(model)
                    evidence = audit.record()
                    if (model._n_updates != 1 or evidence['replay_batches'] != 1
                            or set(evidence['optimizer_steps'].values()) != {1}
                            or set(evidence['optimizer_step_attempts'].values()) != {1}
                            or any(set(o['per_parameter_steps'].values()) != {1}
                                   or o['state_count'] != o['parameter_tensors'] for o in final['optimizers'].values())):
                        raise ValueError('one actual update and all three actual Adam steps required')
                    after_action = stochastic_action(model, payload['postupdate_observation'])
                    torch_states['after_action'] = torch.get_rng_state().numpy().copy()
                    arrays = state_arrays(model)
                    states[arm] = arrays
                    record = dict(initial=initial, final=final, audit=evidence,
                        before_action=before_action.tolist(), before_action_sha256=fingerprint(before_action), before_rng=before_rng,
                        after_update_rng=after_update_rng, after_action=after_action.tolist(),
                        after_action_sha256=fingerprint(after_action), after_rng=rng_record(),
                        state_sha256=tensor_fingerprint(arrays))
                    records[arm] = record
                    np.savez_compressed(folder / 'torch-rng.npz', **torch_states)
                    np.savez_compressed(folder / 'numerical-state.npz', **arrays)
                    np.savez_compressed(folder / 'sampled-batch.npz', **audit.batch)
                except BaseException as error:
                    (folder / 'failure.json').write_bytes(canonical(dict(
                        failure_type=type(error).__name__, audit=audit.record(),
                        observed_native_updates=model._n_updates, environment_transitions=0)))
                    raise
                finally:
                    audit.close()
                model.save(folder / 'post-update.zip')
                (folder / 'record.json').write_bytes(canonical(record))
                checkpoint=os.environ.get('RL_CHECKPOINT_ARCHIVE')
                if checkpoint:
                    checkpoint=Path(checkpoint)
                    temporary=checkpoint.with_suffix('.tmp')
                    with tarfile.open(temporary,'w:gz') as archive:
                        archive.add(output.parent,arcname='artifacts')
                    temporary.replace(checkpoint)
            finally:
                model.env.close()
        a, b = spec['arms']
        comparison = dict(
            initial_policy_exact=records[a]['initial']['policy_tensor_sha256'] == records[b]['initial']['policy_tensor_sha256'],
            preupdate_action_exact=records[a]['before_action_sha256'] == records[b]['before_action_sha256'],
            preupdate_rng_exact=records[a]['before_rng'] == records[b]['before_rng'],
            sampled_batch_exact=records[a]['audit']['sampled_field_sha256'] == records[b]['audit']['sampled_field_sha256'],
            postupdate_different_fields=[k for k in sorted(states[a])
                                         if fingerprint(states[a][k]) != fingerprint(states[b][k])],
            postupdate_tensor_state_exact=records[a]['state_sha256'] == records[b]['state_sha256'],
            postupdate_rng_exact=records[a]['after_rng'] == records[b]['after_rng'],
            postupdate_action_exact=records[a]['after_action_sha256'] == records[b]['after_action_sha256'],
        )
        if not all(comparison[k] for k in ('initial_policy_exact', 'preupdate_action_exact', 'preupdate_rng_exact', 'sampled_batch_exact')):
            raise ValueError('matched numerical input/initial state/preupdate randomness required')
    result = dict(protocol_sha256=PROTOCOL_SHA256, input_identity=INPUT_ID, status='complete',
                  execution_backend=backend, records=records, comparison=comparison,
                  total_model_gradient_updates=2, total_adam_steps=6, environment_transitions=0,
                  performance_evaluated=False, crossq_stop_cleared=False, root_cause_established=False)
    (output / 'summary.json').write_bytes(canonical(result))
    return result
