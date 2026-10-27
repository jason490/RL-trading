"""Independent read-only native-state, batch and RNG restoration; never optimize."""
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.crossq_restoration import _same_restored, _raw_entropy
from rl_stocks.training.crossq_state import model_record
from rl_stocks.training.distributional_critics import rng_record, capture_global_rng, restore_global_rng
from rl_stocks.training import sac_update_worker as worker
from rl_stocks.training.sac_update_input import PROTOCOL_SHA256, protocol, validate_payload
from rl_stocks.training.shared_initialization import tensor_fingerprint


FILES = ('record.json', 'post-update.zip', 'numerical-state.npz', 'sampled-batch.npz', 'torch-rng.npz')


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError('unique numerical archive fields required')
        return {k:archive[k].copy() for k in archive.files}


def torch_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def validate_counts(record):
    audit = record['audit']
    if (record['final']['algorithm'] != 'sac' or record['final']['seed'] != 11
            or type(record['final']['updates']) is not int or record['final']['updates'] != 1
            or record['final']['num_timesteps'] != 1001
            or audit['optimizer_steps'] != dict(actor=1, critic=1, entropy=1)
            or any(type(v) is not int for v in audit['optimizer_steps'].values())
            or audit['optimizer_step_attempts'] != dict(actor=1, critic=1, entropy=1)
            or any(type(v) is not int for v in audit['optimizer_step_attempts'].values())
            or audit['step_order'] != ['entropy', 'critic', 'actor']
            or type(audit['replay_batches']) is not int or audit['replay_batches'] != 1
            or audit['one_step_discounts_none'] is not True):
        raise ValueError('one actual native update, batch and three completed Adam steps required')
    for state in record['final']['optimizers'].values():
        if (state['state_count'] != state['parameter_tensors'] or not state['per_parameter_steps']
                or any(type(v) is not int or v != 1 for v in state['per_parameter_steps'].values())):
            raise ValueError('actual saved per-parameter Adam step1 required')


def verify_condition(folder, payload, arm):
    """Only previously retrieved task-owned archives with trusted source may be loaded."""
    validate_payload(payload)
    if digest(payload) != worker.INPUT_ID or arm not in protocol()['arms']:
        raise ValueError('bound diagnostic input and condition required')
    folder = Path(folder)
    protected = {str(folder / name):checksum(folder / name) for name in FILES}
    record = json.loads((folder / 'record.json').read_text())
    validate_counts(record)
    with worker.preserved_rng():
        initial, expected_initial = worker.prepare(payload, arm)
        try:
            if canonical(expected_initial) != canonical(record['initial']):
                raise ValueError('exact original native initial tensors/settings/RNG required')
            rng = read_npz(folder / 'torch-rng.npz')
            if set(rng) != {'initial', 'before_update', 'after_update', 'after_action'}:
                raise ValueError('all four actual Torch RNG snapshots required')
            for value in rng.values():
                if value.dtype != np.uint8 or value.shape != tuple(torch.get_rng_state().shape):
                    raise ValueError('native CPU Torch RNG dtype/shape required')
                torch.Generator(device='cpu').set_state(torch.from_numpy(value.copy()))
            expected_hashes = dict(initial=record['initial']['rng']['torch'],
                before_update=record['before_rng']['torch'], after_update=record['after_update_rng']['torch'],
                after_action=record['after_rng']['torch'])
            if any(torch_sha(rng[k]) != expected_hashes[k] for k in rng):
                raise ValueError('saved Torch RNG snapshots must match actual recorded streams')
            before = worker.stochastic_action(initial, payload['arrays']['observations']['values'][1000][0])
            if canonical(rng_record()) != canonical(record['before_rng']):
                raise ValueError('forward-only preupdate RNG advancement mismatch')
            if record['audit']['batch_rng'] != record['before_rng']:
                raise ValueError('actual replay sampling must start from preupdate streams')
            sample = initial.replay_buffer.sample(256)
            expected_batch = {k:v.detach().cpu().numpy() for k,v in sample._asdict().items() if v is not None}
            batch = read_npz(folder / 'sampled-batch.npz')
            if set(batch) != set(expected_batch):
                raise ValueError('complete native one-step batch fields required')
            for k, value in batch.items():
                if (fingerprint(value) != fingerprint(expected_batch[k])
                        or fingerprint(value) != record['audit']['sampled_field_sha256'][k]):
                    raise ValueError('saved sampled batch differs from native replay reconstruction')
            after_sample = rng_record()
            post_sample_globals = capture_global_rng()
            with torch.no_grad():
                initial.actor.action_log_prob(sample.observations)
                initial.actor.action_log_prob(sample.next_observations)
            if torch_sha(torch.get_rng_state().numpy()) != expected_hashes['after_update']:
                raise ValueError('forward-only two-actor-call Torch advancement mismatch')
            for name in ('numpy', 'python'):
                if (record['after_update_rng'][name] != after_sample[name]
                        or record['after_rng'][name] != after_sample[name]):
                    raise ValueError('non-Torch streams may advance only through the native replay sample')
        finally:
            initial.env.close()
        model = SAC.load(folder / 'post-update.zip', device='cpu')
        actual = model_record(model)
        if not _same_restored(actual, record['final']):
            raise ValueError('native numerical checkpoint weights/Adam/settings did not restore')
        _raw_entropy(model, folder / 'post-update.zip')
        saved = read_npz(folder / 'numerical-state.npz')
        actual_arrays = worker.state_arrays(model)
        if set(saved) != set(actual_arrays):
            raise ValueError('complete saved policy/target/entropy/Adam array set required')
        if any(fingerprint(saved[k]) != fingerprint(actual_arrays[k]) for k in saved):
            raise ValueError('actual numerical tensors/moments differ from NPZ evidence')
        if tensor_fingerprint(saved) != record['state_sha256']:
            raise ValueError('actual full numerical state fingerprint mismatch')
        for key in ('before', 'after'):
            action = np.asarray(record[key + '_action'], dtype=np.float32)
            if action.shape != (4,) or not np.isfinite(action).all() or fingerprint(action) != record[key + '_action_sha256']:
                raise ValueError('finite exact four-component recorded action required')
        restore_global_rng(post_sample_globals)
        torch.set_rng_state(torch.from_numpy(rng['after_update'].copy()))
        after = worker.stochastic_action(model, payload['postupdate_observation'])
        if canonical(rng_record()) != canonical(record['after_rng']):
            raise ValueError('saved postupdate stochastic action RNG advancement mismatch')
        torch.set_rng_state(torch.from_numpy(rng['after_update'].copy()))
        repeat = worker.stochastic_action(model, payload['postupdate_observation'])
        np.testing.assert_array_equal(after, repeat)
        if tensor_fingerprint(worker.state_arrays(model)) != record['state_sha256']:
            raise ValueError('read-only prediction changed learned numerical state')
    if any(checksum(p) != value for p, value in protected.items()):
        raise ValueError('owned numerical artifacts changed during verification')
    # Remote action arithmetic may round differently on the verifier CPU. Report
    # that observation; do not turn it into an undeclared tolerance or fit again.
    return dict(actual_saved_model_and_adam_restored=True, numerical_npz_exact=True,
                sampled_batch_reconstructed=True, forward_only_rng_replay=True,
                local_prediction_repeat_exact=True, source_files_unchanged=True,
                remote_preupdate_action_recomputed_exact=fingerprint(before) == record['before_action_sha256'],
                remote_postupdate_action_recomputed_exact=fingerprint(after) == record['after_action_sha256'],
                raw_log_entropy_exact=True, protected_source_sha256=protected,
                training_performed=False, optimizer_steps=0)


def verify_directory(folder, payload):
    folder = Path(folder)
    summary_path = folder / 'summary.json'
    protected = checksum(summary_path)
    summary = json.loads(summary_path.read_text())
    backend = summary['execution_backend']
    if (backend.get('schema') != 'numerical-execution-v1' or backend.get('selected_device') != 'cpu'
            or type(backend.get('torch_intraop_threads')) is not int or backend['torch_intraop_threads'] != 1
            or backend.get('gpu_name') is not None):
        raise ValueError('actual fixed single-thread CPU numerical backend record required')
    if (any(type(summary[k]) is not int for k in ('total_model_gradient_updates', 'total_adam_steps', 'environment_transitions'))
            or summary['protocol_sha256'] != PROTOCOL_SHA256 or summary['input_identity'] != worker.INPUT_ID
            or summary['status'] != 'complete' or summary['total_model_gradient_updates'] != 2
            or summary['total_adam_steps'] != 6 or summary['environment_transitions'] != 0
            or summary['performance_evaluated'] is not False or summary['crossq_stop_cleared'] is not False
            or summary['root_cause_established'] is not False
            or set(summary['records']) != set(protocol()['arms'])):
        raise ValueError('fixed complete numerical-only diagnostic summary required')
    proofs, arrays = {}, {}
    for arm in protocol()['arms']:
        record = json.loads((folder / arm / 'record.json').read_text())
        if canonical(record) != canonical(summary['records'][arm]):
            raise ValueError('summary and physical condition record mismatch')
        proofs[arm] = verify_condition(folder / arm, payload, arm)
        arrays[arm] = read_npz(folder / arm / 'numerical-state.npz')
    a, b = protocol()['arms']
    left, right = summary['records'][a], summary['records'][b]
    if set(arrays[a]) != set(arrays[b]):
        raise ValueError('same numerical state schema across both conditions required')
    expected = dict(
        initial_policy_exact=left['initial']['policy_tensor_sha256'] == right['initial']['policy_tensor_sha256'],
        preupdate_action_exact=left['before_action_sha256'] == right['before_action_sha256'],
        preupdate_rng_exact=left['before_rng'] == right['before_rng'],
        sampled_batch_exact=left['audit']['sampled_field_sha256'] == right['audit']['sampled_field_sha256'],
        postupdate_different_fields=[k for k in sorted(arrays[a]) if fingerprint(arrays[a][k]) != fingerprint(arrays[b][k])],
        postupdate_tensor_state_exact=left['state_sha256'] == right['state_sha256'],
        postupdate_rng_exact=left['after_rng'] == right['after_rng'],
        postupdate_action_exact=left['after_action_sha256'] == right['after_action_sha256'])
    if (canonical(expected) != canonical(summary['comparison'])
            or not all(expected[k] for k in ('initial_policy_exact', 'preupdate_action_exact', 'preupdate_rng_exact', 'sampled_batch_exact'))
            or checksum(summary_path) != protected):
        raise ValueError('independently recomputed matched comparison required')
    return dict(conditions=proofs, comparison=expected, saved_artifacts_verified=True,
                training_performed=False, optimizer_steps=0, performance_evaluated=False,
                crossq_stop_cleared=False, root_cause_established=False)
