"""Physical learned-smoke verification, forward only; never optimizes a model."""
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.training.crossq_restoration import _same_restored, _raw_entropy
from rl_stocks.training.crossq_state import model_record
from rl_stocks.training.recurrent_sac_checkpoint import load_checkpoint
from rl_stocks.training.recurrent_sac_smoke import PROTOCOL_SHA256, prepare_smoke, protocol
from rl_stocks.training.recurrent_sac_smoke_worker import INITIAL_RECURRENT_SHA256, network_hash, prediction_parity
from rl_stocks.training.sac_update_worker import preserved_rng

FILES = {'input.json', 'summary.json', 'recurrent-progress.json', 'recurrent-predictions.json',
         'recurrent-audit.json', 'native-audit.json', 'native-initial.zip', 'native-final.zip', 'loss-rows.json'}
FILES |= {f'{arm}/{name}' for arm in ('recurrent-initial', 'recurrent-final')
          for name in ('manifest.json', 'state.pt')}
FILES |= {f'polyak-{i}.npz' for i in range(1, 5)}


def _json(root, name):
    return json.loads((root/name).read_text())


def verify_audit(value, *, recurrent):
    if value['steps'] != dict.fromkeys(('entropy', 'critic', 'actor'), 4):
        raise ValueError('four completed native steps per optimizer required')
    events = value['events']
    if len(events) != 12 or [e['role'] for e in events] != ['entropy', 'critic', 'actor']*4:
        raise ValueError('actual optimizer phase order mismatch')
    for e in events:
        if e.get('finite_gradients') is not True or type(e['parameter_count']) is not int or e['parameter_count'] < 1:
            raise ValueError('observed finite parameter gradients required')
        if recurrent:
            expected = {'entropy': [], 'critic': [True, True], 'actor': [True]}[e['role']]
            if canonical(e['nonzero_memory_gradients']) != canonical(expected):
                raise ValueError('actual active recurrent memory gradients required')
    if recurrent:
        records = value['records']
        if len(records) != 4:
            raise ValueError('four actual recurrent update records required')
        for i, record in enumerate(records, 1):
            if (type(record['update']) is not int or record['update'] != i
                    or record['polyak_exact'] is not True
                    or record['order'] != ['entropy', 'critic', 'actor', 'targets']
                    or record['completed_steps'] != dict.fromkeys(('entropy', 'critic', 'actor'), i)
                    or not all(type(record[k]) in (float, int) and np.isfinite(record[k])
                               for k in ('actor_loss', 'critic_loss', 'entropy_loss', 'alpha_before', 'alpha_after'))
                    or record['critic_loss'] < 0 or min(record['alpha_before'], record['alpha_after']) <= 0):
                raise ValueError('actual finite phase/update/target evidence mismatch')


def verify_polyak(root, initial, final):
    shapes = {name: tuple(p.shape) for name, p in final.networks.critics.named_parameters()}
    previous = {name: p.detach().clone() for name, p in initial.networks.targets.named_parameters()}
    for i in range(1, 5):
        with np.load(root/f'polyak-{i}.npz', allow_pickle=False) as archive:
            if set(archive.files) != {prefix+'/'+name for prefix in ('old', 'source', 'actual') for name in shapes}:
                raise ValueError('exact Polyak fields required')
            arrays = {k: archive[k].copy() for k in archive.files}
        for name, shape in shapes.items():
            for prefix in ('old', 'source', 'actual'):
                value = arrays[prefix+'/'+name]
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError('finite same-geometry target evidence required')
            old, source, actual = [torch.tensor(arrays[prefix+'/'+name]) for prefix in ('old', 'source', 'actual')]
            if not torch.equal(old, previous[name]):
                raise ValueError('target update lineage mismatch')
            expected = old.mul(1 - final.tau).add(source, alpha=final.tau)
            if not torch.equal(expected, actual):
                raise ValueError('actual saved Polyak equation mismatch')
            previous[name] = actual
            if i == 4:
                if not torch.equal(source, dict(final.networks.critics.named_parameters())[name]):
                    raise ValueError('final online critic does not match target-update source')
    for name, target in final.networks.targets.named_parameters():
        if not torch.equal(previous[name], target):
            raise ValueError('final target does not match saved equation chain')


def verify_directory(folder):
    root = Path(folder)
    spec = protocol()
    with preserved_rng():
        payload = _json(root, 'input.json')
        if digest(payload) != spec['input_identity']:
            raise ValueError('bound original training replay required')
        summary = _json(root, 'summary.json')
        expected = dict(protocol_sha256=PROTOCOL_SHA256, input_identity=spec['input_identity'],
                        model_updates=8, adam_steps=24, recurrent_updates=4, native_updates=4,
                        prediction_count=9, performance_evaluation=False,
                        environment_transitions=0, crossq_stop_cleared=False)
        if canonical({k: summary.get(k) for k in expected}) != canonical(expected):
            raise ValueError('fixed learned smoke summary required; no performance claim')
        if (summary['backend']['selected_device'] != 'cpu'
                or summary['backend']['torch_intraop_threads'] != 1):
            raise ValueError('actual CPU single-thread backend required')
        prepared = prepare_smoke(payload)
        try:
            if _json(root, 'loss-rows.json') != [r.tolist() for r in prepared.loss_rows]:
                raise ValueError('native and recurrent matched loss-row identities differ')
            lineage = dict(source='recurrent-sac-smoke-'+PROTOCOL_SHA256,
                           dataset=spec['input_identity'], observation_schema='execution-full-v1-normalized21')
            initial, _ = load_checkpoint(root/'recurrent-initial', expected_lineage=lineage)
            final, _ = load_checkpoint(root/'recurrent-final', expected_lineage=lineage)
            if (initial.updates != 0 or final.updates != 4 or final.max_updates != 4
                    or network_hash(initial.networks) != INITIAL_RECURRENT_SHA256
                    or network_hash(initial.networks) != network_hash(prepared.learner.networks)
                    or network_hash(final.networks) == network_hash(initial.networks)
                    or torch.equal(final.log_alpha, initial.log_alpha)):
                raise ValueError('actual initial/final recurrent learning state required')
            verify_polyak(root, initial, final)
            audit = _json(root, 'recurrent-audit.json')
            verify_audit(audit, recurrent=True)
            if canonical(audit['records']) != canonical(_json(root, 'recurrent-progress.json')):
                raise ValueError('progress/audit mismatch')
            for field in ('actor', 'critics'):
                left, right = getattr(initial.networks, field), getattr(final.networks, field)
                if all(torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters(), strict=True)):
                    raise ValueError('actor and critics must each change')
            local_predictions = prediction_parity(final, load_checkpoint(root/'recurrent-final',
                                      expected_lineage=lineage)[0], prepared.arrays)
            remote_predictions = _json(root, 'recurrent-predictions.json')
            if len(remote_predictions) != 9:
                raise ValueError('nine recorded carried/reset predictions required')
            for local, remote in zip(local_predictions, remote_predictions):
                action = np.asarray(remote['action'], dtype=np.float32)
                if (remote['row'] != local['row'] or remote['reset'] is not local['reset']
                        or action.shape != (4,) or not np.isfinite(action).all() or (np.abs(action) > 1).any()
                        or len(remote['state_sha256']) != 64):
                    raise ValueError('recorded recurrent prediction schema mismatch')
            native_audit = _json(root, 'native-audit.json')
            verify_audit(native_audit, recurrent=False)
            before = SAC.load(root/'native-initial.zip', device='cpu')
            after = SAC.load(root/'native-final.zip', device='cpu')
            initial_record, final_record = model_record(before), model_record(after)
            if (initial_record['updates'] != 0 or final_record['updates'] != 4
                    or initial_record['policy_tensor_sha256'] != prepared.control_initial['policy_tensor_sha256']
                    or final_record['policy_tensor_sha256'] == initial_record['policy_tensor_sha256']
                    or final_record['alpha'] == 1
                    or not _same_restored(final_record, native_audit['final_state'])):
                raise ValueError('physical native learned restoration mismatch')
            _raw_entropy(after, root/'native-final.zip')
            for opt in final_record['optimizers'].values():
                if opt['state_count'] != opt['parameter_tensors'] or set(opt['per_parameter_steps'].values()) != {4}:
                    raise ValueError('physical native Adam counters mismatch')
            return dict(saved_artifacts_verified=True, polyak_chain_verified=True,
                        matched_loss_rows=True, local_frozen_prediction_parity=True,
                        remote_recurrent_action_recomputed_exact=all(np.array_equal(
                            np.asarray(a['action'], dtype=np.float32), np.asarray(b['action'], dtype=np.float32))
                            for a, b in zip(local_predictions, remote_predictions)),
                        model_gradient_updates=8, optimizer_steps=24, performance_evaluation=False)
        finally:
            prepared.control.get_env().close()
