"""Fixed saved GRU/SAC fitted-period diagnosis; never learns or selects policies."""
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import _evaluate
from rl_stocks.evaluation.training_fit import training_window

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA256 = 'cfa2cac907c5292762b8a2b12dbdd97254d01b71ad988188049d216a76ff4fe2'
SOURCE_FILES = ('configs/experiments/recurrent_offpolicy_fit_diagnostic.json',
                'src/rl_stocks/evaluation/offpolicy_fit.py',
                'scripts/diagnose_offpolicy_fit.py',
                'src/rl_stocks/reporting/participation.py',
                'src/rl_stocks/evaluation/training_fit.py',
                'src/rl_stocks/evaluation/runner.py',
                'src/rl_stocks/rl/registry/prediction.py',
                'src/rl_stocks/reporting/html.py')


def protocol():
    value = json.loads((ROOT/SOURCE_FILES[0]).read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('fixed inference-only fitted-period protocol required')
    return value


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(folder):
    entries = list(Path(folder).rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('regular accepted cell files required')
    return {str(p.relative_to(folder)):checksum(p) for p in entries if p.is_file()}


def accepted_cells():
    from rl_stocks.evaluation.recurrent_comparison_acceptance import receipt_hash
    spec = protocol()
    receipt = json.loads((ROOT/spec['acceptance']).read_text())
    if (receipt_hash(receipt) != spec['acceptance_receipt_sha256']
            or receipt['receipt_sha256'] != spec['acceptance_receipt_sha256']
            or receipt['dataset'] != spec['dataset'] or receipt['stage'] != 'screen'
            or receipt['gate']['passed'] or receipt['retired_test_access']):
        raise ValueError('exact accepted failed-screen receipt required')
    cells = {}
    for arm in spec['arms']:
        source = receipt['source_cells'][arm+'-11']
        folder = Path(source['folder']).resolve()
        proof = receipt['cells']['11'][arm]
        if (not folder.is_relative_to(ROOT.resolve()) or not proof['saved_artifacts_verified']
                or digest(inventory(folder)) != proof['artifact_inventory_sha256']):
            raise ValueError('unchanged complete accepted cell inventory required')
        cells[arm] = dict(folder=folder, source=source, proof=proof)
    return receipt, cells


def evaluate_fixed_window(model, dataset):
    """Keep ordinary evaluation strict; only this diagnostic accepts fixed train rows."""
    view = training_window(dataset, original_panel_only=True)
    return _evaluate(model, view, 'train', SimulationConfig(), 'execution_full', 'unit_magnitude')


@contextmanager
def inference_only():
    from stable_baselines3 import SAC
    from rl_stocks.data.generation.datasets import Normalizer
    from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner
    def forbidden(*args, **kwargs):
        raise RuntimeError('fitting/backward/optimizer steps forbidden in fitted-period diagnosis')
    with ExitStack() as stack:
        for owner, name in [(SAC, 'learn'), (SAC, 'train'), (RecurrentSACLearner, 'update'),
                            (torch.optim.Adam, 'step'), (torch.Tensor, 'backward'),
                            (torch.autograd, 'backward'), (torch.autograd, 'grad'),
                            (Normalizer, 'fit')]:
            stack.enter_context(patch.object(owner, name, forbidden))
        stack.enter_context(torch.no_grad())
        yield


def fingerprint(value):
    """Include every tensor, optimizer moment/group, raw alpha and declared counter."""
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(canonical(['tensor', str(item.dtype), list(item.shape)]))
            result.update(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                result.update(canonical([type(key).__name__, key]))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(canonical([type(item).__name__, len(item)]))
            for element in item:
                visit(element)
        else:
            result.update(canonical(item))
    visit(value)
    return result.hexdigest()


def load_phase(arm, phase, cell):
    from rl_stocks.rl.recurrent_sac.prediction import FrozenRecurrentActor
    from rl_stocks.training.recurrent_sac_checkpoint import load_checkpoint
    from rl_stocks.training.recurrent_comparison_verify import verify_native_checkpoint
    if arm not in protocol()['arms'] or phase not in protocol()['phases']:
        raise ValueError('only fixed GRU/SAC selected/final states allowed')
    pointer = json.loads((cell['folder']/(phase+'.json')).read_text())
    lineage = dict(source=cell['source']['source_hash'], dataset=protocol()['dataset'],
                   observation_schema='execution-full-v1-normalized21')
    folder = cell['folder']/pointer['checkpoint']
    if arm == 'gru_sac':
        learner, manifest = load_checkpoint(folder, expected_lineage=lineage)
        if manifest['updates'] != pointer['updates']:
            raise ValueError('fixed GRU optimizer counter mismatch')
        model = FrozenRecurrentActor(learner.networks, observation_schema=lineage['observation_schema'])
        def state():
            return dict(networks=learner.networks.state_dict(), alpha=learner.log_alpha,
                        optimizers={k:v.state_dict() for k,v in learner.optimizers.items()},
                        updates=learner.updates, attempts=learner.attempts,
                        completed_steps=learner.completed_steps, actor=model._actor.state_dict())
    else:
        model = verify_native_checkpoint(folder, arm=arm, seed=11, steps=pointer['steps'],
                                         lineage=lineage)
        model.policy.set_training_mode(False)
        def state():
            return dict(policy=model.policy.state_dict(), alpha=model.log_ent_coef,
                        actor=model.actor.optimizer.state_dict(), critic=model.critic.optimizer.state_dict(),
                        entropy=model.ent_coef_optimizer.state_dict(), steps=model.num_timesteps,
                        updates=model._n_updates)
    return model, lambda:fingerprint(state()), pointer
