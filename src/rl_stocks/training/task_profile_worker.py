"""Research-geometry timing in owned Colab; no development scoring or selection."""
from pathlib import Path
import re
import time

import torch

from rl_stocks.core.contracts import digest
from rl_stocks.core.provenance import repository_state, source_identity
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.evaluation.task_conditioning import evaluate_task_training
from rl_stocks.rl.sac.task_factory import create_task_sac
from rl_stocks.rl.sac.task_policy import check_role
from rl_stocks.training.checkpoint_export import export_artifacts
from rl_stocks.training.recurrent_loop import _atomic_json
from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.task_conditioning import PROTOCOL_SHA256, protocol, validate_type_manifest
from rl_stocks.training.task_environment import build_task_environment
from rl_stocks.training.task_optimizer import TaskOptimizerAudit


PROFILE_STEPS = 4096
PROFILE_UPDATES = 3096


def execute_profile_colab(dataset, type_manifest, output, *, task_role):
    require_colab_cpu()  # Before any local path/model/optimizer/RNG mutation.
    role = check_role(task_role)
    spec = protocol()
    if digest(dataset) != spec['dataset'] or digest(dataset['normalizer']) != spec['normalizer_sha256']:
        raise ValueError('fixed research dataset and preprocessing required')
    evidence_sha = validate_type_manifest(dataset, type_manifest)
    output = Path(output)
    git = repository_state(Path.cwd())
    if git['dirty'] != '' or not re.fullmatch(r'[a-f0-9]{40}', git['revision'] or ''):
        raise ValueError('clean committed profile source required')
    if output.exists() or output.parent.name != 'artifacts':
        raise ValueError('fresh owned profile artifacts required')
    output.mkdir(parents=True)
    env = audit = model = None
    started = time.monotonic()
    metadata = dict(task_role=role, protocol_sha256=PROTOCOL_SHA256,
        dataset=spec['dataset'], type_manifest_sha256=evidence_sha, repository=git,
        source=source_identity(), profile_only=True, seed=11,
        development_evaluated=False, test_evaluated=False, performance_promotion=False)
    try:
        with preserved_rng():
            torch.set_num_threads(1)
            env = build_task_environment(dataset, type_manifest, task_role=role, smoke=False)
            model = create_task_sac(env, task_role=role, seed=11, smoke=False)
            if model.batch_size != 256 or model.learning_starts != 1000:
                raise ValueError('research batch/warmup required for profiling')
            initial = fingerprint(model.policy.state_dict())
            model.save(output / 'initial.zip')
            _atomic_json(output / 'input.json', metadata)
            export_artifacts(output.parent)
            audit = TaskOptimizerAudit(model, maximum=PROFILE_UPDATES)
            learning_started = time.monotonic()
            model.learn(total_timesteps=PROFILE_STEPS)
            learning_seconds = time.monotonic() - learning_started
            audit.verify(model, PROFILE_STEPS, 1000)
            model.save(output / 'profile-final.zip')
            export_artifacts(output.parent)
            if initial == fingerprint(model.policy.state_dict()):
                raise ValueError('profile did not change learned parameters')
            before = fingerprint((model.policy.state_dict(),
                                  {k: v.state_dict() for k, v in audit.optimizers.items()},
                                  model.log_ent_coef))
            evaluation_started = time.monotonic()
            with preserved_rng(), torch.no_grad():
                results = evaluate_task_training(model, dataset, type_manifest)
            evaluation_seconds = time.monotonic() - evaluation_started
            if before != fingerprint((model.policy.state_dict(),
                                      {k: v.state_dict() for k, v in audit.optimizers.items()},
                                      model.log_ent_coef)):
                raise ValueError('profile inference mutated learner')
            if len(results) != 26:
                raise ValueError('complete fixed training-window panel required')
            _atomic_json(output / 'profile.json', {**metadata, 'status': 'complete',
                'actual_transitions': model.num_timesteps, 'optimizer_steps': dict(audit.counts),
                'learning_seconds': learning_seconds, 'evaluation_seconds': evaluation_seconds,
                'inclusive_elapsed_seconds': time.monotonic() - started,
                'sampling': env.unwrapped.sampling_statistics(),
                'active_gradients': audit.active,
                'holding_path_sha256': digest({k: v['baselines']['buy_hold'] for k, v in results.items()}),
                'performance_scores_exported': False})
    except BaseException as error:
        _atomic_json(output / 'failure.json', {**metadata, 'error_type': type(error).__name__})
        raise
    finally:
        try:
            if audit is not None:
                audit.close()
            if env is not None:
                env.close()
        finally:
            export_artifacts(output.parent)
