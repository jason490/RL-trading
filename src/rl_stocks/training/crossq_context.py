"""Registered two-arm request/data dispatch before environment construction."""

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.training.crossq_spec import PROTOCOL_SHA256 as PROTOCOL_SHA256, protocol, parameters
from rl_stocks.training.crossq_assets import validate_assets


def validate_dispatch(algorithm, protocol_id):
    if algorithm == 'crossq' and protocol_id != PROTOCOL_SHA256:
        raise ValueError('CrossQ requires explicit registered study dispatch')
    return protocol_id == PROTOCOL_SHA256


def validate_payload(data):
    spec = protocol()
    if (digest(data) != spec['dataset'] or data['schema'] != spec['feature_schema']
        or data['names'] != list(NAMES) or digest(data['normalizer']) != spec['normalizer_sha256']
        or data['manifest']['included_splits'] != ['train', 'validation']):
        raise ValueError('fixed original SAC/CrossQ data and preprocessing required')
    return data


def validate_request(dataset_id, algorithm, overrides, *, seed, smoke, steps, every,
                     initialization, policy_variant, observation_profile, start_sampling,
                     reward_kind, reward_scale, training_exposure_mode, evaluation_exposure_mode,
                     final_test, compact_checkpoints, milestones, device):
    spec = protocol()
    if (algorithm not in spec['arms'] or dataset_id != spec['dataset']
        or type(seed) is not int or seed not in spec['confirmation_seeds']
        or type(smoke) is not bool or smoke and seed != spec['screen_seed']):
        raise ValueError('fixed SAC/CrossQ algorithm/data/seed/phase required')
    if canonical(overrides) != canonical(parameters(algorithm, smoke)):
        raise ValueError('fixed SAC/CrossQ package settings required')
    actual = dict(initialization=initialization, policy_variant=policy_variant,
                  observation_profile=observation_profile, start_sampling=start_sampling,
                  reward_kind=reward_kind, reward_scale=reward_scale,
                  training_exposure_mode=training_exposure_mode,
                  evaluation_exposure_mode=evaluation_exposure_mode, final_test=final_test,
                  compact_checkpoints=compact_checkpoints, device=device)
    expected = dict(initialization='standard', policy_variant='native',
                    observation_profile=spec['observation_profile'], start_sampling=spec['start_sampling'],
                    reward_kind=spec['reward']['kind'], reward_scale=spec['reward']['scale'],
                    training_exposure_mode=spec['action_execution']['training_exposure_mode'],
                    evaluation_exposure_mode=spec['action_execution']['evaluation_exposure_mode'],
                    final_test=False, compact_checkpoints=True, device='cpu')
    if canonical(actual) != canonical(expected):
        raise ValueError('fixed SAC/CrossQ reward/execution/checkpoint/CPU contract required')
    wanted = spec['smoke_steps' if smoke else 'steps_per_cell']
    cadence = spec['smoke_evaluation_every' if smoke else 'evaluation_every']
    if (type(steps) is not int or steps != wanted or type(every) is not int or every != cadence
        or milestones is not None and canonical(milestones) != canonical([wanted])):
        raise ValueError('fixed SAC/CrossQ transition/cadence/milestone contract required')
    validate_assets()
    return algorithm
