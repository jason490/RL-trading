"""Declared PPO action-distribution initializations for a bounded factorial study."""

ARMS = ('standard', 'low_noise', 'wide', 'wide_low_noise')


def initialization_spec(arm='standard', algorithm='ppo'):
    if arm not in ARMS or algorithm != 'ppo' and arm != 'standard':
        raise ValueError('unsupported policy initialization')
    if arm == 'standard':
        return {'arm': arm, 'applied': False}
    return {'arm': arm, 'applied': True,
            'protective_mean_bias': .75 if arm in ('wide', 'wide_low_noise') else 0.,
            'log_std': -1. if arm in ('low_noise', 'wide_low_noise') else 0.,
            'exposure_mean_bias': 0., 'action_weights': 'unchanged seeded native weights'}


def initialize_policy(model, arm='standard', algorithm='ppo'):
    spec = initialization_spec(arm, algorithm)
    if not spec['applied']:
        return spec  # Preserve the native model and RNG stream exactly.
    import torch
    if model.policy.action_net.bias.shape != (4,) or model.policy.log_std.shape != (4,):
        raise ValueError('four-component diagonal Gaussian policy required')
    with torch.no_grad():
        model.policy.action_net.bias[0] = spec['exposure_mean_bias']
        model.policy.action_net.bias[1:] = spec['protective_mean_bias']
        model.policy.log_std.fill_(spec['log_std'])
    return spec
