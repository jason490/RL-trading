"""Native TQC settings; matches SAC collection/update defaults."""
from rl_stocks.rl.sac.config import parameters as sac_parameters


def parameters(smoke=False):
    return dict(sac_parameters(smoke), top_quantiles_to_drop_per_net=2, use_sde=False,
                policy_kwargs={'n_critics': 2, 'n_quantiles': 25})


def validate_parameters(params):
    policy = params['policy_kwargs']
    for key in ('n_critics', 'n_quantiles'):
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise ValueError('positive integer TQC critic/quantile counts required')
    dropped = params.get('top_quantiles_to_drop_per_net')
    if type(dropped) is not int or not 0 <= dropped < policy['n_quantiles']:
        raise ValueError('TQC truncation must retain at least one quantile per critic')
    if params.get('use_sde') is not False or policy.get('use_sde', False) is not False:
        raise ValueError('native TQC contract requires squashed Gaussian without gSDE')
    return params
