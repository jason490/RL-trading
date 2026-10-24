"""Validate actual distributional heads and their saved inference declaration."""
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import TQCPolicy

from rl_stocks.core.contracts import canonical


def model_spec(model):
    if not isinstance(model, TQC) or not isinstance(model.policy, TQCPolicy):
        raise ValueError('actual TQC algorithm and policy required')
    critic, target = model.critic, model.critic_target
    n_critics, n_quantiles = critic.n_critics, critic.n_quantiles
    dropped = model.top_quantiles_to_drop_per_net
    if (type(n_critics) is not int or n_critics < 1
            or type(n_quantiles) is not int or n_quantiles < 1
            or type(dropped) is not int or not 0 <= dropped < n_quantiles):
        raise ValueError('invalid loaded TQC quantile geometry')
    for network in (critic, target):
        if (type(network.n_critics) is not int or type(network.n_quantiles) is not int
                or type(network.quantiles_total) is not int
                or network.n_critics != n_critics or network.n_quantiles != n_quantiles
                or network.quantiles_total != n_critics * n_quantiles
                or len(network.q_networks) != n_critics
                or any(head[-1].out_features != n_quantiles for head in network.q_networks)):
            raise ValueError('actual TQC critic/target head geometry mismatch')
    return dict(version=1, kind='distributional_quantile_critics', n_critics=n_critics,
                n_quantiles=n_quantiles, total_quantiles=n_critics * n_quantiles,
                top_quantiles_to_drop_per_net=dropped,
                retained_target_quantiles=n_critics * (n_quantiles - dropped))


def validate_loaded(model, declared, algorithm):
    actual_policy = isinstance(model.policy, TQCPolicy)
    if algorithm != 'tqc':
        if actual_policy or isinstance(model, TQC) or declared is not None:
            raise ValueError('algorithm and saved critic geometry mismatch')
        return None
    actual = model_spec(model)
    if not isinstance(declared, dict) or canonical(actual) != canonical(declared):
        raise ValueError('explicit saved TQC critic geometry mismatch')
    return actual
