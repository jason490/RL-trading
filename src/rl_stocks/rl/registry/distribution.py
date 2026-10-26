"""Saved action-distribution identity, independent of execution transforms."""


def distribution_spec(cash_enabled=None):
    if cash_enabled is None:
        return {'version':1,'kind':'diagonal_gaussian','components':4}
    if type(cash_enabled) is not bool:
        raise ValueError('explicit boolean cash availability required')
    return {'version':1,'kind':'categorical_direction_gaussian_protection',
            'directions':[-1,0,1] if cash_enabled else [-1,1],
            'cash_enabled':cash_enabled,'gaussian_components':3}


def model_distribution(model):
    from rl_stocks.rl.recurrent_ppo.hybrid import HybridRecurrentPolicy,DirectionProtectionDistribution
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from stable_baselines3 import SAC,TD3
    from sb3_contrib import TQC,CrossQ
    policy=model.policy
    if isinstance(policy,HybridRecurrentPolicy):
        if (not isinstance(policy.action_dist,DirectionProtectionDistribution)
                or policy.action_dist.cash_enabled is not policy.cash_enabled
                or policy.action_net.out_features!=6 or tuple(policy.log_std.shape)!=(3,)):
            raise ValueError('hybrid model distribution mismatch')
        return distribution_spec(policy.cash_enabled)
    from rl_stocks.rl.recurrent_ppo.squashed import SquashedRecurrentPolicy
    from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
    if isinstance(policy,SquashedRecurrentPolicy):
        if (type(policy.action_dist) is not SquashedDiagGaussianDistribution or policy.use_sde is not False
                or policy.squash_output is not True or policy.action_net.out_features!=4
                or tuple(policy.log_std.shape)!=(4,) or policy.action_dist.epsilon!=1e-6):
            raise ValueError('squashed recurrent distribution mismatch')
        return {'version':1,'kind':'tanh_squashed_diagonal_gaussian','components':4,'jacobian_epsilon':1e-6}
    if isinstance(model,TQC):
        from rl_stocks.rl.tqc.geometry import model_spec
        model_spec(model)
        actor=policy.actor
        if (model.use_sde is not False or type(actor.action_dist) is not SquashedDiagGaussianDistribution
                or actor.mu.out_features!=4 or actor.log_std.out_features!=4
                or model.action_space.shape!=(4,)):
            raise ValueError('TQC four-component squashed Gaussian contract mismatch')
        return {'version':1,'kind':'squashed_gaussian','components':4}
    if isinstance(model,CrossQ):
        from rl_stocks.rl.crossq.geometry import model_spec
        model_spec(model)
        actor=policy.actor
        if (type(actor.action_dist) is not SquashedDiagGaussianDistribution
                or actor.mu.out_features!=4 or actor.log_std.out_features!=4):
            raise ValueError('CrossQ four-component squashed Gaussian required')
        return {'version':1,'kind':'squashed_gaussian','components':4}
    # SAC uses a squashed Gaussian; TD3 is deterministic with external exploration.
    if isinstance(model,SAC):return {'version':1,'kind':'squashed_gaussian','components':4}
    if isinstance(model,TD3):return {'version':1,'kind':'deterministic','components':4}
    if type(getattr(policy,'action_dist',None)) is not DiagGaussianDistribution:
        raise ValueError('unsupported model action distribution')
    return distribution_spec()


def validate_model_distribution(model,declared,algorithm):
    actual=model_distribution(model)
    hybrid=actual['kind']=='categorical_direction_gaussian_protection'
    if hybrid != (algorithm=='hybrid_recurrent_ppo'):
        raise ValueError('algorithm and action distribution mismatch')
    bounded=actual['kind']=='tanh_squashed_diagonal_gaussian'
    if bounded != (algorithm=='squashed_recurrent_ppo'):
        raise ValueError('algorithm and squashed distribution mismatch')
    # Legacy bundles predate this field; only established native policies qualify.
    if declared is None:
        if hybrid:raise ValueError('hybrid distribution metadata required')
        if bounded:raise ValueError('squashed distribution metadata required')
        return actual
    # bool is an int subclass, so dict equality alone would accept version=True.
    from rl_stocks.core.contracts import canonical
    if not isinstance(declared,dict) or type(declared.get('version')) is not int or canonical(declared)!=canonical(actual):
        raise ValueError('saved action distribution contract mismatch')
    if hybrid and type(declared.get('cash_enabled')) is not bool:
        raise ValueError('explicit saved cash availability required')
    return actual
