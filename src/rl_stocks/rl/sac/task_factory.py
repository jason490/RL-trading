"""Opt-in factory for the fixed task-conditioning SAC comparison."""
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.rl.sac.task_policy import TaskSACPolicy, check_role


def create_task_sac(env, *, task_role, seed=11, device='cpu', smoke=False):
    from stable_baselines3 import SAC
    role = check_role(task_role)
    if getattr(env, 'task_role', None) != role:
        raise ValueError('environment task role mismatch')
    if env.observation_space.shape != (22,):
        raise ValueError('task environment schema mismatch')
    parameters = resolved_parameters('sac', smoke)
    # Smokes exercise actual research geometry; only replay batch/warmup shrink.
    parameters['policy_kwargs'] = {'net_arch': [64, 64], 'task_role': role}
    return SAC(TaskSACPolicy, env, seed=seed, device=device, verbose=0, **parameters)

