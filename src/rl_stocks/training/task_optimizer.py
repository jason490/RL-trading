"""Observe bounded actual SAC Adam steps; never perform an optimizer update."""
import torch


class TaskOptimizerAudit:
    def __init__(self, model, *, maximum):
        if type(maximum) is not int or maximum < 1:
            raise ValueError('positive bounded update allowance required')
        self.optimizers = {'actor': model.actor.optimizer, 'critic': model.critic.optimizer,
                           'entropy': model.ent_coef_optimizer}
        self.maximum = maximum
        self.counts = dict.fromkeys(self.optimizers, 0)
        self.active = {}
        self.handles = []
        for role, optimizer in self.optimizers.items():
            self.handles.append(optimizer.register_step_pre_hook(
                lambda opt, args, kwargs, role=role: self.before(role, opt)))
            self.handles.append(optimizer.register_step_post_hook(
                lambda opt, args, kwargs, role=role: self.after(role, opt)))
        self.named = {'actor': dict(model.actor.named_parameters()),
                      'critic': dict(model.critic.named_parameters()),
                      'entropy': {'log_ent_coef': model.log_ent_coef}}

    def before(self, role, optimizer):
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        if self.counts[role] >= self.maximum or any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('bounded finite actual task gradients required')
        for name, parameter in self.named[role].items():
            if torch.count_nonzero(parameter.grad):
                self.active[role + '/' + name] = self.active.get(role + '/' + name, 0) + 1

    def after(self, role, optimizer):
        self.counts[role] += 1
        if any(not torch.isfinite(p).all() for group in optimizer.param_groups
               for p in group['params']):
            raise ValueError('nonfinite task parameter after Adam')

    def verify(self, model, transitions, warmup):
        updates = max(0, transitions - warmup)
        if (model.num_timesteps != transitions or model._n_updates != updates
                or self.counts != dict.fromkeys(self.optimizers, updates)):
            raise ValueError('actual task optimizer counters mismatch')
        for role, optimizer in self.optimizers.items():
            owned = [p for group in optimizer.param_groups for p in group['params']]
            if (len(owned) != len(self.named[role])
                    or {id(p) for p in owned} != {id(p) for p in self.named[role].values()}):
                raise ValueError('exact task optimizer ownership required')
            if len(optimizer.state) != (len(owned) if updates else 0):
                raise ValueError('complete task Adam parameter state required')
            for parameter in self.named[role].values():
                state = optimizer.state.get(parameter, {})
                if updates and (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                                or state['step'].item() != updates
                                or bool((state['exp_avg_sq'] < 0).any())
                                or any(state[k].shape != parameter.shape
                                       or not torch.isfinite(state[k]).all()
                                       for k in ('exp_avg', 'exp_avg_sq'))):
                    raise ValueError('actual task Adam moments mismatch')
                if not updates and state:
                    raise ValueError('initial Adam state must be empty')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
