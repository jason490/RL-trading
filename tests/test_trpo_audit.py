from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from rl_stocks.training.trpo_audit import UpdateAudit, actor_parameters


class Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class FakeOptimizer:
    def __init__(self):
        self.state = {}

    def register_step_pre_hook(self, fn):
        return Handle()

    def register_step_post_hook(self, fn):
        return Handle()


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(4))
        self.mlp_extractor = nn.Module()
        self.mlp_extractor.policy_net = nn.Linear(21, 4)
        self.action_net = nn.Linear(4, 4)
        self.value_net = nn.Linear(21, 1)
        self.optimizer = FakeOptimizer()


class Model:
    def __init__(self, algorithm):
        self.policy = Policy()
        self.algorithm = algorithm
        self.num_timesteps = 0
        self._n_updates = 0
        self.n_steps = 128
        self.n_epochs = 10
        self.cg_max_steps = 15
        self.target_kl = 0.01
        self.rollout_buffer = SimpleNamespace(observations=np.zeros((128, 1, 21), dtype=np.float32))
        self.logger = SimpleNamespace(name_to_value={})
        self.success = True
        self.corrupt_rollback = False

    def hessian_vector_product(self, *args, **kwargs):
        return torch.ones(2)

    def train(self):
        audit = self.audit
        if self.algorithm == "trpo":
            self.hessian_vector_product(None)
            self.hessian_vector_product(None)
            if self.success or self.corrupt_rollback:
                with torch.no_grad():
                    self.policy.log_std.add_(0.001)
        for _ in range(2):
            actor = actor_parameters(self.policy)
            for name, p in self.policy.named_parameters():
                p.grad = None if self.algorithm == "trpo" and name in actor else torch.ones_like(p)
            # Manually populate fixture state and invoke observer hooks. No
            # optimizer.step, backward, loss or fitting is executed locally.
            audit.before_step(self.policy.optimizer, (), {})
            for name, p in self.policy.named_parameters():
                if self.algorithm == "ppo" or name not in actor:
                    self.policy.optimizer.state[p] = {
                        "step": torch.tensor(float(audit.adam_steps + 1)),
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                    }
            audit.after_step(self.policy.optimizer, (), {})
        self.num_timesteps += 128
        self._n_updates += 10 if self.algorithm == "ppo" else 1
        self.logger.name_to_value = {
            "train/is_line_search_success": int(self.success),
            "train/kl_divergence_loss": 0.005 if self.success else 0.0,
            "train/policy_objective": 0.1,
        }


def attach(algorithm="trpo"):
    model = Model(algorithm)
    audit = UpdateAudit(model, algorithm, expected_updates=2, expected_adam_steps=4)
    model.audit = audit
    return model, audit


@pytest.mark.parametrize("algorithm", ["ppo", "trpo"])
def test_observer_counts_real_hooks_and_native_counters_without_learning(algorithm):
    model, audit = attach(algorithm)
    model.train()
    model.train()
    proof = audit.verify(256)
    assert proof["actual_adam_steps"] == 4 and proof["nominal_policy_updates"] == 2
    assert proof["actual_hvp_calls"] == (4 if algorithm == "trpo" else 0)
    assert len(audit.actor_states) == 2 and audit.actor_states[0]["observations"].shape == (128, 21)
    audit.close()
    assert "train" not in model.__dict__ and "hessian_vector_product" not in model.__dict__
    assert not audit.handles


def test_failed_search_rolls_back_and_mixed_search_counts_remain_distinct():
    model, audit = attach()
    model.success = False
    model.train()
    model.success = True
    model.train()
    assert audit.verify(256)["accepted_actor_line_searches"] == 1
    audit.close()
    model, audit = attach()
    model.success = False
    model.corrupt_rollback = True
    with pytest.raises(ValueError, match="rollback"):
        model.train()
    audit.close()


def test_missing_gradients_actor_gradients_nonfinite_hvp_and_counter_corruption_rejected():
    model, audit = attach()
    with pytest.raises(ValueError, match="gradients"):
        audit.before_step(model.policy.optimizer, (), {})
    for p in model.policy.parameters():
        p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="actor gradients"):
        audit.before_step(model.policy.optimizer, (), {})
    audit.close()
    model = Model("trpo")
    model.hessian_vector_product = lambda *a, **k: torch.tensor([float("nan")])
    # Instance overrides deliberately reject the audit rather than serialize callbacks.
    with pytest.raises(ValueError, match="unwrapped"):
        UpdateAudit(model, "trpo", expected_updates=2, expected_adam_steps=4)
    model, audit = attach()
    model.train()
    model.train()
    next(iter(model.policy.optimizer.state.values()))["step"] = torch.tensor(4.5)
    with pytest.raises(ValueError, match="Adam state"):
        audit.verify(256)
    audit.close()


def test_nonfinite_hvp_and_critic_actor_mutation_fail_closed():
    model, audit = attach()
    audit.original_hvp = lambda *a, **k: torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="Hessian"):
        audit.hvp(None)
    audit.close()
    model, audit = attach()
    for name, p in model.policy.named_parameters():
        p.grad = None if name in audit.actor else torch.ones_like(p)
    audit.before_step(model.policy.optimizer, (), {})
    with torch.no_grad():
        model.policy.log_std.add_(0.001)
    with pytest.raises(ValueError, match="critic Adam changed"):
        audit.after_step(model.policy.optimizer, (), {})
    audit.close()


def test_reported_accepted_kl_cannot_exceed_fixed_constraint():
    model, audit = attach()
    original = audit.original_train

    def invalid_kl():
        original()
        model.logger.name_to_value["train/kl_divergence_loss"] = 0.02

    audit.original_train = invalid_kl
    with pytest.raises(ValueError, match="acceptance"):
        model.train()
    audit.close()
