"""Observe native policy updates; never initiate learning or an optimizer step."""

import math

import torch

from rl_stocks.evaluation.offpolicy_fit import fingerprint


def actor_parameters(policy):
    return {
        k: v
        for k, v in policy.named_parameters()
        if k == "log_std" or k.startswith(("mlp_extractor.policy_net.", "action_net."))
    }


def clone_actor(policy):
    return {k: v.detach().cpu().clone() for k, v in actor_parameters(policy).items()}


class UpdateAudit:
    def __init__(self, model, algorithm, *, expected_updates, expected_adam_steps):
        if (
            algorithm not in ("ppo", "trpo")
            or type(expected_updates) is not int
            or expected_updates < 1
            or type(expected_adam_steps) is not int
            or expected_adam_steps < 1
            or "train" in model.__dict__
            or "hessian_vector_product" in model.__dict__
        ):
            raise ValueError("bounded unwrapped native update audit required")
        self.model = model
        self.algorithm = algorithm
        self.limit = expected_updates
        self.adam_limit = expected_adam_steps
        self.adam_steps = 0
        self.hvp_calls = 0
        self.records = []
        self.actor_states = []
        self.active = set()
        self.handles = []
        self.named = dict(model.policy.named_parameters())
        self.actor = actor_parameters(model.policy)
        self.critic = {k: v for k, v in self.named.items() if k not in self.actor}
        if not self.actor or not self.critic:
            raise ValueError("separate actor/value parameters required")
        self.original_train = model.train
        self.handles = [
            model.policy.optimizer.register_step_pre_hook(self.before_step),
            model.policy.optimizer.register_step_post_hook(self.after_step),
        ]
        if algorithm == "trpo":
            self.original_hvp = model.hessian_vector_product
            model.hessian_vector_product = self.hvp
        model.train = self.train

    def before_step(self, optimizer, args, kwargs):
        self._actor_before_step = fingerprint(self.actor) if self.algorithm == "trpo" else None
        if self.adam_steps >= self.adam_limit:
            raise ValueError("actual Adam allowance exceeded")
        for name, p in self.named.items():
            required = self.algorithm == "ppo" or name in self.critic
            if required and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise ValueError("finite required policy/value gradients missing")
            if not required and p.grad is not None:
                raise ValueError("TRPO critic optimizer must exclude actor gradients")
            if p.grad is not None and torch.count_nonzero(p.grad):
                self.active.add("actor" if name in self.actor else "critic")

    def after_step(self, optimizer, args, kwargs):
        self.adam_steps += 1
        if self.algorithm == "trpo" and fingerprint(self.actor) != self._actor_before_step:
            raise ValueError("critic Adam changed constrained actor parameters")
        if any(not torch.isfinite(p).all() for p in self.named.values()):
            raise ValueError("nonfinite policy state after Adam")

    def hvp(self, *args, **kwargs):
        if self.hvp_calls >= self.limit * (self.model.cg_max_steps + 2):
            raise ValueError("bounded actual Hessian-vector products required")
        value = self.original_hvp(*args, **kwargs)
        self.hvp_calls += 1
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite Hessian-vector product")
        return value

    def train(self, *args, **kwargs):
        if len(self.records) >= self.limit:
            raise ValueError("nominal policy-update allowance exceeded")
        before = clone_actor(self.model.policy)
        obs = torch.as_tensor(self.model.rollout_buffer.observations.copy()).reshape(-1, 21).clone()
        steps_before = self.adam_steps
        hvp_before = self.hvp_calls
        result = self.original_train(*args, **kwargs)
        after = clone_actor(self.model.policy)
        changed = fingerprint(before) != fingerprint(after)
        row = {
            "transition": self.model.num_timesteps,
            "adam_steps": self.adam_steps,
            "adam_steps_in_update": self.adam_steps - steps_before,
            "hvp_calls_in_update": self.hvp_calls - hvp_before,
            "actor_changed": changed,
            "actor_before_sha256": fingerprint(before),
            "actor_after_sha256": fingerprint(after),
        }
        if self.algorithm == "trpo":
            logs = self.model.logger.name_to_value
            success = float(logs["train/is_line_search_success"])
            kl = float(logs["train/kl_divergence_loss"])
            objective = float(logs["train/policy_objective"])
            if (
                success not in (0.0, 1.0)
                or not all(math.isfinite(x) for x in (kl, objective))
                or kl < 0
                or success == 1
                and (not changed or kl >= self.model.target_kl)
                or success == 0
                and (changed or kl != 0)
            ):
                raise ValueError("actual line-search acceptance/rollback inconsistent")
            row.update(line_search_accepted=bool(success), kl=kl, policy_objective=objective)
        self.records.append(row)
        self.actor_states.append({"before": before, "after": after, "observations": obs})
        return result

    def verify(self, transitions):
        if (
            len(self.records) != self.limit
            or self.adam_steps != self.adam_limit
            or self.model.num_timesteps != transitions
        ):
            raise ValueError("actual transition/update/Adam counters mismatch")
        if [r["transition"] for r in self.records] != [
            self.model.n_steps * i for i in range(1, self.limit + 1)
        ]:
            raise ValueError("post-update transition cadence mismatch")
        if self.algorithm == "trpo" and any(
            not 2 <= r["hvp_calls_in_update"] <= self.model.cg_max_steps + 2 for r in self.records
        ):
            raise ValueError("actual Hessian-vector count outside native bound")
        expected_n = self.limit * (self.model.n_epochs if self.algorithm == "ppo" else 1)
        if self.model._n_updates != expected_n:
            raise ValueError("native update counter mismatch")
        required = self.named if self.algorithm == "ppo" else self.critic
        optimizer = self.model.policy.optimizer
        if set(optimizer.state) != set(required.values()):
            raise ValueError("exact actual Adam parameter ownership required")
        for p, state in optimizer.state.items():
            if (
                set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or float(state["step"].item()) != self.adam_steps
                or state["exp_avg"].shape != p.shape
                or state["exp_avg_sq"].shape != p.shape
                or not torch.isfinite(state["exp_avg"]).all()
                or not torch.isfinite(state["exp_avg_sq"]).all()
                or state["exp_avg"].dtype != p.dtype
                or state["exp_avg_sq"].dtype != p.dtype
                or (state["exp_avg_sq"] < 0).any()
            ):
                raise ValueError("finite complete actual Adam state required")
        roles = {"actor", "critic"} if self.algorithm == "ppo" else {"critic"}
        if self.active != roles:
            raise ValueError("actual nonzero required gradients not observed")
        if self.algorithm == "trpo" and not any(r["line_search_accepted"] for r in self.records):
            raise ValueError("no accepted actor line search in smoke/profile")
        return {
            "nominal_policy_updates": self.limit,
            "native_n_updates": self.model._n_updates,
            "actual_adam_steps": self.adam_steps,
            "actual_hvp_calls": self.hvp_calls,
            "accepted_actor_line_searches": sum(
                r.get("line_search_accepted", False) for r in self.records
            ),
            "records": self.records,
        }

    def close(self):
        self.model.__dict__.pop("train", None)
        if self.algorithm == "trpo":
            self.model.__dict__.pop("hessian_vector_product", None)
        for handle in self.handles:
            handle.remove()
        self.handles = []
