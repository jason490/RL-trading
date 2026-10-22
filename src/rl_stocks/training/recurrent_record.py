"""Shared read-only recurrent model, optimizer and initialization evidence."""
import json


def model_record(model):
    from rl_stocks.training.recurrent_state import model_record as base
    from rl_stocks.reporting.optimizer import optimizer_record
    return dict(base(model),target_kl=model.target_kl,clip_range=float(model.clip_range(1.)),
                max_grad_norm=model.max_grad_norm,policy_class=type(model.policy).__name__,
                optimizer=json.loads(json.dumps(optimizer_record(model))),
                shared_initialization=json.loads(json.dumps(getattr(model,'_research_initialization',None))))


