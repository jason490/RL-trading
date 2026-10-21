"""Frozen training/evaluation sizing-alignment hypothesis; no allocation."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.recurrent_state import model_record

PROTOCOL_SHA256='bf5c5ab3a3230367eb763405d1764a77968fd0ccb461f64751ae35adb79cb47e'


def protocol():
    spec=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/recurrent_alignment.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('alignment protocol changed')
    return spec


def modes(arm):
    spec=protocol()
    if arm not in spec['arms']:raise ValueError('fixed alignment arm required')
    item=spec['execution_modes'][arm]
    return execution_contract(item['training'],item['evaluation'])


def overrides(arm,smoke=False):
    modes(arm)
    if type(smoke) is not bool:raise ValueError('explicit boolean smoke required')
    from rl_stocks.training.recurrent_state import overrides as base
    return base('recurrent_masked',smoke)


def validate_record(record,arm,smoke=False,initial=False,selected_steps=None):
    modes(arm)
    from rl_stocks.training.recurrent_state import validate_record as validate
    return validate(record,'recurrent_masked',smoke,initial,selected_steps)
