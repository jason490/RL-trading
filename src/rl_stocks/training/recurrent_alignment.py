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


def validate_payload(data):
    from rl_stocks.training.recurrent_state import validate_payload as validate
    return validate(data)


def validate_execution_counts(counts,arm,steps):
    expected=modes(arm)['training_exposure_mode']
    fields={'transitions','requested_nonzero','processed_nonunit','sizing_changed','protective_changed'}
    if (set(counts)!={'mode',*fields} or counts['mode']!=expected
        or any(type(counts[k]) is not int or not 0<=counts[k]<=steps for k in fields)
        or counts['transitions']!=steps or counts['protective_changed']!=0):
        raise ValueError('actual training execution counters mismatch')
    if expected=='unit_magnitude':
        if counts['processed_nonunit'] or counts['sizing_changed']<=0:raise ValueError('unit action mapping not applied')
    elif counts['sizing_changed'] or counts['processed_nonunit']<=0:
        raise ValueError('adaptive action mapping mismatch')
    return counts


def validate_execution_report(results,arm):
    import numpy as np
    from rl_stocks.risk.exposure import filter_exposure
    mode=modes(arm)['evaluation_exposure_mode']
    if not results:raise ValueError('nonempty execution report required')
    for result in results.values():
        if result['split']!='validation' or not result['rows']:raise ValueError('validation execution rows required')
        for row in result['rows']:
            if mode=='unit_magnitude':
                if row.get('exposure_mode')!=mode or 'policy_action' not in row:
                    raise ValueError('unit evaluation action trace missing')
                if not np.array_equal(filter_exposure(row['policy_action'],mode),row['raw_action']):
                    raise ValueError('unit evaluation exposure/protection mapping mismatch')
            elif row.get('exposure_mode','unrestricted')!='unrestricted':
                raise ValueError('adaptive evaluation execution mismatch')
    return results
