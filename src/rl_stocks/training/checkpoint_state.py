"""Bind interruption evidence to actual saved weights and evaluation phase."""
import hashlib
import json
import math
from pathlib import Path
from rl_stocks.inference.bundle import checksum
from rl_stocks.evaluation.runner import selection_score


def _atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def write_checkpoint_state(directory,results,model_file,steps,phase,selected_steps,selected_phase,final_post_update=False):
    directory=Path(directory)
    _atomic_json(directory/'checkpoint-validation.json',results)
    def entry(model,report,step,evaluation_phase):
        return dict(steps=step,phase=evaluation_phase,model=model,report=report,
                    model_sha256=checksum(directory/model),report_sha256=checksum(directory/report))
    state=dict(version=1,latest=entry(model_file,'checkpoint-validation.json',steps,phase),
               selected=entry('best.zip','selected-validation.json',selected_steps,selected_phase),
               score=selection_score(results),final_post_update=bool(final_post_update))
    validate_checkpoint_state(directory,state)
    # Written last, before the existing atomic snapshot export. An older export
    # remains intact if writing a later report/model fails.
    _atomic_json(directory/'checkpoint-state.json',state)
    return state


def validate_checkpoint_state(directory,state,read=None):
    read=read or (lambda name:(directory/name).read_bytes())
    if state['version']!=1 or type(state['final_post_update']) is not bool:
        raise ValueError('checkpoint-state version/final flag mismatch')
    if state['final_post_update'] and state['latest']['phase']!='post_update':
        raise ValueError('on-step checkpoint cannot be final post-update')
    for key in ('latest','selected'):
        item=state[key]
        if type(item['steps']) is not int or item['steps']<=0 or item['phase'] not in ('on_step','post_update'):
            raise ValueError('positive checkpoint step and known evaluation phase required')
        for field in ('model','report'):
            name=item[field]
            if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
                raise ValueError('checkpoint file must be a safe basename')
            if hashlib.sha256(read(name)).hexdigest()!=item[field+'_sha256']:
                raise ValueError('checkpoint model/report checksum mismatch')
        results=json.loads(read(item['report']))
        if not results or any(r.get('split')!='validation' for r in results.values()):
            raise ValueError('development validation paths required')
        score=selection_score(results)
        if not math.isfinite(score):raise ValueError('finite checkpoint score required')
        if key=='latest' and not math.isclose(score,state['score'],rel_tol=0,abs_tol=1e-12):
            raise ValueError('checkpoint report/score mismatch')
    if state['selected']['steps']>state['latest']['steps']:
        raise ValueError('selected checkpoint cannot come from future steps')
    return state


def read_checkpoint_state(directory):
    directory=Path(directory)
    return validate_checkpoint_state(directory,json.loads((directory/'checkpoint-state.json').read_text()))
