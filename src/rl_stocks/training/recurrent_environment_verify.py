"""Independent physical environment-smoke verification; no gradient operations."""
import csv
import json
from pathlib import Path



from rl_stocks.core.contracts import canonical, digest
from rl_stocks.evaluation.recorded_economics import verify_report
from rl_stocks.training.curves import curve_record
from rl_stocks.training.recurrent_environment_spec import protocol, validate_view, PROTOCOL_SHA256
from rl_stocks.training import recurrent_environment_worker as worker
from rl_stocks.training.recurrent_sac_checkpoint import load_checkpoint
from rl_stocks.training.recurrent_sac_smoke_worker import INITIAL_RECURRENT_SHA256, network_hash
from rl_stocks.training.sac_update_worker import preserved_rng

FILES = {'dataset.json', 'run.json', 'learning-curve.jsonl', 'learning-curve.csv', 'learning-curve.html',
         'selected.json', 'final.json', 'selected-validation.json', 'final-validation.json',
         'selected-validation.html', 'final-validation.html', 'training-execution.json', 'training-sampling.json',
         'validation-64.json', 'validation-128.json'}
FILES |= {f'{folder}/{name}' for folder in ('initial', 'checkpoint-64', 'checkpoint-128')
          for name in ('state.pt', 'manifest.json')}


def _json(root, name):
    return json.loads((root/name).read_text())



def verify_directory(folder):
    root = Path(folder)
    spec = protocol()
    with preserved_rng():
        dataset = validate_view(_json(root,'dataset.json'))
        if digest(dataset) != worker.INPUT_ID:
            raise ValueError('accepted immutable environment view required')
        run = _json(root,'run.json')
        lineage = dict(source='recurrent-environment-smoke-'+PROTOCOL_SHA256, dataset=worker.INPUT_ID,
                       observation_schema='execution-full-v1-normalized21')
        expected = dict(status='complete', config=spec['loop'], lineage=lineage,
                        exposure_mode=spec['exposure_mode'], exact_resume_supported=False,
                        actual_transitions=128, actual_updates=4)
        if canonical({k: run.get(k) for k in expected}) != canonical(expected):
            raise ValueError('fixed completed environment/update counts required')
        models = {}
        for folder, updates in [('initial',0), ('checkpoint-64',0), ('checkpoint-128',4)]:
            model, _ = load_checkpoint(root/folder, expected_lineage=lineage)
            if model.updates != updates or model.max_updates != 4:
                raise ValueError('actual native checkpoint update counts mismatch')
            models[folder] = model
        if (network_hash(models['initial'].networks) != INITIAL_RECURRENT_SHA256
                or network_hash(models['checkpoint-64'].networks) != INITIAL_RECURRENT_SHA256
                or network_hash(models['checkpoint-128'].networks) == INITIAL_RECURRENT_SHA256
                or models['checkpoint-128'].log_alpha.item() == 0):
            raise ValueError('empty initial/warmup and learned final state required')
        execution = _json(root,'training-execution.json')
        sampling = _json(root,'training-sampling.json')
        if (execution['mode'] != 'unrestricted' or type(execution['transitions']) is not int
                or execution['transitions'] != 128 or sampling['mode'] != 'stock_etf_cycle_windows'
                or type(sampling['transitions']) is not int or sampling['transitions'] != 128):
            raise ValueError('actual collection/execution/sampling counts required')
        history = [json.loads(line) for line in (root/'learning-curve.jsonl').read_text().splitlines()]
        if len(history) != 2 or [p['steps'] for p in history] != [64,128]:
            raise ValueError('two fixed checkpoint curve points required')
        reports = {}
        previous = []
        for point, updates in zip(history, (0,4)):
            steps = point['steps']
            results = _json(root,f'validation-{steps}.json')
            verify_report(results,dataset,models[f'checkpoint-{steps}'].networks)
            reports[steps] = results
            expected_point = curve_record(results,previous,steps,'post_update',point['elapsed_seconds'],point['training'])
            expected_point.update(learner_updates=updates, checkpoint=f'checkpoint-{steps}')
            if canonical(point) != canonical(expected_point):
                raise ValueError('curve metrics/update counts differ from physical reports')
            if updates and (point['training']['update'] != updates
                    or point['training']['order'] != ['entropy','critic','actor','targets']):
                raise ValueError('final actual scheduled update phase evidence required')
            previous.append(point)
        with (root/'learning-curve.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 2:
            raise ValueError('two matching CSV curve points required')
        for row, point in zip(rows,history):
            for key, actual in row.items():
                expected_value = '' if point[key] is None else str(point[key])
                if actual != expected_value:
                    raise ValueError('CSV/JSONL curve disagreement')
        selected = max(history,key=lambda point: point['score'])
        for name, point in [('selected',selected),('final',history[-1])]:
            pointer = _json(root,name+'.json')
            expected_pointer = dict(steps=point['steps'], updates=point['learner_updates'],
                                    checkpoint=point['checkpoint'], score=point['score'])
            if canonical(pointer) != canonical(expected_pointer):
                raise ValueError('selected/final pointer or tie-selection mismatch')
            if canonical(_json(root,name+'-validation.json')) != canonical(reports[point['steps']]):
                raise ValueError('selected/final report identity mismatch')
            if f'{name.capitalize()} development checkpoint' not in (root/(name+'-validation.html')).read_text():
                raise ValueError('selected/final visual report missing')
        if (run['selected_step'] != selected['steps'] or run['selected_score'] != selected['score']
                or run['final_score'] != history[-1]['score']
                or 'Validation learning curve' not in (root/'learning-curve.html').read_text()):
            raise ValueError('completed selected/final summary/visual mismatch')
        return dict(saved_artifacts_verified=True, transitions=128, actual_updates=4, adam_steps=12,
                    recorded_action_economics_verified=True, report_inputs_baselines_directional_verified=True,
                    selection_final_curve_verified=True, frozen_carried_prediction_parity=True, retired_test_access=False, performance_promotion=False)
