"""The CLI's zero exit status alone does not prove remote Python succeeded."""
from collections import Counter
import hashlib
import json
import math
import csv
from io import StringIO
from pathlib import PurePosixPath
import tarfile

from rl_stocks.core.contracts import digest


def validate_archive(path, plan, dataset_id):
    with tarfile.open(path) as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        if 'artifacts/failure.txt' in names:
            raise RuntimeError('remote training failed; see retrieved artifacts/failure.txt')
        def read(name):
            stream = archive.extractfile(name)
            if stream is None:
                raise RuntimeError('missing required remote artifact: '+name)
            return stream.read()
        try:
            if plan.get('version') in (12,15,16,17,18,19,20,21,22,23) and dataset_id!=plan['dataset_contract']['identity']:
                raise RuntimeError('trend-context dataset identity mismatch')
            completion = json.loads(read('artifacts/completion.json'))
            hardware_ok = (completion['gpu'] is None if plan['accelerator']=='CPU'
                           else plan['accelerator'] in (completion['gpu'] or ''))
            if (completion['plan'] != plan or completion['dataset'] != dataset_id
                    or completion['python'][:2] != [3, 13] or not hardware_ok):
                raise RuntimeError('remote completion manifest mismatch')
            runs = [json.loads(read(name)) for name in sorted(names)
                    if name.startswith('artifacts/runs/') and name.endswith('/metadata.json')]
            expected = Counter((j['algorithm'], j['device'], j['smoke'], j['steps'], j.get('seed',plan.get('seed')))
                               for j in plan['jobs'])
            actual = Counter()
            for run in runs:
                if (run['status'] != 'complete' or run['dataset'] != dataset_id
                        or run['actual_steps'] != run['requested_steps']):
                    raise RuntimeError('incomplete remote training run')
                if plan.get('version') in (4,5,7,8,9,10,11,12,15,16,17,18,19,20,21,22,23):
                    job=plan['jobs'][0]
                    if plan.get('version') in (7,8,9,10,11,12,15,16,17,18,19,20,21,22,23):
                        from rl_stocks.training.initialization import initialization_spec
                        if run.get('initialization')!=initialization_spec(job['initialization'],job['algorithm']):
                            raise RuntimeError('initialization-study configuration mismatch')
                    if plan.get('version') in (9,10,11,12,15,16,17,18,19,20,21,22,23):
                        from rl_stocks.environment.observations.state import observation_spec
                        if (run.get('observation')!=observation_spec(job['observation_profile'])
                                or plan.get('version') not in (15,19,20,21,22,23) and run.get('start_sampling')!={'mode':'legacy_uniform'}):
                            raise RuntimeError('execution-observation configuration mismatch')
                    if plan.get('version') in (10,11,12,15,16,17,18,19,20,21,22,23):
                        from rl_stocks.rl.ppo.architecture import architecture_spec
                        from rl_stocks.rl.registry.factory import resolved_parameters
                        if (run.get('policy_architecture')!=architecture_spec(job['policy_variant'],7)
                                or run.get('resolved_parameters')!=resolved_parameters(job['algorithm'],job['smoke'],
                                    job.get('overrides') if 'overrides' in job or job['smoke'] or job['algorithm'] not in ('ppo','recurrent_ppo') else {'n_steps':1000,'batch_size':50},job['policy_variant'],7)):
                            raise RuntimeError('policy-architecture configuration mismatch')
                    if plan.get('version')==8:
                        if run.get('start_sampling',{}).get('mode')!=job['start_sampling']:
                            raise RuntimeError('causal-sampling mode mismatch')
                    if (run.get('reward_config')!={'kind':job.get('reward_kind','net_log_equity'),'scale':job['reward_scale']}
                            or run.get('milestones')!=job['milestones'] or run.get('evaluation_every')!=job['evaluation_every']
                            or not run.get('compact_checkpoints')):
                        raise RuntimeError('learning-study configuration mismatch')
                    prefix=next(name.rsplit('/',1)[0]+'/' for name in names
                        if name.endswith('/metadata.json') and json.loads(read(name)).get('bundle_id')==run['bundle_id'])
                    if plan.get('version')==8:
                        counts=run.get('sampling_diagnostics',{})
                        if (counts!=json.loads(read(prefix+'training-sampling.json'))
                                or counts.get('mode')!=job['start_sampling'] or counts.get('transitions')!=job['steps']
                                or not 0<=counts.get('downtrend_transitions',-1)<=job['steps']
                                or not 0<=counts.get('downtrend_starts',-1)<=counts.get('starts',0)
                                or counts.get('starts',0)<=0):
                            raise RuntimeError('causal-sampling diagnostic mismatch')
                    curve=[json.loads(line) for line in read(prefix+'learning-curve.jsonl').splitlines()]
                    points={r['steps']:r for r in curve}
                    if (sorted(points)!=list(range(job['evaluation_every'],job['steps']+1,job['evaluation_every']))
                            or any(not math.isfinite(r['score']) for r in curve)
                            or any(points[s]['phase']!='post_update' for s in job['milestones'])
                            or curve!=run['validation_checkpoints']):
                        raise RuntimeError('incomplete or inconsistent learning curve')
                    if plan.get('version') in (21,22,23) and prefix+'checkpoint-state.json' not in names:
                        raise RuntimeError('mandatory checkpoint identity missing')
                    if prefix+'checkpoint-state.json' in names:
                        from rl_stocks.training.checkpoint_state import validate_checkpoint_state
                        checkpoint=validate_checkpoint_state(None,json.loads(read(prefix+'checkpoint-state.json')),
                            read=lambda name:read(prefix+name))
                        if (not checkpoint['final_post_update'] or checkpoint['latest']['steps']!=job['steps']
                            or checkpoint['selected']['steps']!=run['selected_step']
                            or not math.isclose(checkpoint['score'],points[job['steps']]['score'],abs_tol=1e-12,rel_tol=0)):
                            raise RuntimeError('checkpoint identity/final completion mismatch')
                    table=list(csv.DictReader(StringIO(read(prefix+'learning-curve.csv').decode())))
                    if ([int(r['steps']) for r in table]!=sorted(points)
                            or any(not math.isclose(float(r['return_pct']),100*points[int(r['steps'])]['score'],abs_tol=1e-9) for r in table)):
                        raise RuntimeError('learning curve CSV mismatch')
                    read(prefix+'learning-curve.html')
                actual[(run['algorithm'], run['device'], run['smoke'], run['actual_steps'], run['seed'])] += 1
                folder = 'artifacts/frozen/'+run['bundle_id']+'/'
                hashes = json.loads(read(folder+'checksums.json'))
                if set(hashes) != {name[len(folder):] for name in names
                                   if name.startswith(folder) and name != folder+'checksums.json'}:
                    raise RuntimeError('remote bundle file set mismatch')
                for name, checksum in hashes.items():
                    if PurePosixPath(name).name != name or hashlib.sha256(read(folder+name)).hexdigest() != checksum:
                        raise RuntimeError('remote bundle checksum mismatch')
                if plan.get('version') in (9,10,11,12,15,16,17,18,19,20,21,22,23) and json.loads(read(folder+'manifest.json'))['metadata'].get('observation')!=run['observation']:
                    raise RuntimeError('frozen observation schema mismatch')
                if plan.get('version') in (10,11,12,15,16,17,18,19,20,21,22,23) and json.loads(read(folder+'manifest.json'))['metadata'].get('policy_architecture')!=run['policy_architecture']:
                    raise RuntimeError('frozen policy architecture mismatch')
                if plan.get('version') in (11,12,15,16,17,18,19,20,21,22,23) and json.loads(read(folder+'manifest.json'))['metadata'].get('reward_config')!=run['reward_config']:
                    raise RuntimeError('frozen reward configuration mismatch')
                if plan.get('version')==12:
                    from rl_stocks.features.registry.assemble import NAMES as BASE
                    from rl_stocks.features.trend.context import NAMES as EXTRA
                    frozen=json.loads(read(folder+'manifest.json'));c=plan['dataset_contract']
                    expected_names=list(BASE)+(list(EXTRA) if c['mode']!='matched_baseline' else [])
                    if (frozen.get('names')!=expected_names or frozen.get('feature_schema')!=c['schema']
                        or frozen.get('data_manifest',{}).get('trend_context',{}).get('mode')!=c['mode']):
                        raise RuntimeError('frozen trend-context contract mismatch')
                if plan.get('version')==13:
                    from rl_stocks.training.supervised.artifacts import verify_report
                    from rl_stocks.training.supervised.spec import SIGNAL_DATASET
                    manifest=json.loads(read(folder+'manifest.json'))
                    prefix=next(n.rsplit('/',1)[0]+'/' for n in names if n.endswith('/metadata.json') and json.loads(read(n)).get('bundle_id')==run['bundle_id'])
                    if (dataset_id!=SIGNAL_DATASET or manifest.get('kind')!='signal-feasibility-v1'
                        or run['algorithm']!='signal_feasibility' or run.get('device')!='cpu'
                        or manifest['metadata']!={k:v for k,v in run.items() if k!='bundle_id'}):raise RuntimeError('signal archive contract mismatch')
                    verify_report(read,prefix,run,manifest)
                if plan.get('version')==14:
                    from rl_stocks.training.supervised.forest_artifacts import verify_report
                    from rl_stocks.training.supervised.spec import SIGNAL_DATASET
                    manifest=json.loads(read(folder+'manifest.json'))
                    prefix=next(n.rsplit('/',1)[0]+'/' for n in names if n.endswith('/metadata.json') and json.loads(read(n)).get('bundle_id')==run['bundle_id'])
                    if (dataset_id!=SIGNAL_DATASET or manifest.get('kind')!='forest-signal-feasibility-v1'
                        or run['algorithm']!='forest_signal_feasibility' or run.get('device')!='cpu'
                        or manifest['metadata']!={k:v for k,v in run.items() if k!='bundle_id'}):raise RuntimeError('forest archive contract mismatch')
                    if hashlib.sha256(read(folder+'requirements.lock')).hexdigest()!=run['requirements_lock_sha256'] or json.loads(read(folder+'dependencies.json'))!=run['versions']:raise RuntimeError('forest dependency manifest mismatch')
                    verify_report(read,prefix,run,manifest)
                if plan.get('version')==15:
                    from rl_stocks.training.history.spec import PROTOCOL_SHA256,validate_counts
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'));arm=plan['history_study']['arm'];job=plan['jobs'][0]
                    prefix=next(n.rsplit('/',1)[0]+'/' for n in names if n.endswith('/metadata.json') and json.loads(read(n)).get('bundle_id')==run['bundle_id'])
                    counts=run.get('sampling_diagnostics',{})
                    summary=validate_counts(counts,arm,job['smoke'],job['steps'])
                    if (counts!=json.loads(read(prefix+'training-sampling.json')) or run.get('start_sampling')!=summary
                        or run.get('history_protocol_sha256')!=PROTOCOL_SHA256
                        or frozen.get('metadata')!={k:v for k,v in run.items() if k!='bundle_id'}
                        or frozen.get('names')!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or frozen['data_manifest'].get('history_augmentation')!=run.get('history_augmentation')
                        or run['history_augmentation']['arm']!=arm):raise RuntimeError('historical bundle/sampling contract mismatch')
                if plan.get('version')==16:
                    from rl_stocks.training.sac_state import PROTOCOL_SHA256
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'))
                    if (run.get('study_protocol_sha256')!=PROTOCOL_SHA256
                        or frozen.get('metadata')!={k:v for k,v in run.items() if k!='bundle_id'}
                        or frozen.get('names')!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or frozen['data_manifest'].get('included_splits')!=['train','validation']):raise RuntimeError('SAC state bundle contract mismatch')
                if plan.get('version')==17:
                    from rl_stocks.training.sac_temperature import PROTOCOL_SHA256,validate_initialization
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'))
                    validate_initialization(run.get('temperature_initialization',{}),plan['temperature_study']['arm'])
                    if (run.get('study_protocol_sha256')!=PROTOCOL_SHA256
                        or frozen.get('metadata')!={k:v for k,v in run.items() if k!='bundle_id'}
                        or frozen.get('names')!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or frozen['data_manifest'].get('included_splits')!=['train','validation']):raise RuntimeError('temperature bundle contract mismatch')
                if plan.get('version')==18:
                    from rl_stocks.training.td3_state import PROTOCOL_SHA256,validate_initialization
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'))
                    validate_initialization(run.get('td3_initialization',{}))
                    if (run.get('study_protocol_sha256')!=PROTOCOL_SHA256
                        or frozen.get('metadata')!={k:v for k,v in run.items() if k!='bundle_id'}
                        or frozen.get('names')!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or frozen['data_manifest'].get('included_splits')!=['train','validation']):raise RuntimeError('TD3 state bundle contract mismatch')
                if plan.get('version')==19:
                    from rl_stocks.training.diversity import PROTOCOL_SHA256,validate_initialization,validate_counts,protocol as diversity_protocol
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'));arm=plan['diversity_study']['arm'];job=plan['jobs'][0]
                    validate_initialization(run.get('diversity_initialization',{}))
                    checked=validate_counts(run.get('sampling_diagnostics',{}),arm,job['smoke'],job['steps'])
                    if (run.get('study_protocol_sha256')!=PROTOCOL_SHA256 or run.get('start_sampling')!=checked['summary']
                        or run['sampling_diagnostics']!=json.loads(read(prefix+'training-sampling.json'))
                        or frozen.get('metadata')!={k:v for k,v in run.items() if k!='bundle_id'}
                        or frozen.get('names')!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or digest(frozen.get('normalizer',{}))!=diversity_protocol()['normalizer_sha256']
                        or frozen['data_manifest'].get('included_splits')!=['train','validation']):raise RuntimeError('diversity bundle contract mismatch')
                if plan.get('version')==20:
                    from rl_stocks.training.sac_regularization import PROTOCOL_SHA256,validate_optimizer_record,validate_final_optimizer_record,protocol as reg_protocol
                    from rl_stocks.training.diversity import validate_counts
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'));arm=plan['regularization_study']['arm'];job=plan['jobs'][0]
                    validate_optimizer_record(run['regularization_initialization'],arm)
                    for field in ('regularization_final','regularization_selected'):
                        validate_final_optimizer_record(run[field],arm)
                    checked=validate_counts(run['sampling_diagnostics'],'expanded',job['smoke'],job['steps'])
                    if (run.get('study_protocol_sha256')!=PROTOCOL_SHA256 or run.get('start_sampling')!=checked['summary']
                        or run['sampling_diagnostics']!=json.loads(read(prefix+'training-sampling.json'))
                        or frozen.get('metadata')!={k:v for k,v in run.items() if k!='bundle_id'}
                        or frozen.get('names')!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or digest(frozen.get('normalizer',{}))!=reg_protocol()['normalizer_sha256']
                        or frozen['data_manifest'].get('included_splits')!=['train','validation']):raise RuntimeError('regularization bundle contract mismatch')
                if plan.get('version')==23:
                    from rl_stocks.training.recurrent_alignment import PROTOCOL_SHA256,validate_record,modes,validate_execution_counts,validate_execution_report,protocol as recurrent_protocol
                    from rl_stocks.training.diversity import validate_counts,sampling_summary
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'));arm=plan['alignment_study']['arm'];job=plan['jobs'][0]
                    validate_record(run['alignment_initialization'],arm,job['smoke'],initial=True)
                    validate_record(run['alignment_final'],arm,job['smoke'])
                    validate_record(run['alignment_selected'],arm,job['smoke'],selected_steps=run['selected_step'])
                    validate_execution_counts(run['training_execution'],arm,job['steps'])
                    for report_file in ('selected-validation.json','validation.json'):
                        validate_execution_report(json.loads(read(prefix+report_file)),arm)
                    if (run.get('action_execution')!=modes(arm) or run['training_execution']!=json.loads(read(prefix+'training-execution.json'))):
                        raise RuntimeError('actual alignment execution contract mismatch')
                    checked=validate_counts(run.get('sampling_diagnostics',{}),'expanded',job['smoke'],job['steps'])
                    if (run.get('study_protocol_sha256')!=PROTOCOL_SHA256
                        or run.get('start_sampling')!=sampling_summary('expanded',job['smoke'])
                        or run['sampling_diagnostics']!=json.loads(read(prefix+'training-sampling.json'))
                        or digest(frozen['normalizer'])!=recurrent_protocol()['normalizer_sha256']
                        or frozen['names']!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or frozen['data_manifest'].get('included_splits')!=['train','validation']
                        or frozen['metadata']!={k:v for k,v in run.items() if k!='bundle_id'}):
                        raise RuntimeError('alignment bundle/data/sampling contract mismatch')
                if plan.get('version')==22:
                    from rl_stocks.training.recurrent_state import PROTOCOL_SHA256,validate_record,protocol as recurrent_protocol
                    from rl_stocks.training.diversity import validate_counts,sampling_summary
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'));arm=plan['recurrent_state_study']['arm'];job=plan['jobs'][0]
                    validate_record(run['recurrent_state_initialization'],arm,job['smoke'],initial=True)
                    validate_record(run['recurrent_state_final'],arm,job['smoke'])
                    validate_record(run['recurrent_state_selected'],arm,job['smoke'],selected_steps=run['selected_step'])
                    checked=validate_counts(run.get('sampling_diagnostics',{}),'expanded',job['smoke'],job['steps'])
                    if (run.get('study_protocol_sha256')!=PROTOCOL_SHA256
                        or run.get('start_sampling')!=sampling_summary('expanded',job['smoke'])
                        or run['sampling_diagnostics']!=json.loads(read(prefix+'training-sampling.json'))
                        or digest(frozen['normalizer'])!=recurrent_protocol()['normalizer_sha256']
                        or frozen['names']!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or frozen['data_manifest'].get('included_splits')!=['train','validation']
                        or frozen['metadata']!={k:v for k,v in run.items() if k!='bundle_id'}):
                        raise RuntimeError('recurrent-state bundle/data/sampling contract mismatch')
                if plan.get('version')==21:
                    from rl_stocks.training.sac_discount import PROTOCOL_SHA256,validate_record,protocol as reg_protocol
                    from rl_stocks.training.diversity import validate_counts
                    from rl_stocks.features.registry.assemble import NAMES
                    frozen=json.loads(read(folder+'manifest.json'));arm=plan['discount_study']['arm'];job=plan['jobs'][0]
                    validate_record(run['discount_initialization'],arm,initial=True)
                    if (run['discount_final']['num_timesteps']!=job['steps'] or run['discount_selected']['num_timesteps']!=run['selected_step']):
                        raise RuntimeError('discount actual saved-model step mismatch')
                    for field in ('discount_final','discount_selected'):
                        validate_record(run[field],arm)
                    checked=validate_counts(run['sampling_diagnostics'],'expanded',job['smoke'],job['steps'])
                    if (run.get('study_protocol_sha256')!=PROTOCOL_SHA256 or run.get('start_sampling')!=checked['summary']
                        or run['sampling_diagnostics']!=json.loads(read(prefix+'training-sampling.json'))
                        or frozen.get('metadata')!={k:v for k,v in run.items() if k!='bundle_id'}
                        or frozen.get('names')!=list(NAMES) or frozen.get('feature_schema')!='daily-indicators-v1'
                        or digest(frozen.get('normalizer',{}))!=reg_protocol()['normalizer_sha256']
                        or frozen['data_manifest'].get('included_splits')!=['train','validation']):raise RuntimeError('discount bundle contract mismatch')
                if digest(json.loads(read(folder+'manifest.json'))) != run['bundle_id']:
                    raise RuntimeError('remote bundle identity mismatch')
            if actual != expected or len(completion['completed']) != len(runs):
                raise RuntimeError('missing or unexpected remote training jobs')
            return runs
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError('missing or malformed remote training artifacts') from exc
