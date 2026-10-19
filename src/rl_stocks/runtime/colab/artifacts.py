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
                if plan.get('version') in (4,5,7,8,9,10):
                    job=plan['jobs'][0]
                    if plan.get('version') in (7,8,9,10):
                        from rl_stocks.training.initialization import initialization_spec
                        if run.get('initialization')!=initialization_spec(job['initialization'],job['algorithm']):
                            raise RuntimeError('initialization-study configuration mismatch')
                    if plan.get('version') in (9,10):
                        from rl_stocks.environment.observations.state import observation_spec
                        if (run.get('observation')!=observation_spec(job['observation_profile'])
                                or run.get('start_sampling')!={'mode':'legacy_uniform'}):
                            raise RuntimeError('execution-observation configuration mismatch')
                    if plan.get('version')==10:
                        from rl_stocks.rl.ppo.architecture import architecture_spec
                        from rl_stocks.rl.registry.factory import resolved_parameters
                        if (run.get('policy_architecture')!=architecture_spec(job['policy_variant'],7)
                                or run.get('resolved_parameters')!=resolved_parameters('ppo',job['smoke'],
                                    None if job['smoke'] else {'n_steps':1000,'batch_size':50},job['policy_variant'],7)):
                            raise RuntimeError('policy-architecture configuration mismatch')
                    if plan.get('version')==8:
                        if run.get('start_sampling',{}).get('mode')!=job['start_sampling']:
                            raise RuntimeError('causal-sampling mode mismatch')
                    if (run.get('reward_config')!={'kind':'net_log_equity','scale':job['reward_scale']}
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
                if plan.get('version') in (9,10) and json.loads(read(folder+'manifest.json'))['metadata'].get('observation')!=run['observation']:
                    raise RuntimeError('frozen observation schema mismatch')
                if plan.get('version')==10 and json.loads(read(folder+'manifest.json'))['metadata'].get('policy_architecture')!=run['policy_architecture']:
                    raise RuntimeError('frozen policy architecture mismatch')
                if digest(json.loads(read(folder+'manifest.json'))) != run['bundle_id']:
                    raise RuntimeError('remote bundle identity mismatch')
            if actual != expected or len(completion['completed']) != len(runs):
                raise RuntimeError('missing or unexpected remote training jobs')
            return runs
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError('missing or malformed remote training artifacts') from exc
