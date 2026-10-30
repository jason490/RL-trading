"""Allowlisted100k cell transport and independent full saved-artifact verification."""
import json
from pathlib import Path
import tarfile
import tempfile

from rl_stocks.core.contracts import canonical
from rl_stocks.training.recurrent_comparison import protocol
from rl_stocks.training.recurrent_comparison_verify import ROOT_FILES,verify_cell

ROOT=Path(__file__).resolve().parents[4]


def validate_screen_archive(path,plan,dataset_id):
    from rl_stocks.runtime.colab.jobs import validate_plan
    validate_plan(plan)
    if dataset_id!=protocol()['dataset']:
        raise RuntimeError('fixed full development screen dataset required')
    arm=plan['comparison_screen']['arm']
    suffixes=('manifest.json','state.pt' if arm=='gru_sac' else 'model.zip')
    folders=['initial']+[f'checkpoint-{s}' for s in range(10000,100001,10000)]
    local_files=ROOT_FILES|{f'{folder}/{name}' for folder in folders for name in suffixes}
    required={'artifacts/completion.json'}|{'artifacts/comparison-cell/'+name for name in local_files}
    directories={'artifacts','artifacts/comparison-cell'}|{'artifacts/comparison-cell/'+name for name in folders}
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        files={m.name for m in members if m.isfile()}
        if (len({m.name for m in members})!=len(members) or not required<=files
                or files-required-{'artifacts/bootstrap-log.txt'}
                or any(not m.isfile() and (not m.isdir() or m.name not in directories) for m in members)
                or sum(m.size for m in members)>256*1024*1024):
            raise RuntimeError('complete bounded allowlisted screen archive required')
        with tempfile.TemporaryDirectory(prefix='recurrent-screen-verify-') as temporary:
            archive.extractall(temporary,filter='data')
            root=Path(temporary)/'artifacts'
            completion=json.loads((root/'completion.json').read_text())
            if (canonical(completion['plan'])!=canonical(plan) or completion['dataset']!=dataset_id
                    or completion['gpu'] is not None or completion['python'][:2]!=[3,13]
                    or completion['source_git'].get('dirty')!=''):
                raise RuntimeError('clean default-kernel CPU screen completion required')
            dataset=json.loads((ROOT/'data/datasets'/dataset_id/'dataset.json').read_text())
            proof=verify_cell(root/'comparison-cell',dataset,ROOT/'.runtime/recurrent-comparison-starts-v1',
                expected_arm=arm,expected_seed=11,expected_source_hash=completion['source_hash'],
                expected_git_revision=completion['source_git']['revision'])
            return [dict(algorithm=arm,seed=11,actual_steps=100000,status='complete',
                         bundle_id=proof['artifact_inventory_sha256'],cell=proof)]
