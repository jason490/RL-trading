"""Isolated per-session upload payloads; never share mutable job-plan files."""
import json
from pathlib import Path
import tarfile
from rl_stocks.core.provenance import repository_state
from rl_stocks.runtime.colab.source_archive import source_member


def prepare_payload(root,dataset,plan,session):
    root=Path(root);folder=root/'.runtime/payloads'/session;folder.mkdir(parents=True,exist_ok=False)
    job=folder/'job-plan.json';job.write_text(json.dumps(plan,indent=2))
    source=folder/'source-git.json';source.write_text(json.dumps(repository_state(root),indent=2))
    archive=folder/'research-upload.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ('src','scripts','pyproject.toml','requirements.txt','requirements.lock','configs'):
            tar.add(root/name,arcname=name,filter=source_member)
        if plan.get('version') in (41,42,43):
            tar.add(root/'.runtime/recurrent-comparison-starts-v1',arcname='comparison-starts')
        if plan.get('version') == 48:
            from rl_stocks.training.task_reference_runtime import checksum, execution_spec
            initial = root/'artifacts/colab/rl-stocks-cpu-bf3f86798a17/retrieved/artifacts/task-screen/initial.zip'
            if checksum(initial) != execution_spec(root)['initial_control_zip_sha256']:
                raise ValueError('retained shared start archive changed')
            tar.add(initial, arcname='initial-control.zip')
        tar.add(Path(dataset)/'dataset.json',arcname='dataset.json')
        tar.add(job,arcname='job-plan.json');tar.add(source,arcname='source-git.json')
    return archive
