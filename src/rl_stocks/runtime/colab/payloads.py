"""Isolated per-session upload payloads; never share mutable job-plan files."""
import json
from pathlib import Path
import tarfile
from rl_stocks.core.provenance import repository_state


def prepare_payload(root,dataset,plan,session):
    root=Path(root);folder=root/'.runtime/payloads'/session;folder.mkdir(parents=True,exist_ok=False)
    job=folder/'job-plan.json';job.write_text(json.dumps(plan,indent=2))
    source=folder/'source-git.json';source.write_text(json.dumps(repository_state(root),indent=2))
    archive=folder/'research-upload.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ('src','scripts','pyproject.toml','requirements.txt','requirements.lock','configs'):
            tar.add(root/name,arcname=name)
        tar.add(Path(dataset)/'dataset.json',arcname='dataset.json')
        tar.add(job,arcname='job-plan.json');tar.add(source,arcname='source-git.json')
    return archive
