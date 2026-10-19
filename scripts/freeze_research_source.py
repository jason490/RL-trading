"""Export committed research code while preserving the project's single ledger."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='HEAD')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path.cwd().resolve();target=a.output.resolve()
    if not (root/'.runtime/usage-ledger.json').is_file():raise ValueError('existing project ledger required')
    if target.exists():raise ValueError('new export directory required; do not overwrite source')
    target.mkdir(parents=True)
    revision=subprocess.check_output(['git','rev-parse',a.revision],text=True).strip()
    archive=target.with_suffix('.tar')
    with archive.open('xb') as out:
        subprocess.run(['git','archive',revision,'src','scripts','configs','pyproject.toml','requirements.txt','requirements.lock'],stdout=out,check=True)
    with tarfile.open(archive) as stream:stream.extractall(target,filter='data')
    (target/'source-git.json').write_text(json.dumps({'revision':revision,'dirty':''})+'\n')
    for name in ('.runtime','data','artifacts','.venv'):(target/name).symlink_to(root/name,target_is_directory=True)
    assert (target/'.runtime/usage-ledger.json').resolve()==(root/'.runtime/usage-ledger.json').resolve()
    assert (target/'.runtime/allocation.lock').resolve()==(root/'.runtime/allocation.lock').resolve()
    code='from rl_stocks.core.provenance import source_identity;print(source_identity())'
    source_hash=subprocess.check_output([str(root/'.venv/bin/python'),'-c',code],cwd=target,
        env=dict(os.environ,PYTHONPATH=str(target/'src')),text=True).strip()
    proof={'source_revision':revision,'source_hash':source_hash,'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
           'shared_original_ledger_and_lock_verified':True,'root':str(target),'sessions_allocated':0}
    (target/'export-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__=='__main__':main()
