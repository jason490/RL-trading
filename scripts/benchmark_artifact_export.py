"""Measure lossless archive compression on retained artifacts; never trains."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tarfile
import tempfile
import time


def inventory(root):
    files={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('regular retained artifact tree required')
        if path.is_file():
            with path.open('rb') as stream:
                files['artifacts/'+str(path.relative_to(root))]=hashlib.file_digest(stream,'sha256').hexdigest()
    return files


def benchmark(root,work):
    root=Path(root).resolve()
    if root.name!='artifacts' or not root.is_dir():
        raise ValueError('retained artifacts directory required')
    work=Path(work).resolve()
    if work.is_relative_to(root):
        raise ValueError('benchmark workspace must be outside retained artifacts')
    before=inventory(root)
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if not before or size>256*1024*1024:
        raise ValueError('nonempty at most256MiB retained tree required')
    results=[]
    with tempfile.TemporaryDirectory(prefix='archive-benchmark-',dir=work) as temporary:
        for level in (6,1,1,6):
            destination=Path(temporary)/'archive.tar.gz'
            start=time.perf_counter()
            with tarfile.open(destination,'w:gz',compresslevel=level) as archive:
                archive.add(root,arcname='artifacts')
            elapsed=time.perf_counter()-start
            restored={}
            with tarfile.open(destination) as archive:
                for member in archive.getmembers():
                    if member.isfile():
                        restored[member.name]=hashlib.file_digest(archive.extractfile(member),'sha256').hexdigest()
            if restored!=before:
                raise ValueError('lossless archive file inventory mismatch')
            results.append(dict(compresslevel=level,seconds=elapsed,bytes=destination.stat().st_size,
                                exact_member_content_verified=True))
    if inventory(root)!=before:
        raise ValueError('retained artifacts changed during read-only benchmark')
    return dict(schema='archive-compression-benchmark-v1',raw_bytes=size,files=len(before),
                python=platform.python_version(),machine=platform.machine(),samples=results,
                retained_files_unchanged=True,temporary_archives_removed=True,
                training_performed=False,colab_allocation=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact_root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.artifact_root.resolve()):
        parser.error('benchmark output must be outside retained artifacts')
    result=benchmark(args.artifact_root,args.work)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result,indent=2))
