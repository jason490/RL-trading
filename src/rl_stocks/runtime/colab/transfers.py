"""Bounded streaming downloads without putting runtime credentials in errors."""
from pathlib import Path
import time
from urllib.parse import quote


class TransferError(RuntimeError):
    pass


def download_binary(base_url, token, remote_path, destination, get=None):
    if get is None:
        import requests
        get=requests.get
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.partial')
    deadline=time.monotonic()+110
    try:
        with get(base_url.rstrip('/')+'/files/'+quote(remote_path.strip('/'),safe='/'),
                 params={'authuser':'0','colab-runtime-proxy-token':token},
                 stream=True,timeout=(10,15),allow_redirects=False) as response:
            if response.status_code!=200:
                raise TransferError(f'binary download HTTP {response.status_code}')
            count=0
            with temporary.open('wb') as output:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if time.monotonic()>deadline:raise TransferError('binary download deadline exceeded')
                    if chunk:output.write(chunk);count+=len(chunk)
            length=response.headers.get('Content-Length')
            if length is not None and not response.headers.get('Content-Encoding') and count!=int(length):
                raise TransferError('incomplete binary download')
            if remote_path.endswith('.tar.gz'):
                with temporary.open('rb') as stream:
                    if stream.read(2)!=b'\x1f\x8b':raise TransferError('expected gzip archive')
        temporary.replace(destination)
    except TransferError:
        raise
    except Exception as error:
        raise TransferError('binary download failed: '+type(error).__name__) from None
    finally:
        temporary.unlink(missing_ok=True)
