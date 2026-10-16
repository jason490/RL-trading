"""Detect terminal CLI failures without exposing authenticated tracebacks."""
import json,re
from pathlib import Path


class TransportLost(RuntimeError):pass


def export_ready(root,session,get=None):
    """Probe only the atomic export marker, never print authenticated errors."""
    if get is None:
        import requests
        get=requests.get
    try:
        state=json.loads((Path(root)/'.runtime/sessions.json').read_text())[session]
        response=get(state['url'].rstrip('/')+'/api/contents/content/research-artifacts.ready',
            params={'authuser':'0','colab-runtime-proxy-token':state['token'],'content':'0'},
            timeout=(5,10),allow_redirects=False)
        if response.status_code==200:return True
        if response.status_code==404:return False
    except Exception:pass
    raise TransportLost('remote export status unavailable') from None


def connection_lost(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536))
            tail=stream.read().decode(errors='replace')
    except OSError:return False
    tail=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',tail)
    return any(line.strip()=='RuntimeError: Connection was lost.' for line in tail.splitlines())


def worker_not_started(root,session,get=None):
    """Only an authenticated 404 permits retry; outages/unknown states fail closed."""
    if get is None:
        import requests
        get=requests.get
    try:
        state=json.loads((Path(root)/'.runtime/sessions.json').read_text())[session]
        response=get(state['url'].rstrip('/')+'/api/contents/content/rl-stocks',
            params={'authuser':'0','colab-runtime-proxy-token':state['token'],'content':'0'},
            timeout=(5,10),allow_redirects=False)
        return response.status_code==404
    except Exception:return False
