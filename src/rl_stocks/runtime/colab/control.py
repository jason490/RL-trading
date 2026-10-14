"""File-based graceful interruption for an owned runner across PID namespaces."""
import json
import os
from pathlib import Path
import signal
import time


def process_stop_request(root, parent_info=None, send_signal=None):
    root=Path(root).resolve()
    request_path=root/'.runtime/stop-owned-runner.json'
    if not request_path.exists():return False
    request=json.loads(request_path.read_text())
    ledger=json.loads((root/'.runtime/usage-ledger.json').read_text())
    if not ledger['attempts']:return False
    active=ledger['attempts'][-1]
    if active['status']!='active' or request.get('session')!=active['session']:
        return False
    if parent_info is None:
        pid=os.getppid()
        arguments=[a.decode() for a in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if a]
        cwd=Path(os.readlink(f'/proc/{pid}/cwd'))
    else:
        pid,arguments,cwd=parent_info
    approved=any(a=='rl_stocks.runtime.colab.runner' or Path(a).name in
                 ('run-study-campaign.py','run-study-batches.py') for a in arguments)
    if pid<=1 or Path(cwd).resolve()!=root or not approved:
        return False
    request_path.unlink()
    (root/'.runtime/handled-stop.json').write_text(json.dumps({**request,'handled':time.time()}))
    (send_signal or os.kill)(pid,signal.SIGINT)
    return True
