"""Record explicit user-approved budget review; does not provision a session."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from rl_stocks.runtime.colab.budget import Ledger,parse_usage
from rl_stocks.training.recurrent_execution_review import authorize_profile_review,review_spec


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization',required=True,help='Explicit user authorization for this one corrected profile')
    parser.add_argument('--cli-python',required=True,help='Existing official Colab CLI interpreter')
    args=parser.parse_args()
    root=Path.cwd()
    with (root/'.runtime/allocation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        usage=parse_usage(subprocess.run([args.cli_python,str(root/'scripts/colab_safe.py'),'usage'],
            check=True,text=True,capture_output=True,timeout=30).stdout)
        ledger=Ledger(root/'.runtime/usage-ledger.json')
        ledger.settle_stopped(review_spec()['original_stopped_fleet'],usage)
        record=authorize_profile_review(ledger,usage,args.authorization)
    print(json.dumps(dict(review_recorded=True,allocation_started=False,
                         consumed_preserved=record['resumption_review']['consumed_preserved'],
                         scope=record['scope']),indent=2))
