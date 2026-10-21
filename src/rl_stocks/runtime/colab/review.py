"""Explicit operator-reviewed budget resumption; never called by the fleet."""
import argparse
import fcntl
import json
from pathlib import Path
from rl_stocks.runtime.colab.budget import Ledger,parse_usage
from rl_stocks.runtime.colab.fleet import CLI


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli-python',required=True)
    parser.add_argument('--stopped-session',required=True)
    parser.add_argument('--authorization',required=True)
    parser.add_argument('--minimum-account-balance',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path.cwd()
    if args.output.exists():raise ValueError('new review output required')
    with (root/'.runtime/allocation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        usage=parse_usage(CLI(root,args.cli_python).short(['usage']))
        ledger=Ledger(root/'.runtime/usage-ledger.json')
        review=ledger.review_resume(usage,args.authorization,args.stopped_session,args.minimum_account_balance)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(review,indent=2)+'\n')
        print(json.dumps(review))


if __name__=='__main__':main()
