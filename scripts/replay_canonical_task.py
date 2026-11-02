"""Execute one fixed reference saved-market diagnostic with a hard local deadline."""
import argparse
from pathlib import Path
import signal

from rl_stocks.evaluation.canonical_diagnostic import protocol, run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/canonical-market-replay-v1')
    args = parser.parse_args()
    root = Path.cwd()

    def expired(signum, frame):
        raise TimeoutError('reference market replay exceeded fixed deadline')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(protocol(root)['deadline_seconds'])
    try:
        result = run(root, args.output)
        print({k: result[k] for k in ('episodes', 'predictions', 'replay_seconds', 'results_sha256')})
    finally:
        signal.alarm(0)
