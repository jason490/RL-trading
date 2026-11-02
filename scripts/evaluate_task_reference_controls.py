"""Fixed selected/final control evaluation; no training or Colab allocation."""
import argparse
from pathlib import Path
import signal

from rl_stocks.evaluation.task_reference_controls import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/task-reference-controls-v1')
    args = parser.parse_args()
    def expired(signum, frame):
        raise TimeoutError('reference-control evaluation exceeded1200s deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1200)
    try:
        result = run(Path.cwd(), Path(args.output).resolve())
        print('Complete reference controls retained; candidate admission remains pending', flush=True)
    finally:
        signal.alarm(0)
