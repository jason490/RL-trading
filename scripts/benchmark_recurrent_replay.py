"""Local metadata-only replay benchmark; no model, market data or optimization."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

from rl_stocks.training.recurrent_replay import RecurrentReplayIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('benchmark evidence must not be overwritten')
    index = RecurrentReplayIndex(100_000)
    start = time.perf_counter()
    for i in range(110_000):
        env = i % 8
        step = i // 8
        index.append(environment=env, episode=step // 512, step=step % 512,
                     symbol=f'SYNTHETIC-{env}', truncated=step % 512 == 511)
    append_seconds = time.perf_counter() - start
    start = time.perf_counter()
    count = index.valid_window_count(burn_in=16, unroll=16)
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    oracle = index.windows(burn_in=16, unroll=16)
    reference_seconds = time.perf_counter() - start
    assert {w.transitions[0].insertion_id for w in oracle} == {
        t.insertion_id for t in index._pool.starts}
    del oracle
    rng = np.random.default_rng(11)
    durations = []
    for _ in range(30):
        start = time.perf_counter()
        batch = index.sample(batch_size=256, burn_in=16, unroll=16, rng=rng)
        durations.append(time.perf_counter() - start)
        for seq in batch:
            index.resolve(seq)
    # Exercise cached maintenance after initialization, not only static sampling.
    start = time.perf_counter()
    for i in range(110_000, 120_000):
        env = i % 8
        step = i // 8
        index.append(environment=env, episode=step // 512, step=step % 512,
                     symbol=f'SYNTHETIC-{env}', truncated=step % 512 == 511)
    maintenance_seconds = time.perf_counter() - start
    root = Path(__file__).resolve().parents[1]
    source = root / 'src/rl_stocks/training/recurrent_replay.py'
    result = dict(schema='recurrent-replay-benchmark-v1', python=sys.version,
                  platform=platform.platform(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  capacity=100_000, initial_writes=110_000, environments=8, episode_length=512,
                  burn_in=16, unroll=16, batch_size=256, repetitions=30,
                  complete_windows=count, pool_equals_reference=True,
                  append_seconds=append_seconds, pool_initialization_seconds=build_seconds,
                  reference_enumeration_seconds=reference_seconds,
                  cached_sample_median_seconds=median(durations),
                  cached_sample_max_seconds=max(durations),
                  sample_timing_includes_resolve=False, all_sampled_sequences_resolved=True,
                  maintenance_writes=10_000, maintenance_seconds=maintenance_seconds,
                  model_constructed=False, optimizer_steps=0, colab_allocations=0,
                  scope='local synthetic metadata only; not learner/Colab throughput')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
