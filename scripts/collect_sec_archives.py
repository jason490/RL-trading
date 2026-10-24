"""Bounded sequential acquisition of the preregistered SEC archive range."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from rl_stocks.data.providers.sec.download import download_archive, download_master_archive, identity

QUARTERS = tuple(f'{year}q{quarter}' for year in range(2009, 2024)
                 for quarter in range(1, 5) if (year, quarter) >= (2009, 2))
MAX_BYTES = 8_000_000_000
DEADLINE_SECONDS = 3600


def _save(path, state):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(state, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def collect(root, state_path, user_agent, *, source_kind="financial"):
    """New run only; never automatically restart a failed/interrupted collection."""
    if source_kind not in {'financial', 'master'}:
        raise ValueError('supported source kind required')
    request_limit = 256_000_000 if source_kind == 'financial' else 10_000_000
    total_limit = MAX_BYTES if source_kind == 'financial' else 1_000_000_000
    root, state_path = Path(root), Path(state_path)
    root.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            raise ValueError('existing collection journal requires review; automatic restart disabled')
        name = 'filing_coverage_audit.json' if source_kind == 'financial' else 'filing_master_audit.json'
        protocol = (Path('configs/experiments') / name).read_bytes()
        spec = json.loads(protocol)
        if spec['archive_quarters'] != {'first': QUARTERS[0], 'last': QUARTERS[-1]}:
            raise ValueError('collection differs from preregistered source range')
        started = time.time()
        state = dict(schema='sec-quarter-collection-v1', status='running', source_kind=source_kind,
                     protocol_sha256=hashlib.sha256(protocol).hexdigest(),
                     started_utc_epoch=started, deadline_utc_epoch=started + DEADLINE_SECONDS,
                     compressed_byte_limit=total_limit, acquired_bytes=0,
                     planned_quarters=list(QUARTERS), completed=[], current_quarter=None,
                     training_performed=False, colab_allocated=False,
                     financial_features_enabled=False)
        _save(state_path, state)
        try:
            for quarter in QUARTERS:
                if time.time() + 140 > state['deadline_utc_epoch']:
                    raise TimeoutError('insufficient operation reserve before collection deadline')
                if state['acquired_bytes'] + request_limit > total_limit:
                    raise ValueError('collection byte reserve exhausted')
                if shutil.disk_usage(root).free < 2_000_000_000 + request_limit:
                    raise ValueError('collection free-disk reserve exhausted')
                state['current_quarter'] = quarter
                _save(state_path, state)
                downloader = download_archive if source_kind == 'financial' else download_master_archive
                archive = downloader(root, quarter, user_agent)
                size = archive.stat().st_size
                state['completed'].append(dict(quarter=quarter, archive_sha256=archive.parent.name,
                                               path=str(archive), bytes=size))
                state['acquired_bytes'] += size
                _save(state_path, state)
                print(f"Retained {quarter}: {len(state['completed'])}/{len(QUARTERS)}, "
                      f"{state['acquired_bytes']} compressed bytes", flush=True)
            state['status'] = 'downloaded_only'
            state['current_quarter'] = None
        except BaseException as error:
            state['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
            state['error_type'] = type(error).__name__
            raise
        finally:
            state['updated_utc_epoch'] = time.time()
            _save(state_path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source-kind', choices=['financial', 'master'], default='financial')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    args = parser.parse_args()
    root = args.root or Path('data/raw/sec' if args.source_kind == 'financial' else 'data/raw/sec-masters')
    collect(root, args.state, identity(args.env_file), source_kind=args.source_kind)


if __name__ == '__main__':
    main()
