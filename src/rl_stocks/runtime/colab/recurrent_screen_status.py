"""Compact local observations of an existing screen; never allocates, trains or accepts."""
import argparse
import io
import json
from pathlib import Path
import tarfile
import time


PREFIX = 'artifacts/comparison-cell/'
WANTED = {'learning-curve.jsonl', 'selected.json', 'learning-curve.html'}


def snapshot(path):
    """Read one bounded immutable byte copy; incomplete downloads remain observations."""
    if not path.exists():
        return {}, 'not_downloaded'
    with path.open('rb') as stream:
        raw = stream.read(256 * 1024 * 1024 + 1)
    if len(raw) > 256 * 1024 * 1024:
        raise ValueError('snapshot transport exceeds observation bound')
    found = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            total = 0
            seen = set()
            for index, member in enumerate(archive):
                total += member.size
                if index >= 512 or total > 256 * 1024 * 1024 or member.name in seen:
                    raise ValueError('bounded unique snapshot members required')
                seen.add(member.name)
                relative = member.name.removeprefix(PREFIX)
                if member.name.startswith(PREFIX) and relative in WANTED:
                    limit = 10 * 1024 * 1024 if relative.endswith('.html') else 1024 * 1024
                    if not member.isfile() or member.size > limit:
                        raise ValueError('bounded regular progress member required')
                    found[relative] = archive.extractfile(member).read()
    except (tarfile.TarError, EOFError):
        return {}, 'download_incomplete'
    return found, 'downloaded'


def observe(root, state_path, *, charts=None, now=None):
    root = Path(root).resolve()
    state = json.loads(Path(state_path).read_text())
    ledger = json.loads((root/'.runtime/usage-ledger.json').read_text())
    attempts = [a for a in ledger['attempts'] if a['session'] == state['fleet']]
    if len(attempts) != 1:
        raise ValueError('exact recorded fleet required')
    attempt = attempts[0]
    rows = []
    for session, cell in state['sessions'].items():
        if not session.startswith('rl-stocks-cpu-') or Path(session).name != session:
            raise ValueError('owned CPU session name required')
        folder = root/'artifacts/colab'/session
        archive = folder/'artifacts.tar.gz'
        kind = 'final_export'
        if not archive.exists():
            archive = folder/'checkpoint.tar.gz'
            kind = 'checkpoint_snapshot'
        files, status = snapshot(archive)
        curve = [json.loads(line) for line in files.get('learning-curve.jsonl', b'').splitlines()]
        row = dict(session=session, task=cell['task'], phase=cell['phase'], snapshot=status,
                   archive_kind=kind,
                   provisional_curve=[{k:point[k] for k in ('steps', 'return_pct', 'excess_return_pp')}
                                      for point in curve],
                   selected=None if 'selected.json' not in files else json.loads(files['selected.json']))
        if charts is not None and 'learning-curve.html' in files:
            directory = Path(charts)
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory/(session+'-provisional-curve.html')
            if destination.is_symlink():
                raise ValueError('chart destination must not be a symlink')
            destination.write_bytes(files['learning-curve.html'])
            row['chart'] = str(destination.resolve())
        rows.append(row)
    now = time.time() if now is None else now
    return dict(fleet=state['fleet'], recorded_status=state['status'],
                process_liveness_verified=False, full_artifact_acceptance=False, partial=True,
                project_units=ledger['consumed'], budget_stopped=ledger['budget_stopped'],
                aggregate_units_per_hour=attempt['rate'],
                meter_age_seconds=round(now-attempt['last_time'], 1), sessions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Captured source root with artifacts')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--charts', type=Path, help='Optionally refresh partial HTML copies here')
    args = parser.parse_args()
    print(json.dumps(observe(args.root, args.state, charts=args.charts), sort_keys=True))


if __name__ == '__main__':
    main()
