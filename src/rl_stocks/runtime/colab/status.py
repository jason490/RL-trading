"""Compact read-only retained status; never establishes remote process liveness."""

import argparse
import json
from pathlib import Path
import tarfile


def checkpoint(path):
    try:
        with tarfile.open(path) as archive:
            matches = [m for m in archive if m.name.endswith('/checkpoint-state.json')]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 1048576:
                return dict(status='checkpoint_state_unavailable')
            value = json.load(archive.extractfile(matches[0]))
            return dict(status='retrieved', latest_steps=value['latest']['steps'],
                        latest_phase=value['latest']['phase'],
                        selected_steps=value['selected']['steps'],
                        latest_score=value['score'], provisional=True)
    except FileNotFoundError:
        return dict(status='not_retrieved')
    except (OSError, tarfile.TarError, EOFError, ValueError, KeyError, TypeError):
        return dict(status='unreadable_or_being_replaced')


def summarize(state_path, artifact_root, ledger_path):
    state = json.loads(Path(state_path).read_text())
    ledger = json.loads(Path(ledger_path).read_text())
    rows = []
    for session, record in state.get('sessions', {}).items():
        # Controller-generated names are single path components.
        if not session or Path(session).name != session or session in ('.', '..'):
            raise ValueError('safe owned session name required')
        rows.append(dict(session=session, task=record['task'], recorded_phase=record['phase'],
                         checkpoint=checkpoint(Path(artifact_root) / session / 'checkpoint.tar.gz')))
    return dict(recorded_fleet=state.get('fleet'), recorded_status=state.get('status'),
                remote_liveness_verified=False, cleanup_verified=state.get('cleanup_verified'),
                completed_tasks=sorted(state.get('completed', {})),
                conservative_project_units=ledger['consumed'],
                remaining_project_units=ledger['ceiling'] - ledger['consumed'],
                budget_stopped=ledger['budget_stopped'], sessions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'artifact-root', 'ledger'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.state, args.artifact_root, args.ledger), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
