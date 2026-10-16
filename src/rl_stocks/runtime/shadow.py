"""Local/server shadow CLI. Inference and simulation only; no broker dependency."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import time
from rl_stocks.core.contracts import canonical, utcnow
from rl_stocks.core.files import write_once
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.inference.shadow.protocol import register, read_protocol
from rl_stocks.inference.shadow.journal import transaction, records
from rl_stocks.inference.shadow.runner import tick
from rl_stocks.inference.shadow.collect import collect
from rl_stocks.reporting.shadow import report


def prepare(recipe, bundle_root, snapshot, output):
    recipe = json.loads(Path(recipe).read_text())
    entries = []
    for arm in recipe['arms']:
        matches = list(Path(bundle_root).glob('**/'+arm['bundle_id']+'/manifest.json'))
        if not matches:
            raise ValueError('missing frozen bundle: '+arm['bundle_id'])
        entries.append({k: v for k, v in arm.items() if k != 'bundle_id'} |
                       {'bundle': str(matches[0].parent.resolve())})
    value = {k: v for k, v in recipe.items() if k != 'arms'}
    value.update(snapshot=str(Path(snapshot).resolve()), arms=entries)
    write_once(output, canonical(value))
    return output


def ready(root, now=None):
    """Only poll the provider after a new completed, post-registration session."""
    import exchange_calendars as xc
    current = timestamp(now or utcnow())
    protocol = read_protocol(root)
    calendar = xc.get_calendar('XNYS', start=str(current.date()-timedelta(days=14)),
                               end=str(current.date()+timedelta(days=2)))
    sessions = [s for s in calendar.sessions
                if timestamp(calendar.session_close(s).isoformat())+timedelta(minutes=15) <= current]
    last = sessions[-1]
    if timestamp(calendar.session_close(last).isoformat()) <= timestamp(protocol['not_before']):
        return {'status': 'waiting_for_forward_close'}
    with transaction(Path(root)/'journal.sqlite3') as db:
        journal = records(db)
    settled = sum(bool(r['settlements']) for r in journal)
    if settled >= protocol['review_sessions']:
        return {'status': 'review_due', 'settled_sessions': settled}
    if journal and journal[-1]['session'] == str(last.date()):
        return {'status': 'already_recorded', 'session': str(last.date())}
    return {'status': 'ready'}


def cycle(root, receipts):
    status = ready(root)
    if status['status'] == 'ready':
        status = tick(root, collect(root, receipts))
    if status['status'] == 'recorded' or not (Path(root)/'report.html').exists():
        report(root, Path(root)/'report.html')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('prepare')
    build.add_argument('--recipe', required=True)
    build.add_argument('--bundles', required=True)
    build.add_argument('--snapshot', required=True)
    build.add_argument('--output', required=True)
    init = sub.add_parser('init')
    init.add_argument('config')
    init.add_argument('--output', default='artifacts/shadow')
    for command in ('tick', 'report', 'run'):
        cmd = sub.add_parser(command)
        cmd.add_argument('campaign')
        if command == 'tick':
            cmd.add_argument('snapshot')
        if command == 'run':
            cmd.add_argument('--receipts', default='data/shadow')
            cmd.add_argument('--poll-seconds', type=int, default=900)
            cmd.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.recipe, args.bundles, args.snapshot, args.output))
    elif args.command == 'init':
        print(register(json.loads(Path(args.config).read_text()), args.output))
    elif args.command == 'tick':
        print(json.dumps(tick(args.campaign, args.snapshot)))
        report(args.campaign, Path(args.campaign)/'report.html')
    elif args.command == 'report':
        result = report(args.campaign, Path(args.campaign)/'report.html')
        print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'curves')}))
    else:
        if args.poll_seconds < 60:
            parser.error('poll interval must be at least 60 seconds')
        while True:
            try:
                result = cycle(args.campaign, args.receipts)
            except Exception as error:
                # A failure never writes a partial decision. Keep a durable stop
                # event for the operator; no automatic restart or stale backfill.
                with (Path(args.campaign)/'failures.jsonl').open('a') as file:
                    file.write(json.dumps({'time': utcnow(), 'type': type(error).__name__,
                                           'message': str(error)})+'\n')
                    file.flush()
                raise
            print(json.dumps(result), flush=True)
            if args.once or result['status'] == 'review_due':
                break
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
