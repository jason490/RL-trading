"""Import historical evidence, audit coverage and collect listing candidates."""
import argparse
import json
from pathlib import Path
from rl_stocks.data.universe.historical.storage import put_history, load_history
from rl_stocks.data.universe.historical.coverage import audit
from rl_stocks.reporting.historical import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    ingest = sub.add_parser('import')
    ingest.add_argument('file',help='Canonical version-1 history JSON')
    ingest.add_argument('--output',default='data/historical')
    check = sub.add_parser('audit')
    check.add_argument('history',help='Content-addressed history directory')
    check.add_argument('--output',required=True)
    check.add_argument('--require-complete',action='store_true',help='Exit nonzero on any coverage failure')
    listings = sub.add_parser('listings')
    listings.add_argument('--date',required=True)
    listings.add_argument('--state',required=True,choices=['active','delisted'])
    listings.add_argument('--output',default='data/listings')
    args = parser.parse_args()
    if args.command == 'import':
        print(put_history(json.loads(Path(args.file).read_text()),args.output))
    elif args.command == 'listings':
        from rl_stocks.data.providers.alphavantage.listings import fetch_listings
        print(fetch_listings(args.date,args.state,args.output))
    else:
        value = audit(load_history(args.history))
        print(report(value,args.output))
        print(json.dumps({'coverage_passed':value['coverage_passed'],'training_ready':False,
                          'totals':value['totals'],'issue_counts':value['issue_counts']}))
        if args.require_complete and not value['coverage_passed']:
            raise SystemExit(2)


if __name__ == '__main__':main()
