"""Download one fixed-scope SEC archive; request identity stays private."""
import argparse
from pathlib import Path

from rl_stocks.data.providers.sec.download import download_archive, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quarter', required=True)
    parser.add_argument('--root', type=Path, default=Path('data/raw/sec'))
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    args = parser.parse_args()
    path = download_archive(args.root, args.quarter, identity(args.env_file))
    print('Archive retained:', path)
    print('Acquisition only; run structural audit separately. Financial features remain disabled.')


if __name__ == '__main__':
    main()
