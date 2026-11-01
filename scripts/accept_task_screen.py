"""Read completed verified cells and publish the fixed fitted-period decision."""
import argparse
from pathlib import Path

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.training.task_screen_acceptance import accept_stopped_screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = accept_stopped_screen(Path.cwd())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(args.output, canonical(result))
    if not result['gate']['passed']:
        from rl_stocks.core.contracts import digest
        write_once(Path.cwd() / '.runtime/task-conditioning-family-closure-v1.json', canonical(dict(
            gate_sha256=digest(result), gate_path=str(args.output), family_closed=True,
            development_allowed=False, automatic_retry_allowed=False, performance_promotion=False)))
    print('Training-only gate passed:', result['gate']['passed'])
    print('Development allowed:', result['gate']['development_allowed'])


if __name__ == '__main__':
    main()
