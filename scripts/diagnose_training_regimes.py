"""Summarize existing preregistered fitted-path reports; never fit or select."""
import json
from pathlib import Path
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.training_regimes import summarize_training_regimes

root = Path('results/training-fit-2026-10-20')
proof = json.loads((root/'summary.json').read_text())
protected = {str(root/'summary.json'):checksum(root/'summary.json')}
records = {}
for algorithm in ('ppo','sac','td3'):
    path = Path('artifacts/training-fit-2026-10-20')/algorithm/'training.json'
    protected[str(path)] = checksum(path)
    if protected[str(path)] != proof['records'][algorithm]['derived_report_sha256']:
        raise ValueError('verified fitted-path report required')
    records[algorithm] = summarize_training_regimes(json.loads(path.read_text()))
assert all(checksum(Path(p)) == h for p,h in protected.items())
output = Path('results/training-regimes-2026-10-20/summary.json')
if output.exists():raise ValueError('new output required')
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(dict(records=records, source_sha256=protected,
    script_sha256=checksum(Path(__file__)), module_sha256=checksum(Path('src/rl_stocks/reporting/training_regimes.py')),
    horizon_sessions=20, threshold=.1, source_files_unchanged=True,
    training_performed=False, dataset_opened=False, colab_allocated=False,test_evaluated=False,
    interpretation='Overlapping fitted-period decisions, not independent trades or realized short profits. Future cohorts are reporting-only; never learning, sampling or selection labels. Gross reference-price returns omit dividends/costs. Old50k models, not new100k diversity models; no causal algorithm ranking.'),indent=2)+'\n')
print(json.dumps(records,indent=2))
