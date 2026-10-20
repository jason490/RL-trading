# Saved SAC temperature-path diagnosis

This read-only analysis compares selected and final paths from the completed
fixed temperature screen. It opens only saved 2021–2023 validation reports,
metadata and their committed verification summary, never a dataset or retired
test partition. No fitting, optimizer calls, provider requests, new checkpoints,
checkpoint selection or Colab allocation occurs. Source report hashes match the
verified study, and all nine source files remain unchanged.

| Candidate change: selected → final | Stocks | ETFs |
| --- | ---: | ---: |
| Net return change | -10.47pp | -6.25pp |
| Net long contribution change | -7.34pp | -3.45pp |
| Net short contribution change | -3.13pp | -2.80pp |
| Incremental cost-drag change | +0.91pp | -0.75pp |
| Saved-path non-cost excess change | -9.56pp | -7.00pp |
| Mean long exposure change | +5.16pp | +14.53pp |
| Mean absolute short exposure change | -0.36pp | -2.03pp |

Net change equals long plus short contribution changes, and equals saved-path
non-cost change minus incremental cost-drag change. Long/short contributions
already contain costs: do not subtract cost drag from them again. Cost add-back
is a decomposition of actual saved paths, not a zero-cost simulation or causal
estimate of changing fees. End-of-session exposure misses intraday positions.

The final policy is more long-exposed but earns less from both directions.
Stocks' average short contribution turns from +0.26pp to -2.87pp; ETF shorts
turn from +2.20pp to -0.61pp. Cost increases explain only a small part of stock
deterioration and cannot explain ETF deterioration, where costs decline.
Average rapid protective reentries rise from 37.41 to 46.64 for stocks and
50.25 to 52.75 for ETFs, but this association does not prove they cause losses.
Default selected/final weights coincide, so their path differences are zero.

The screen remains failed. Do not rescue it by selecting falling stocks,
removing costs, extending training, or sweeping temperature. The evidence does
not isolate exploration, value estimation or policy optimization as the cause.
Before another allocation, inspect the existing deterministic off-policy TD3
configuration and saved input/exploration contracts to determine whether a
full-execution-state, delayed-update comparison is a distinct justified test.
Prior TD3 development results were weak; a different algorithm is a hypothesis,
not an expected solution. Any later comparison needs fixed matched controls,
absolute stock/ETF/median gates, bounded smoke/screen, and prospective evidence.

[Full numeric records](summary.json),
[original failed study](../sac-temperature-2026-10-20/README.md).
Reproduce with a new output path:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_temperature_paths.py \
  --output artifacts/temperature-paths/new-summary.json
```

Eleven participation/report-contract checks pass. Actual full reports reconcile
costs, directional cash flows, equity, holding dates and net returns. Project
usage remains 36.069198/150 units; all task-owned Colab sessions remain stopped.
The objective is not achieved or promoted by this descriptive diagnosis.
