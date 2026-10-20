# Saved SAC diversity path diagnosis

Two completed100k seed11 SAC runs, selected/final development paths only.
No fitting, provider calls, dataset opening, Colab allocation or retired-test
access. Twelve numeric/behavior/report-contract checks pass; source report and
metadata hashes remain unchanged. [Exact accounting](summary.json), [CSV](summary.csv).

| Expanded final group | Net excess vs holding | Incremental cost drag | Saved-path non-cost excess |
| --- | ---: | ---: | ---: |
| Stocks | −6.56pp | 3.51pp | −3.05pp |
| ETFs | −5.59pp | 2.66pp | −2.93pp |

Net excess equals non-cost excess minus incremental costs. Adding costs back
is descriptive accounting, not a zero-cost replay: policy behavior would change
under different costs. Directional contributions already include costs.

Expanded final stock long/short contributions are25.83%/+0.23%; ETF14.00%/
−0.12%. Average end-of-session long exposure90.25%/91.53%, short absolute
exposure0.56%/0.04%. These are mostly-long policies, not established profitable
short strategies. The full study still fails joint declining-stock total/short
success0/3. Future decline labels are reporting-only.

Original selected-to-final stock return falls13.67pp: long contribution−7.77pp,
short−5.91pp; incremental cost drag increases1.91pp. ETF return falls15.57pp:
long−11.78pp, short−3.79pp; cost drag increases1.37pp. Both costs and non-cost
participation matter; these paths do not identify an optimizer failure.

Fees/impact on discretionary same-side resizes below5% equity contribute only
0.23pp stocks/0.21pp ETFs in expanded final. A small-resize deadband alone cannot
close this saved gap. Holding-limit/protective rules remain required.

Reproduce using retained reports, without fitting:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_participation.py \
  --state .runtime/diversity-screen-state.json --algorithm sac --steps 100000 \
  --output /tmp/sac-diversity-paths.json
```

Algorithm and transition count are explicit acceptance constraints; defaults
remain PPO/50000. Source states must be complete with verified cleanup, and all
phases must share the exact holding panel. Output must be new.

No extension of the failed diversity family or promotion follows. Before another
allocation, investigate whether the mostly-long fitted policy can distinguish
persistent declines from rebounds using existing strictly training-only paths;
any new fitting needs a separate frozen hypothesis and unchanged benchmark gates.
Cumulative usage stays36.263895/150; no active owned training sessions.
