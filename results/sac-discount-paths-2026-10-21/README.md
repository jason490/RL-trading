# Saved discount-policy path diagnosis

Completed100k seed11 selected/final development reports, inference/accounting only.
No fitting, provider calls, dataset opening, retired-test access or Colab allocation.
Source report/metadata hashes remain unchanged. [Exact results](summary.json),
[compact table](summary.csv).

| Final candidate group | Excess vs holding | Incremental cost drag | Non-cost excess |
| --- | ---: | ---: | ---: |
| Stocks | −8.74pp | 4.01pp | −4.73pp |
| ETFs | −6.82pp | 2.89pp | −3.94pp |

Net excess equals non-cost excess minus incremental costs. Cost addback describes
saved paths; it does not predict changed behavior under zero costs. Both components
worsen against control (stock−6.56pp, ETF−5.59pp net excess). Lower discount did not
resolve participation or costs; these paths alone do not establish a learning cause.

Candidate final stock long/short contributions25.30%/−1.42%; ETFs13.94%/−1.30%.
Average end-session long exposure88.05%/89.91%, short0.80%/0.30%. Positive short
P&L on three reporting-only declining stocks therefore does not imply positive
short contribution across the whole panel. Control stock shorts contribute+0.23%,
ETF−0.12%. The candidate remains predominantly long and fails the original gates.

Candidate selected-to-final stock return drops0.79pp, with nearly unchanged
incremental cost drag: long contribution drops0.46pp, shorts0.33pp. ETF return
improves0.20pp, principally long contribution. More steps are not supported by
this failed fixed comparison; no discount sweep or confirmation follows.

Reproduce with retained reports:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_participation.py \
 --state .runtime/discount-screen-state.json --algorithm sac --steps 100000 \
 --output /tmp/discount-paths.json
```

Project usage remains37.897158/150. Next audit existing recurrent-policy input
and memory-state contracts before proposing a distinct matched hypothesis; this
is an audit, not authorization to extend this failed discount family. Robust
multi-seed and prospective success remain unproven.

Verification:12 participation, behavior and report-contract checks pass.
