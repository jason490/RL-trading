# Recurrent fill-cost attribution

Saved selected/final reports for all three completed100k seed11 arms.18 numeric,
CLI, behavior and report checks pass. No fitting, dataset opening, provider calls,
retired-test access or Colab allocation; source report/metadata hashes unchanged.
[Exact per-symbol records](summary.json), [category CSV](fill-costs.csv).

| Masked recurrent final category | Stocks | ETFs |
| --- | ---: | ---: |
| Same-side share resizing | 2.1305pp | 1.5146pp |
| Fresh entries | 1.2242pp | 0.9743pp |
| Discretionary closes | 0.0033pp | 0.0025pp |
| Both reversal legs | 0.4531pp | 0.2335pp |
| Protective exits | 1.0531pp | 0.5388pp |
| Required holding-limit exits | 0.6357pp | 0.7255pp |
| Short borrowing | 0.6412pp | 0.7663pp |

These are gross observed costs as percentages of initial equity, averaged over
separate episodes. Their sums reconcile to recorded costs; incremental drag
versus holding subtracts holding's own costs. Fee/impact are counted exactly once
per fill; borrow is separate. Entries following protective exits stay entries,
not free reentries. Reasons such as `hold` can resize actual shares after price
changes, so categorization uses executed old/target quantities. Reverse close and
new opposite entry are both reversal legs. Unknown reasons/nonfinite/negative
costs or inconsistent totals reject. No daily decisions or simulator rules change.

The masked control's final stock/ETF net gaps are6.8056/5.1915pp. Arithmetic
addback of every same-side resize cost still leaves4.6751/3.6769pp gaps, outside
the−2pp target. Adding back all entries, resizes, discretionary closes AND
reversal costs still leaves approximately2.99/2.47pp gaps. These are bookkeeping
bounds on saved paths, not achievable counterfactual returns: exposure and risk
would change if fills changed. Mandatory stops, targets and20-session exits must
remain. Borrow is a real cost, not a penalty that can be switched off for success.
Tiny resize costs are still only0.067/0.081pp; no small deadband experiment follows.

Cost pressure exists, but cheaper discretionary turnover alone is insufficient
to establish comparability on these paths. A distinct future hypothesis needs
better net directional participation as well; do not extend the failed recurrent
state family or select the masked control retrospectively. More transitions or
removing required protections is not justified by this audit.

Reproduce with retained reports using the updated diagnostic:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_participation.py \
 --state .runtime/recurrent-state-screen-state.json \
 --algorithm ppo --algorithm recurrent_ppo --steps 100000 \
 --output /tmp/recurrent-fill-costs.json
```

Category means are under `records[].groups.*.equal_weight_cost_categories`;
per-symbol partitions retain exact fill counts, fees, impact, borrow and reasons.
Project remains38.014408/150, no active owned sessions. No promotion or prospective
success. Next diagnose whether saved policy direction or learned protective
exits dominate the remaining non-cost gap before another frozen protocol.
