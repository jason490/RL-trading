# Causal execution diagnostic — 2026-10-19

Four fixed rules were simulated on the existing 22-stock/four-ETF development
panel, 2021–2023, using the same next-session execution, fees, impact, dividends,
borrow, stops, partial exits and maximum 20-session holding period as the policy.
This is local deterministic simulation, not model training or a new Colab run.
The rules are diagnostic baselines, not learned models or deployable signals.

| Rule | Mean stock return | Mean broad-ETF return |
| --- | ---: | ---: |
| Cash | 0.00% | 0.00% |
| Fixed +95% exposure, wide protection | 27.61% | 11.64% |
| Fixed -95% exposure, wide protection | -37.29% | -27.77% |
| Causal EMA trend/flat rule | -16.26% | -29.59% |
| Same-symbol buy-and-hold | 32.63% | 19.47% |

The EMA rule goes +95% when price > EMA20 > EMA50, -95% when price < EMA20 <
EMA50, and flat otherwise. It reads only the current causal observation, using
the immutable training normalizer to recover the ratios. All noncash rules use
fixed widest action-contract protection: stop 5 ATR, near 8 ATR, width 4 ATR.
These fixed target zones are not substituted for learned policy outputs.

The constrained long rule demonstrates that the existing execution rules can
produce substantially higher returns than the matched models on this panel,
but still lag holding. Its shortfall is not an upper bound on learned returns:
95% exposure, forced holding limits and protective exits differ from holding.
The tested trend rule fails badly and should **not** be used as a profitable
teacher by assumption. These results support investigating action initialization,
exploration, protection churn and direction decisions before longer blind runs;
they do not establish the causal source of deterioration.

`protocol.json` fixes rule definitions, dataset/universe identities, default
execution config and relevant source checksums. `summary.json` contains group
metrics and the reporting-only declining-stock cohort; `per-symbol.csv` retains
all 104 rule/symbol observations and reconciled long/short P&L. No test interval
was evaluated. No model or checkpoint was selected. The additional 110-stock
acquisition has not been silently mixed into this existing-panel diagnostic.

Reproduce:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_execution_rules.py \
  --dataset data/datasets/672a21621091a56c1c7beb30557464e7edd8df38874494659bab2c83738e1bf9 \
  --universe configs/universes/sector_research.json \
  --output artifacts/execution-rules-audit
```
