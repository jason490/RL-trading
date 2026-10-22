# Alignment risk and action audit

Report-only diagnosis of all six accepted selected/final phases from the [alignment study](../recurrent-alignment-2026-10-21/README.md). The original 22 stocks and four ETFs are unchanged. No fitting, dataset opening, provider requests, Colab allocation or retired-test evaluation occurred. Source report hashes were checked against the accepted study and verified unchanged afterward.

## Final-checkpoint observations

| Model | Stock return | ETF return | Stock incremental cost drag | ETF incremental cost drag | Mean stock maximum drawdown | Mean ETF maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive | 25.82% | 14.27% | 6.11pp | 4.73pp | −30.81% | −25.46% |
| Posthoc unit | 35.40% | 18.55% | 7.26pp | 4.51pp | −34.40% | −26.58% |
| Aligned unit | 14.86% | 8.77% | 11.08pp | 8.37pp | −37.89% | −34.78% |

Matched holding returns are 32.63% stocks and 19.47% ETFs. Incremental cost drag is relative to matched holding, expressed as percentage points of initial equity. Saved-path noncost excess for aligned unit is −6.68pp stocks and −2.33pp ETFs; for posthoc unit it is +10.03pp and +3.59pp. These identities are bookkeeping, not cost-free counterfactual performance.

Aligned unit's worst peak-to-trough windows contain mean costs of 2.87% of peak equity for stocks and 2.63% for ETFs, versus drawdowns of 37.89% and 34.78%. Costs alone do not account for these losses. Its final short contributions are −3.46pp stocks and +0.25pp ETFs; posthoc unit contributes +2.54pp and −0.95pp respectively. Positive short contribution does not establish positive total return on every declining stock.

## Action and window definitions

Every saved development policy exposure request is nonzero in all three arms. Gaussian sign mapping offers no explicit categorical cash choice. This supports investigating cash availability as a separate hypothesis, not claiming it caused the failure or that cash will improve results. End-of-session flat positions can still arise from rejection, protection, holding limits or rounding.

`policy_action` is the original policy output; `raw_action` is the issued action after the unit transform, where applicable. Their magnitude summaries are separate. Unit requests are full ±1 before risk and execution; actual end exposure can be smaller. Action categories reconcile observed same-session P&L but cannot isolate the causal value of that action.

Each symbol's maximum drawdown window begins after its preceding equity peak and ends at its trough. Initial equity is included as a possible peak. Costs on the peak day are excluded; fees, impact and borrow after that peak through the trough are included and divided by peak equity. Adding that cost fraction to observed drawdown produces the reported noncost loss fraction; it is not a replay without costs. Group values average individual symbol windows with different dates, not a synchronized portfolio drawdown. Windows are descriptive, ex post and prohibited as model inputs or selection signals.

## Evidence and reproduction

[summary.json](summary.json) retains per-symbol windows, original policy outputs, action categories, accounting, accepted source hashes and script/module hashes. [groups.csv](groups.csv) contains compact group summaries for all six phases. Run from the repository root with retained private study artifacts:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_recurrent_actions.py --study alignment --output /tmp/alignment-risk-actions-new.json
```

The output must be a new path. Nineteen focused action/accounting/report tests pass, including known peak/trough costs, metric mismatch rejection and original-versus-issued action distinction. This audit consumes no additional Colab units: last verified project total 38.136658; last account reading 349.71 with zero assignments/rate. All six study sessions were already verified stopped. The aligned family remains failed; posthoc unit remains a one-seed reused-development result, with no promotion or prospective success.
