# Saved-path participation and cost diagnosis

Inspect already completed development reports before another training change.
`reporting/participation.py` reuses the existing behavior summary and reconciles
reported costs against fees, adverse impact, borrow and side-specific costs.
It checks aggregate long plus short P&L against terminal equity and recorded net
return. Existing simulation attribution supplies the per-session cash-flow audit;
this report does not replace it or reconstruct missing source prices.

## Exact accounting identity

For each symbol, after dividing dollar costs by the common initial capital:

`net excess = saved-path non-cost excess - incremental cost drag`

Incremental cost drag is policy costs minus holding costs. Non-cost excess adds
costs back to both original saved paths. This is an accounting decomposition,
**not a zero-cost backtest**. Changing costs changes equity, affordability,
observations, actions and fills, so these numbers cannot predict a free-trading
counterfactual. All reported returns still use the actual modeled costs.

## Participation and exits

Close-of-session long/short/flat fractions and signed/absolute exposure describe
terminal inventory each day. They do not capture every intraday position. Touched
fractions use the existing long/short attribution and can overlap in a session
with reversals. A flat close does not mean no trading or no short P&L.

Fill reasons distinguish discretionary add/reduce trades, reversals, protective
stop/near/far exits and the fixed20-session holding limit. Partial target fills
are not separate complete trades. Re-entry within one session of protection or
the holding limit comes from the existing behavior diagnostic; it is not proof
that a fill should have been skipped. Stops and the holding limit remain active.

Small same-side discretionary resize fills are measured using executed delta
notional divided by pretrade equity, reconstructed as event marked equity plus
fee and adverse impact. The fixed descriptive threshold is5% of equity. The
associated fee/impact sum is the cost on those historical fills, not guaranteed
savings from a rebalance rule. Explicit exits/reversals/protective fills are not
included in that count. No execution threshold or reward is changed here.

## Reproduce the bounded audit

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_participation.py \
  --state .runtime/reward-screen-state.json \
  --state .runtime/trend-context-screen-state.json \
  --output artifacts/participation-diagnostic/summary.json
```

The command accepts at most5 unique completed50k PPO runs, checks study cleanup,
reads both selected/final reports, requires matched holding panels and complete
2021–2023 paths, and preserves report checksums. No dataset, model weights,
retired-test report, provider API or Colab session is opened. Early-terminated
paths with a longer holding horizon are rejected rather than silently compared.
The five studies/arms have different training data/rewards; cross-study numbers
are descriptive, not a matched causal experiment or seed confirmation.

## Current evidence and next decision

[The completed audit](../results/participation-diagnostic-2026-10-20/README.md)
covers five runs and ten phases. Native PPO's stock gap is7.49pp, with4.85pp
incremental cost drag and2.65pp non-cost gap. Its ETF gap is10.83pp, with3.88pp
cost drag and6.95pp non-cost gap. Full-context ETFs lose6.79pp from shorts versus
0.15pp masked, while actual costs are lower. This supports a signal/participation
problem in addition to trading costs; it does not establish the causal value of
a direction filter or any new model.

Observed costs on sub5%-equity resizes are only0.05–0.12pp across these runs.
Do not prioritize a small deadband as a cure for gaps exceeding7–16pp. Forced
holding-limit exits remain a required simulation constraint, not a rule to
remove merely to improve returns. Before another RL allocation, investigate
whether a training-only supervised signal/warm-start control has predictive
value under strict chronological target availability. This is a proposed next
research direction, not an implemented or selected model. Any such experiment
needs purged labels, frozen normalization, matched controls and its own committed
protocol. Validation declining-stock outcomes remain reporting-only; their labels
must never enter supervision, rewards, sampling weights or checkpoint selection.

## Recurrent fill-cost attribution

Observed masked final stock/ETF same-side resizing costs2.13/1.51pp, entries
1.22/0.97, protective+holding exits1.69/1.26, borrow0.64/0.77. Categories reconcile
all fees/impact/borrow and executed quantities; unknown reasons/corruption reject.
Even arithmetic addback of every resize leaves4.68/3.68pp gaps; adding all entries,
resizes, closes and reversals still leaves~2.99/2.47pp. These are saved-path
bookkeeping bounds, not counterfactual profits. Resize-only savings insufficient;
required protections/20-session exits/borrow remain.18 checks pass, source hashes
unchanged, no fitting/data/Colab. Next separate direction from protective-exit
participation before a distinct protocol. [Audit](../results/recurrent-fill-costs-2026-10-21/README.md).

## Saved action/execution relationship

Masked final stock/ETF requested long magnitude74.06%/73.15% versus end-long
67.04%/67.44%; sign agrees93.93%/94.77% of sessions. Flat-after-protection
2.09%/0.92%, flat-after-holding2.84%/3.65%; rejected rows separate. These observed
relationships do not prove a causal protection bottleneck. Nine checks reconcile
action categories and net P&L with unchanged sources; no fit/data/Colab. Next
consider preregistered fixed unit-magnitude sign-preserving replay, with original
protective outputs and exact unrestricted parity gates before interpretation;
no replay yet. [Evidence](../results/recurrent-actions-2026-10-21/README.md).

## Alignment saved risk/action audit — 2026-10-22

All six accepted phases audited without fitting/data/Colab. Aligned final stock/ETF
cost drag11.08/8.37pp and noncost excess−6.68/−2.33pp both contribute to failure.
Mean symbol drawdowns−37.89%/−34.78% contain only2.87%/2.63% of peak equity in
window costs; this is descriptive bookkeeping, not a cost-free replay. All saved
policy exposure requests are nonzero; original Gaussian output and issued unit
action are now distinguished. Consider explicit categorical cash as a separate
preregistered hypothesis; no causal fix established, no allocation or promotion.
19 focused tests pass; accepted source hashes unchanged. Project38.136658 retained,
all owned sessions stopped. [Evidence](../results/alignment-risk-actions-2026-10-22/README.md).
