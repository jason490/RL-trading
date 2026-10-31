# Fixed feasible-reference diagnosis completed

Three predeclared unconditional requests were replayed only on fitted2018–2020
history. No learned policy, simulation rule, checkpoint or promotion changed.
No fitting, learned model loading, provider request or Colab allocation occurred.
These references characterize particular feasible paths, not an optimal return
bound or evidence that holding is impossible to outperform.

## Results

Net total returns averaged across separate-symbol episodes, not annualized or a
combined portfolio. Same original22 stocks/four ETFs,2017-12-29 anchor and
2018-01-02–2020-12-02 executions as the saved-policy fitted diagnosis.

| Reference | Stocks22 | ETFs4 | All26 |
| --- | ---: | ---: | ---: |
| Fixed long request |43.85%|40.14%|43.28%|
| Fixed short request |−45.85%|−44.23%|−45.60%|
| Fixed cash request |0%|0%|0%|
| Matched holding |58.15%|47.15%|56.45%|

The long reference trails holding14.30pp on stocks and7.00pp on ETFs. Its
saved-path accounting separates2.74/2.79pp incremental cost drag from11.56/4.21pp
non-cost gaps. This add-back identity is not a zero-cost replay or causal
attribution to a specific rule. All protective exits, partial targets, re-entry,
holding-limit exits, position reconciliation and gaps remain active.

For context, the accepted fitted replays show SAC final stocks62.78% and GRU final
ETFs61.28%, each above this long reference in that group. Thus this audit neither
proves the return target infeasible nor explains all learned-policy weaknesses as
mechanical friction. Neither learner was comparable across both groups/median
stocks, and their development gates remain failed. No reference is promoted.

## Short-side observation

On the two ex-post negative-holding stocks, fixed short returns+13.61% on CVX
versus holding−17.07%, and−2.95% on XOM versus holding−40.25%. XOM's recorded
short-side costs are10.77% of initial capital. Inventory, protective exits,
resizing, dividends/borrow and path order make short performance different from
simply negating a stock's terminal holding return. This does not identify a sole
cause of that particular loss.

The short reference is poor across the broader rising panel; these two outcomes
are reporting-only. No directions were chosen from future decline labels. The
long and cash references also ran on every symbol. The universal request rule
is fixed before replay and independent of observations/outcomes.

## Contract and verification

Source committed as `13b5df2`; [protocol](../../configs/experiments/feasible_references.json)
SHA256 `67248a9cfa7c99852bdf4a88fe6d7c7584ab14a1ef3b6537b20df20d19d21051`.
Actions are `[1,1,1,1]`, `[-1,1,1,1]`, `[0,1,1,1]`: active5 ATR stop,
8 ATR near target and4 ATR width for nonflat positions. They are manually
specified diagnostic actions, not replacements for learned protection outputs.
Entries can be clipped/rejected and inventory can be closed by existing rules.
Default fees, impact,3% annual borrow, whole shares,100% cap and20-session limit
are unchanged. Cash has zero fills/costs/net return.

35 focused checks and Ruff passed before replay, including actual holding-limit
exits. All three actual replays completed26 episodes/19,136 rows each:
78 episodes and57,408 execution rows. Every row retains the same prescribed
request. All12 protected files, dataset/normalizer, shared ledger and failed-family
closure remain unchanged. Holding-path identity matches all four fitted-policy
replays. Costs, signed P&L and ending equity reconcile. No development/test replay,
new direction selection, fitting or family reopening.

[summary.json](summary.json) retains group/individual-stock excess, cost categories,
participation, decline diagnostics, exact defaults and input/report digests.
Full paths and three interactive chart pages remain under
`artifacts/feasible-references-2026-10-31/{long,short,cash}.{json,html}` outside Git.
No learned weights were opened or duplicated. All outcomes, including losses,
are retained; no redundant upload or download archives were created.

## Decision

The benchmark gap reflects both executed-path differences and costs, so neither
adding a small turnover penalty nor blindly shorting eventual losers is justified
by these results. Preserve the simulator, frozen failures and retired2024+ data.
The next step is a primary-source and existing-implementation review for a
distinct, bounded learned hypothesis, with training-only justification before
another development comparison. Do not run an open-ended parameter search or
reopen a closed family under a new name.

Project ledger remains **39.92585919/150**, with no new Colab allocation or learning.
The broader improvement goal remains active; no robust/prospective winner exists.
