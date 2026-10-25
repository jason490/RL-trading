# Fixed benchmark subtraction — feasibility failed

The protocol/code were committed asdef302a before execution. Reusing all52 saved
training trajectories, subtracting exactly one unit of the matched holding reward
passed only3/8 fixed variance conditions. This fixed subtraction hypothesis is
closed: no coefficient, benchmark, window or discount grid, or training allocation.

## Outcome

Ratios below compare residual to original policy variance. Gate requires<=0.90
for every cell, using equal-weight means of within-episode population variances.
A ratio above1 means subtraction increased observed variation.

| Saved phase/group | Reward variance ratio | Discounted-target variance ratio |
| --- | ---: | ---: |
| Selected stocks |1.0159|0.6412|
| Selected ETFs |1.4952|1.4020|
| Final stocks |0.8474|0.7578|
| Final ETFs |1.1792|1.1167|

Both ETF phases worsen at both levels; selected stock rewards also fail. Lower
stock target variation does not override the other conditions. Full per-symbol,
within-episode and pooled descriptive results are in [summary.json](summary.json).
No coefficient was fitted and no model, policy path, reward function or checkpoint
was changed. Trading returns remain those in the completed research studies.

## Verification and limitations

The5.22-second worker exited0 with a complete progress journal. Independent
cash/share accounting reproduces all26 benchmark paths with exactly zero final
equity/reward discrepancy, including fees, adverse impact, whole shares, splits,
fractional-share cash and dividends. Independent NumPy discount matrices,
covariances and ratios agree within1.12e−15. All source models, data, prior traces
and the original usage ledger remain unchanged. [Verification](verification.json)
checks every fixed group and all eight gates. Eight focused non-optimizing tests
pass; new code passes Ruff.

This is observed path variation, not irreducible noise, gradient variance,
expected-value accuracy or economic improvement. Targets overlap; there is one
stochastic path per symbol/phase. Subtracting the same realized benchmark from
both prediction and target leaves their prediction error unchanged. No claim of
policy invariance with endogenous termination/bootstrap is made.

Full derived benchmark rewards, summary and terminal journal remain under
`artifacts/benchmark-variance-2026-10-25/`. No model replay, optimization, provider
request, development evaluation, retired-test access, Colab allocation or broker
order occurred. No duplicate models/transports were created or require cleanup.
Project remains39.272458/150; last official account348.80 idle, floor150/teardown170
unchanged. All failed families remain closed; no robust winner is established.
