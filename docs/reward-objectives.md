# Reward objective research

The existing reward is `log(equity_now/equity_previous)` after simulator costs.
Checkpoint selection averages arithmetic net returns over validation episodes.
Log and arithmetic objectives can rank distributions across episodes differently.
This is a hypothesis about objective alignment, not proof of the observed losses.

## Optional initial-capital objective

`TradingEnv(..., reward_kind="net_initial_equity_change")` and
`train(..., reward_kind="net_initial_equity_change")` use
`(equity_now-equity_previous)/initial_equity`. The denominator is fixed starting
capital, never short-sale cash proceeds or changing current equity. Fees, impact,
borrow, dividends and corporate-action cash flows remain in simulator equity.
No future labels, short bonus, turnover penalty or extra reward is added.
Undiscounted rewards telescope to the episode net return, including insolvency;
log rewards retain the original -20 insolvency sentinel. Default log behavior
and all existing frozen inference contracts remain unchanged. Run metadata
records the exact kind and scale and gives the candidate a distinct variant.

PPO still uses the same discount and GAE as the control, and random finite
training episodes bootstrap at truncation. Therefore neither reward is claimed
to optimize full validation terminal return exactly. Changing discount/GAE would
be a separate ablation. Arithmetic changes may increase risky behavior: report
stock/ETF returns, drawdown and costs under unchanged gates.

## Locked comparison and operations

[Protocol](../configs/experiments/reward_objectives.json) fixes the original
26-symbol development snapshot, native full-state PPO, wide initialization,
50k transitions and seed11 control/candidate screen. Run two 512-transition CPU
Colab smokes first. Keep evaluation every10k and selected/final checkpoints.
Only a relative-gate pass permits matched seeds23/37; absolute group, median
stock and drawdown gates plus prospective evidence remain necessary. Falling
stocks must report both positive total return and positive net short P&L; these
future cohorts never select a checkpoint or supply rewards.

`runtime.colab.jobs.reward_plan(kind, seed=11, smoke=False)` creates strict
version11 jobs accepted by the existing remote worker and artifact verifier.
The centrally metered fleet must own all allocations and the existing ledger.
Two-unit incremental study cap, 130-unit working stop, 150-unit cumulative ceiling,
two-hour allocation deadlines, bounded retrieval and verified teardown apply.
No reward-study Colab allocation has occurred yet. Do not bypass smoke acceptance.

Prepare smoke tasks without allocation:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_observation_jobs.py \
  --protocol configs/experiments/reward_objectives.json --phase smoke \
  --output .runtime/reward-smoke-jobs.json
```

Use `--phase screen` only after verified remote smoke/frozen acceptance. The same
builder requires a matching passing screen report for `--phase confirm`.

## Local acceptance

Tests prove both telescoping identities and their different cross-episode
rankings, finite inputs, insolvency semantics, exact observations/execution/RNG
under matched actions, cost-aware reconciliation, and strict remote-plan rejection.
These checks do not establish profitability or replace remote save/load acceptance.

## Matched report

After completed research cells and verified fleet cleanup, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_observations.py \
  --protocol configs/experiments/reward_objectives.json \
  --state .runtime/reward-screen-state.json \
  --output artifacts/reward-objectives-screen/summary.json
```

The reporter varies only the declared reward kind in its matched contract;
unplanned scale, parameters, source, data, costs or observation changes fail.
It records both actual kinds per cell and checks selected/final report scores,
matching holding baselines, side attribution and post-update final weights.
Confirmation also requires the matching passing screen receipt.
