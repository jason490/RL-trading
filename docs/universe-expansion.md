# Matched universe expansion

## Actual result

All six 50k CPU PPO cells completed on Colab Python 3.13.16. The expanded panel
failed the declared improvement gate: mean selected/final development-validation
returns were 10.53%/5.60%, versus control 11.77%/10.49% and buy-and-hold 30.60%.
All three expanded final returns were lower. Average final drawdown was less
severe, but Sharpe and final return also fell. Six frozen bundles passed 468 exact
CPU prediction comparisons. All seven owned allocations, including smoke, were
stopped and verified. See [full results and recommendations](../results/universe-expansion-2026-10-17/README.md).
No longer run, additional stock expansion, reward search or test evaluation was
automatically added after the failed gate.

## Question and fixed scope

The 2026-10-17 experiment asks whether more training symbols improve behavior on
the original validation panel. It compares the existing 22 stocks plus four ETFs
against 55 stocks plus the same four ETFs, with five stocks in each of eleven
declared sectors. Eleven sector ETFs remain context-only; rotation features and
issuer sector inputs are disabled. The names are configurable in
[sector_expansion.json](../configs/universes/sector_expansion.json).

This is a retrospective sample of surviving companies. Adding familiar current
stocks does not establish historical membership, delisting coverage or unbiased
market representation. Sector labels are current annotations; they are never
inserted into old observations. The separate [historical evidence](historical-data.md)
milestone remains unfinished for real market data and terminal-aware training.

The [fixed protocol](../configs/experiments/universe_expansion.json) declares PPO,
CPU, seeds 11/23/37, 50,000 transitions per cell, and six cells total. PPO is the
first candidate because prior experiments had a smaller selected-to-final gap
than SAC, while recurrent PPO did not improve results. This does not establish
that PPO is profitable or superior on new data. No reward or indicator changes
are combined with the universe intervention.

## Matched data and evaluation

Both arms derive from the **same fresh immutable Yahoo snapshot**, requested from
2010-01-01 through 2023-12-31. Yahoo supplies the existing split-adjusted price and
cash-dividend convention. ABBV has an explicit 2013-01-02 history start; existing
sector ETF inception bounds remain declared. Missing sessions and failed symbols
are reported, never silently removed. The entire run stops on a coverage failure.

Training ends in 2020; validation is 2021–2023, with the existing 20-session
purging, causal features, next-session execution and simulated costs. Raw feature
rows and executable bars on the common symbols must match exactly. Each arm fits
its own immutable normalizer using only that arm's training rows; preprocessing
is part of the training-panel intervention.

Both arms select checkpoints on the **same original 26-symbol validation panel**.
The added stocks contribute training data only. No 2024 onward bars are requested
for this comparison, no test episodes are created, and the retired test interval
is never used for selection or a new performance claim. Equal averages of separate
stock accounts are not returns of a simultaneously allocated portfolio.

The generator accepts `evaluation_symbols` and `include_test=False` for this
explicit panel. Defaults preserve the original three-split behavior. These options
do not add historical IPO/delisting episodes, membership exits or terminal accounting.

## Run and recover

Use the existing Python environment and official Colab CLI installation. The
remote bootstrap checks the actual kernel Python **before dependency setup and
training** and stops on a changed interpreter family. Never install an older
interpreter to bypass that check.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_universe_expansion.py
PYTHONPATH=src .venv/bin/python scripts/build_universe_jobs.py --smoke \
  --output .runtime/universe-expansion-smoke-jobs.json
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.fleet \
  --jobs .runtime/universe-expansion-smoke-jobs.json --concurrency 1 \
  --cli-python /path/to/existing-colab-cli/python \
  --state-output .runtime/universe-expansion-smoke-state.json
```

After smoke acceptance and verified cleanup, commit and push the reviewed source,
data identities and protocol before the fixed research allocation:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_universe_jobs.py \
  --output .runtime/universe-expansion-research-jobs.json
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.fleet \
  --jobs .runtime/universe-expansion-research-jobs.json --concurrency 2 \
  --cli-python /path/to/existing-colab-cli/python \
  --state-output .runtime/universe-expansion-research-state.json
```

All allocations use the existing cumulative ledger, 150-unit project ceiling,
130-unit teardown threshold, aggregate minute metering, two-hour allocation
deadlines and owned-session cleanup. Follow [pool recovery](colab-concurrency.md)
for any failed cleanup. Credentials remain local: the explicit upload allowlist
does not include `.env`. Colab receives only versioned local datasets, never
fetches brokerage data during training, and never submits orders.

## Reports and decision rule

Keep the original ten-thousand-step learning curves and their incremental
percentage-point changes, each selected checkpoint, post-update final weights,
frozen bundle and simulator reports. Compare paired seeds, mean selected and final
net returns, drawdowns, turnover, costs and buy-and-hold. The declared improvement
gate requires at least two percentage points of gain at both selected and final
weights, final improvement in at least two seeds, and at most two percentage
points of additional average drawdown. A failure ends this six-cell comparison;
it does not automatically authorize longer runs or a new reward search.

After retrieval, run `PYTHONPATH=src .venv/bin/python
scripts/verify_frozen_runs.py ARTIFACT_ROOT... --output artifacts/STUDY/frozen-checks.json`.
For stateless PPO/SAC/TD3 bundles this compares selected and frozen weights,
checks exact deterministic predictions over three observations from every
validation episode, verifies immutable normalization and rejects incompatible
schemas/feeds/nonfinite inputs. No training or test evaluation occurs.
Use `scripts/report_universe.py ARTIFACT_ROOT... --output artifacts/STUDY/comparison.html`
with the same `PYTHONPATH=src .venv/bin/python` prefix for the interactive matched
report, compact JSON summary and per-seed CSV.

## Alpaca remains a separate feed audit

The account's free historical SIP access works, and its corporate-action endpoint
responds. Access alone does not prove complete dividends, split-unit consistency,
terminal payouts or historical membership. Audit 2016–2023 coverage separately
before replacing a dividend-aware dataset. Avoid changing feed and stock count
at once: any later Alpaca training experiment needs matched controls on the same
date range and validated action evidence. Use one downloader process, stay below
the user's 200 requests/minute, and keep all raw snapshots immutable.
