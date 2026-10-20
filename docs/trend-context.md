# Longer-horizon price context

The seven-feature baseline includes one-session returns,20-session volatility,
EMA20/50 ratios,RSI14,ATR14 and relative volume20. EMAs retain exponentially
decaying history; the baseline does not explicitly expose annual/quarterly price
momentum or distance below a trailing annual closing high. This is a possible
information limitation, not a demonstrated explanation of failed policies.

## Feature definitions and availability

`features/trend/context.py` defines three optional inputs at completed session t:

| Input | Definition |
| --- | --- |
| `price_log_return_63` | log(close[t]/close[t-63]) |
| `price_log_return_252` | log(close[t]/close[t-252]) |
| `price_drawdown_252` | close[t]/max(close[t-251:t+1])-1 |

Prices use the assembler's forward-only effective split continuity. Cash
dividends are excluded from these price signals; simulator equity still accounts
for dividends and short dividend obligations. Values are scale-invariant. The
current close is known only at the bar's declared availability time; actions
continue to execute next session. The trailing high includes the current close,
uses no future observations and is a closing-price high, not an intraday high.

All modes other than `none` require253 contiguous observed sessions. Known data
gaps reset lookbacks; no filling or stitching across a gap is permitted. Earlier
training bars may warm validation features causally, but features do not cross
purged execution boundaries as trades. Preprocessing fits training rows only.
No future declining-stock label, future return, cross-symbol rank or fitted
threshold enters these inputs.

## Matched information controls

| Mode | Market inputs | Warmup | Purpose |
| --- | ---: | ---: | --- |
| `none` | original7 | unchanged | Default; existing dataset identities remain unchanged |
| `matched_baseline` | original7 |253 observations | Baseline with the same eligible rows as context arms |
| `masked` | original7 plus3 zeros |253 observations | Same dimension and model capacity as full context |
| `full` | original7 plus3 actual values |253 observations | Causal context treatment |

Masked and full datasets have identical bar/ATR panels and first7 normalizer
statistics. Only full context exposes the extra information; masked columns stay
exactly zero after training-fitted normalization. Legacy versus augmented model
input sizes differ, so report both controls before attributing a gain to context.
The warmup changes training eligibility and its first7 normalization relative to
the older short-warmup dataset. Do not claim an old frozen model is an equivalent
matched control merely because its validation dates agree.

Dataset schemas and manifests bind mode, lookbacks, price convention, readiness,
mask and availability. A full-context frozen policy requires its actual training
normalizer and ordered10-feature schema. Old bundles keep their original input
contract; this feature does not modify existing models or make inference train.

## Build and inspect

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local dataset \
  data/raw/universe-expansion/SNAPSHOT_ID --output data/datasets \
  --universe configs/universes/sector_research.json --development-only \
  --trend-context full

PYTHONPATH=src .venv/bin/python scripts/preflight_trend_context.py \
  --output artifacts/trend-context/preflight.json
```

The fixed preflight uses the retained2010–2023 snapshot and original development
panel. It rejects a different source identity, regenerates the default dataset
to verify byte identity, builds all3 controls/treatment, verifies identical
training/validation bars and ATR across arms, checks original validation dates,
and records feature coverage on training data only. It does not train or
allocate Colab, and does not load the retired2024–September2026 test interval.

## Evidence required before adoption

Tests verify exact formulas, rolling-high expiry,253-session readiness, prefix
invariance, future perturbation isolation, scale/split continuity, masks, shared
panels, training-only fit and gap resets. These establish data correctness, not
predictive value. Before allocation, commit a numeric protocol with dataset
identities, matched native full-state PPO controls, unchanged log-equity reward,
actions/costs/starts/discount, seeds, budgets and selected/final gates. Run CPU
smokes first; a seed11 screen must beat both controls before seeds23/37. Keep
stock/ETF absolute gates, joint total/short reporting and independent prospective
evidence. No model has yet been trained using this feature family.

## Current preflight evidence

[Prepared study](../results/trend-context-2026-10-20/README.md) records dataset
identities and train-only coverage. All3 datasets have64,922 training rows and
19,058 validation rows. Source default identity is unchanged; validation dates,
matched raw panels and first7 normalization agree. Actual Gymnasium/SB3 checkers
pass with21/24/24 observation fields, and masked/full initial tensors match
exactly. The [numeric protocol](../configs/experiments/trend_context.json) is
fixed before allocation. No profitability or training result is available yet.

## Colab jobs and matched reporting

Version12 plans bind each arm to its immutable dataset identity, exact feature
order/count/schema, mode and development-only manifest. The fleet rejects a
wrong task dataset before allocation; the remote worker verifies identity and
content before training. Retrieval checks the frozen feature contract as well
as weights, curves, reward, observations, architecture, Python and hardware.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_observation_jobs.py \
  --protocol configs/experiments/trend_context.json --phase smoke \
  --output .runtime/trend-context-smoke-jobs.json
```

Run through the existing centrally metered fleet with the original cumulative
ledger and a2-unit study allowance. Use the same exact study-start ledger value
across smoke/screen/confirmation, never reset it. Freeze one committed source
for all arms. Smoke acceptance includes CPU frozen prediction parity before
50k tasks. Confirmation task preparation requires a matching passing screen.

After verified screen completion/cleanup, use `scripts/report_observations.py`
with this protocol and state path. Dataset differences are allowed only through
the fixed3-arm mapping. Source, parameters, reward, starts, costs and versions
must agree, and holding baselines must match. Every cell retains its actual
dataset ID and selected/final side attribution. Both control gates must pass.
