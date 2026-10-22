# Completed overnight and intraday context

Status: optional feature implementation verified on synthetic data; no real-data
preflight, predictive result, Colab acceptance or training yet. Default remains
the original seven market features. The failed update-guard family stays closed.

## Hypothesis

The exact saved-fill diagnosis locates substantial ETF overnight exposure gaps
and short-timing losses. This does not establish a cause or predictability.
Baseline close returns, volatility, EMA ratios, RSI, ATR and relative volume do
not explicitly separate overnight and intraday returns. A synthetic witness
changes a bar's open, preserving high/low/close/volume: original seven features
remain exactly equal, while these optional fields differ. Distinct information
does not imply useful information or a profitable policy.

## Four fixed inputs

For completed session t on forward-only split-continuous prices:

| Field | Definition |
| --- | --- |
| overnight_log_return | log(open[t]/close[t−1]) |
| intraday_log_return | log(close[t]/open[t]) |
| overnight_ema_20 | EMA20 of completed overnight returns |
| intraday_ema_20 | EMA20 of completed intraday returns |

EMA alpha is2/21, seeded with the first observed return pair; require20 pairs
(21 observed bars) before release. Existing EMA50 baseline readiness is longer,
so this family adds no eligibility loss on ordinary continuous baseline chunks.
Known gaps reset all lookbacks through the existing dataset chunk boundary.
No annual/quarterly signals or supply/demand values are added in this experiment.

All fields are released at current completed-bar availability, even though the
open occurred earlier. Decisions execute next session. No next open, future
return, future-decline label, sector snapshot or fitted threshold is used.
These are price signals excluding cash dividends; simulation still accounts for
dividends and short dividend obligations. Forward-only effective splits use the
same assembler continuity as the baseline. The two raw returns sum to the
completed close-to-close log price return.

## Controls and schemas

`none` leaves names, schema, payload and default identity unchanged. `masked`
adds four zeros with the same readiness and dimension as `full`. `full` adds
actual values. The generator fits added-column normalization on training rows
only, freezes it during validation/inference, and records mode/formulas/readiness
in the content-addressed schema/manifest. Masked normalized columns stay zero.
Original first7 features and their fitted statistics must agree across arms.

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local dataset SNAPSHOT \
 --output data/datasets --universe UNIVERSE.json --development-only \
 --session-context full
```

This generic command is not an approved historical study invocation: preflight
must bind the retained expanded111 training/original26 development panel, source
IDs, original normalization lineage and all exact episode/accounting invariants.
Never rebuild or open retired2024–September2026 data for this study.

## Before allocation

Build immutable none/masked/full controls from the retained2010–2023 source,
verify unchanged default identity, identical episode bars/ATR/eligibility and
first7 statistics, and inspect training-only feature coverage. Preserve legacy
frozen contracts. Bind all dataset IDs and shared untrained initialization
snapshots in a committed numeric protocol. Test actual Gymnasium/SB3 checkers,
worker/archive/frozen schema rejection and short train/save/load in Colab CPU.

The intended matched screen uses fresh recurrent PPO, unchanged net-log-equity
reward,20-session holding, signed/short-capable four-action contract, costs,
stock/ETF cyclic sampling and fixed100k transitions/10k checkpoints. Include
legacy control for capacity and masked control for added information. Candidate
must pass the original selected/final absolute stock/ETF/median targets, relative
+2pp and drawdown gates against every control; only then seeds11/23/37 and frozen
prospective evaluation. Exact numeric/source/data/budget protocol remains to be
prepared; no allocation, hyperparameter sweep or checkpoint promotion now.

## Verification

53 focused session/trend/core checks pass: formulas/EMA seed, prefix invariance,
scale/split continuity, exact information witness, masks, availability, shared
panels, training-only fit, gap resets and invalid mode rejection. These establish
correct implementation only. Project38.39201565/150 retained; no new Colab usage.


## Actual session-context data preflight

Retained development-only raw source verifies; default generated identity
unchanged. New masked/full datasets preserve all bars/ATR/original features and
original first7 normalization exactly. Each has298,946 training and19,058
development rows,111 training/26 development symbols. Added-column normalization
fits training rows only; future/test data unopened.53 focused checks pass.
No training/allocation; next fixed three-arm protocol and shared untrained
initialization, then Colab integration/checkers/smokes. Project38.39201565 unchanged.
[Evidence](../results/session-context-2026-10-22/README.md).
