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


## Fixed three-arm numeric protocol and shared starts

[Protocol](../configs/experiments/session_context.json) SHA-256
`b5fdd94552db62142aab2c08b3a052133e51c9983d17099e64c6995a21bdf466`
binds the actual three datasets, normalization,100k seed11 screen,10k reporting,
all original relative/absolute/risk gates and conditional11/23/37 confirmation.
Full must pass against both legacy and masked; failed seed11 closes this fixed
feature family with no window/feature/step/reward grid or promotion.

All arms use native Gaussian recurrent PPO, target_klNone, adaptive train/unit-sign
evaluation, masked account/execution observation contract, unchanged net-log
reward/simulation/cyclic sampling. The update-guard family stays closed. Three
CPU smokes must pass before three100k tasks. Shared incremental allowance2 units
starts at project38.39201564833781 across all attempts; original150/130 project,
150/170 account and7200-second per-allocation inclusive deadlines remain.

[Shared initialization](../configs/initializations/session_context_v1/README.md)
adds four untrained25-field files and reuses four existing pinned21-field starts.
No data/learning/performance selection. Masked/full tensor/RNG/settings equality
is enforced; legacy is a different-capacity control. Actual fresh constructions
for all three arms × smoke11/screen11/23/37 restore exact fingerprints and RNG.
Actual save/load retains provenance; corrupt size/guard/step/hash rejection passes.
31 focused protocol/session/shared-start tests pass. No local optimizer updates.

Controller/worker/archive integration, real Gymnasium/SB3 checkers and actual
Colab smokes remain before allocation. No training or performance result yet;
project38.39201565 unchanged and all owned sessions stopped.


## Colab integration and actual construction checks

Version26 plans bind exact arm dataset/schema/feature count, initialization
artifact, source protocol and original account floor. Controller and worker
validate payloads and starts before fitting; runner retains initialization
snapshots and actual initial/selected/final optimizer/model records. Common
retrieval checks remain shared; study-specific validation lives in
`runtime/colab/session_artifacts.py`. Shared read-only recurrent evidence is
extracted into `training/recurrent_record.py`, preserving the legacy guard API.

76 focused protocol/artifact/shared-start/account-floor checks pass, including
all three smoke/screen arms, wrong schema/dimension/preprocessing, missing
checkpoint/snapshot and corrupt RNG evidence. Scripts compile. Actual immutable
three-task payload validation passes. [Construction](../results/session-context-2026-10-22/construction.json)
records real Gymnasium/SB3 checker acceptance and untrained recurrent forwards
with21/25/25 observations; no learning or allocation.

Commands prepared: `build_session_jobs.py`, `accept_session_smokes.py` and
`report_session_context.py`. Actual three512 Colab CPU smokes still required,
including234 exact frozen action/state/reset predictions, same-source/sampling
and masked/full initialization equality, durable retrieval and verified shutdown.
Official usage/source export/default Python checks precede allocation. No screen
before smoke acceptance; original study baseline38.39201564833781/2-unit allowance
and project/account protections remain. No new credits used.


## Actual session-context smokes accepted

Three512 Colab CPU runs from cleanb1cfa00/defaultPython3.13.16 completed and
retrieved; all owned sessions verified stopped.234 exact frozen recurrent
predictions, normalization/rejection checks, actual selected/final optimizer
restoration and checkpoint identities pass. Each has80 Adam steps/40 nominal
epochs. Masked/full initial tensors/RNG/settings match exactly; actual sampling
agrees across all arms. One CLI reply timeout recovered from completed export,
without training restart. Fixed same-source three100kseed11 screen next; no
tuning from smoke returns. Project38.42515390/150, study increment0.03313825;
official account349.48 zero rate/assignments. Original2-unit study baseline
38.39201564833781 and all project/account protections retained.
[Evidence](../results/session-context-2026-10-22/README.md).

Report identity fix: each archive is validated against its actual arm dataset,
including retained-extraction fallback. Six regression cases cover all three
arms and both transport paths;27 focused report/artifact checks pass. This fixes
a report-only single-dataset assumption; recorded training/protocol unchanged.

The report now also passes the arm explicitly into immutable-payload validation.
The six transport/arm regression cases exercise archive and payload binding
together;27 focused report/artifact checks pass. Training evidence unchanged.


## Completed session-context screen — gates failed

Legacy selected/final19.62/18.48%, masked27.66/10.76%, full26.42/16.08%,
holding30.60%. Full fails both relative control gates and both absolute phases;
final stock/ETF16.56/13.45% versus holding32.63/19.47%. Joint short/total
reporting-only cohort2/3 selected→1/3 final. No feature/window/step/reward sweep,
confirmation or promotion; fixed family closed.

Strict immutable source/accounting/holding and actual optimizer restoration pass;
full/masked exact initialization and sampling match. Final20,000 Adam steps,
61,513 legacy/63,561 augmented trainable parameters.234 screen frozen predictions,
all six owned sessions stopped; legacy tensors/sampling reproduce prior control.
Project38.51487741/150, study increment0.12286176; official account349.41 zero
rate/assignments. Source/ledger/data retained. Publication/verified duplicate
cleanup next. [Full evidence](../results/session-context-2026-10-22/README.md).


Session-context publication complete:17 server assets verified, six accepted
model bundles and interactive reports public. Only two duplicate publication
archives removed (49,579,965 bytes);178 retained hashes unchanged. Strict summary
reproduces byte for byte, all three screen models/accounting/gates rechecked.
All six sessions stopped; project38.51487741 unchanged. Fixed family closed;
next distinct preregistered learning hypothesis, no extension/feature grid.
[Evidence](../results/session-context-2026-10-22/README.md).
