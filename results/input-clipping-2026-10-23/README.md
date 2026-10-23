# Fixed standardized market-input clipping comparison

Status: immutable payload, fixed training contracts and version30 Colab integration
implemented; actual local construction/checkers pass. Colab smokes remain before
training screens. No allocation/performance result. [Method](../../docs/feature-preprocessing.md),
[original preregistration](../../configs/experiments/input_clipping_preregistration.json),
[final fixed protocol](../../configs/experiments/input_clipping.json), [preflight](preflight.json).

Compare unchanged frozen normalization versus cap5 on seven standardized market
inputs. Same raw full-history111/26 data, means/scales,64-unit Gaussian recurrent
model,21 masked observations, reward/cost/action/holding/execution/100k budget.
Default zero Adam decay; no failed-L2/window/capacity combination. No clipping
of raw prices/ATR execution/rewards/baselines or portfolio state. No outcome-ranked
stocks/dates or validation-fitted preprocessing.

Candidate immutable ID:
`c895d6af1c98f1326f9f585009e8ad0b224ca13e885027bbc7a02c4ebc5a8894`.
Explicit new schema/normalizer preserve old three-field normalizer/default bundle
behavior.111 training symbols,298,946 training rows/19,058 development rows;
all raw episodes/features/bars/ATR exact, same source means/scales and availability.
Maximum transformed market absolute value5. Old default generation identityc2e6c58c…
reproduced exactly using its training-only statistics; candidate does not fit
normalization. Original frozen sourcecdb93a33… remains unchanged.

83 focused checks pass, including untrained actual recurrent bundle save/load/
frozen action/hidden-state parity, clipping immutability, schema rejection,
fixed-action reward/cost/quantity/event equality and old core/frozen contracts.
No local policy training/backpropagation/optimizer updates or provider requests.

Fixed selected+final relative2pp, stock/ETF/median−2pp and drawdown2pp gates,
conditional11/23/37 then prospective requirements retained. Failure closes cap
family without threshold/model/reward/step grids/extensions/promotion. Shared
2-unit allowance starts38.746179136005324; project150/130/account150/170 safeguards
unchanged. Last official account349.23 zero rate/assignments; prior sessions stopped.

## Additional preprocessing verification

85 focused checks pass, including rejection of standardization overflow before
clipping in both preprocessing and the feature-scale audit. The cap cannot hide
infinite intermediate values. [Legacy restoration](legacy-restoration.json)
passes156 exact current-host saved/frozen recurrent predictions on prior control
and L2 bundles. This does not resolve the previously rejected original-Colab
full-equity-path parity diagnostic. Clipped bundles require clipping-aware code;
legacy bundles retain their original preprocessing behavior. No new training or
allocation in this verification.

## Version30 integration acceptance

Dedicated protocol/plan/worker/runner/artifact/report and smoke-acceptance wiring
implemented.156 focused tests pass, including candidate preprocessing rejection,
legacy regressions and account floors. [Construction](construction.json) passes
both actual immutable-dataset checkers/full64 forwards and exact pinned starting
tensors/RNG without learning. Two512 Colab smokes and measured runtime acceptance
remain before100k. Official preallocation usage349.23, rate0, assignments0;
original project38.74617914/shared2-unit baseline unchanged.

## Actual Colab smokes accepted

Both512-transition CPU runs from frozenac497107a444ec5e1c8771bc01362092c34f56ae
completed on defaultPython3.13.16.156 exact frozen action/hidden/cell/reset/repeat
checks, immutable normalization/schema/feed rejection, actual saved models/Adam,
pinned common initial tensors/RNG and identical measured symbol/year sampling pass.
Both selected/final optimizer records restore; each run performed80 actual Adam
updates with zero decay and eps1e-5. Both owned sessions retrieved and verified
stopped. No smoke returns selected or changed settings.

Control duration85.4786s and clip5 duration151.7602s are hardware-dependent whole
training/evaluation measurements. Fixed conservative projection3970.4465s
(66.2minutes including1200s reserve) passes7200s. Exact accepted-smoke bytes bind
the prepared two100k screen plans. This projection is not a runtime guarantee.

Conservative project38.76724093/150, remaining111.23275907; study increment
0.02106179 including setup/idle/transfers/shutdown versus rounded measured account
debit0.01. Official account349.22, zero rate/assignments. Original study baseline
38.746179136005324/shared2-unit allowance and all project/account reserves retained.
Full models/checkpoints/reports/transports and original data/source/ledger retained.
Next: same-source fixed100k paired screen; no performance result/promotion yet.

[Smoke acceptance](smoke-summary.json), [frozen checks](smoke-restoration.json),
[runtime approval](runtime-profile.json), [metering and cleanup](smoke-compute.json).
