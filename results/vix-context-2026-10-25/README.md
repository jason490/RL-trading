# VIX context comparison — preregistered, not trained

[Fixed protocol](../../configs/experiments/vix_context_preregistration.json)
SHA256`c9972c2ce85d0d9bad54501f1634357919217e5aee8b4b964d197f8dd6ea6008`
uses the accepted111-symbol training dataset and original26-symbol repeated
development panel. The [source and consumer audit](../vix-source-2026-10-25/README.md)
passes; historical publication/vintages and point-in-time universe membership
remain unverified. These limits must accompany any eventual performance result.

## One fixed information hypothesis

Two lagged broad option-implied volatility coordinates (log prior-session VIX
level and20-session log change) may condition policy decisions differently from
own-price or realized broad-market-price indicators. No improvement is assumed.
This is a new information ablation, not a rescue of failed critic, clipping,
reward, capacity or price-market-context families.

| Arm | Market fields | Total inputs | VIX information |
| --- | ---: | ---: | --- |
| Legacy |7|21|Absent|
| Masked |9|23|Exact zero fields, identity extra preprocessing|
| Full |9|23|Fixed lagged VIX coordinates, training-only extra preprocessing|

All arms retain native Gaussian RecurrentPPO,64-wide MLP/LSTM, same reward,
costs, simulation, sampler,100k seed11 steps,10k curves and selected/final
evaluation. Training exposure is unrestricted; evaluation uses the established
unit-magnitude exposure contract in all arms, exactly matching the historical
control. This is not a change to default simulation or a new sizing hypothesis.
The original seven preprocessing coordinates, raw bars, ATR, row keys and splits
must remain exact; only the two new full coordinates are training-fitted.

The original full64 control initializer/CPU RNG is pinned for11/23/37. For both
23-input arms, copy every common policy tensor and original input column, and
set the two new actor/critic LSTM input columns to zero. Both augmented policies
and empty optimizers must match exactly; verify NumPy/Python seeded RNG equality
before learning. Expected counts are61,513 legacy and62,537 augmented parameters, to be checked
during untrained initializer acceptance. No
bitwise21-versus23 floating-point trajectory equivalence is promised.

## Gates and bounded operation

The candidate must beat **both** fresh controls by2pp at selected and final
checkpoints, avoid final stock/ETF deterioration and more than2pp extra mean
drawdown. Both phases must meet stock/ETF holding excess and median-stock excess
at least−2pp. Legacy must reproduce its registered historical checkpoints within
0.01pp mean and0.05pp selected/final per-symbol bounds; otherwise interpretation
and promotion fail without widened tolerances or automatic retry.

Failure closes this fixed two-coordinate family: no level/change/window/feed/
normalization/model/reward/step grid or extension. Only a passing screen proceeds
to fixed11/23/37 confirmation, then independently registered prospective evidence.
Future declining-stock labels remain reporting-only; positive total and positive
net short P&L are both needed for profitable-short claims.

Use Colab CPU after actual kernel-Python inspection. Three512-transition smokes
must establish mapped initialization/RNG, masks, learned Adam/recurrent/frozen
parity (at least78 predictions per arm), reports and a conservative deadline
projection before screens. Study allowance2 units across all retries/controllers,
inside original cumulative150/130 and account150/170 safeguards. Preserve the
original shared ledger/lock and a persistent VIX study baseline, aggregate minute
metering, inclusive7200-second allocations and verified owned cleanup.

## Actual progress

Nine focused tests pass: source bounds/availability/lagging, immutable consumer
alignment and changed-protocol rejection. Both full and smoke parameter
dictionaries exactly equal the original native recurrent control's resolved
contract. No policy has been constructed, optimized or scored for this family.
No Colab session is allocated, no original normalization is refit and no source
2024+ price is opened. Dataset augmentation, initializer generation, frozen
contracts, task integration and actual smokes are next.
