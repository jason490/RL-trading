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
contract. At preregistration no policy had been constructed. Untrained serialization
fixtures are now covered below; no policy has been optimized or scored.
No Colab session is allocated, no original normalization is refit and no source
2024+ price is opened. Dataset augmentation, initializer generation, frozen
contracts, task integration and actual smokes are next.


## Payload/frozen-schema implementation milestone

Full/masked immutable builders, exact source/preprocessing reconstruction and
explicit frozen VIX contracts are implemented.60 focused checks pass, including
actual untrained23-input62,537-parameter recurrent save/load/state parity and
masked-observation rejection. Source rows/ATR/splits/seven original fields and
statistics remain exact; only full's two new fields are training-fitted.
False source/feed/bounds/vintage/mask declarations reject. No local optimizer
step, provider request or Colab session.

Next run the bounded local payload producer, retain generated data identities
and verify all float32 inputs; then implement pinned zero-column mapped
initializers and metered runtime acceptance before actual Colab smokes. These
untrained serialization fixtures are not shared-initializer/learned acceptance.


## Actual payload preparation

[Payload evidence](payloads.json) verifies masked/full immutable data identities,
original episode/bar/ATR/feature/statistic preservation, exact normalized masks
and finite float32 values. Full's extra preprocessing uses298,946 training rows,
masked's uses0; development never enters the fit. The worker completed70.04sec
and exited0. Independent coordinate/statistics verification is recorded separately.
No policy optimization, provider request or Colab unit occurred.

Zero-column shared initializer code is implemented and passes11/23/37 owner,
tensor, shape, seeded-stream and rejection checks. Actual immutable initializer
production, binding and learned/runtime acceptance are still pending.


[Independent payload verification](payload-verification.json) passes: all source
episode fields and first seven coordinates are exact; pandas-derived extra
coordinate maximum error1.11e−16, NumPy training mean/scale errors6.74e−13/3.64e−14
within fixed1e−10 tolerance. All producer-protected source hashes remain exact.
Five untrained mapping checks verify all three seeds, parameter ownership,
seeded streams, rejection before mutation and initial prediction invariance
to the zeroed context columns. Actual initializer generation remains next.


## Actual untrained initialization and binding accepted

[Initializer verification](initializers.json) accepts all12 entries across
legacy/masked/full, full11/23/37 and smoke11. Three new23-input zero-column
archives and the original legacy archives are pinned in tracked configs;
full/masked share exact complete files. Common tensors/account columns and
original per-seed CPU RNG remain exact. NumPy/Python seeded fingerprints,
empty-optimizer construction and61,513/62,537 geometry are verified. Generation
used clean3e01284 source and no dataset/provider/optimizer steps.

The [resolved binding](../../configs/experiments/vix_context_binding.json) pins
these assets and all payload identities/preprocessing contracts. Its loader
rejects modified data bindings, generator-code drift, unregistered seeds/phases
and nonportable paths, then rechecks actual archives/proofs without constructing
a model.70 focused regression checks pass.

No learned optimizer validation, runtime integration, Colab smoke, new training
return or promotion is claimed. Explicit family dispatch is next because the
legacy data/parameters overlap older controls; preserve their original behavior.
Then integrate task/archive/recurrent/report/runtime checks, inspect actual Colab
Python, meter against the original shared ledger and only run the512-transition
smokes before considering the fixed100k screen. No new source prices, retired
rows or Colab units occurred in this milestone.


## Training-runner contract integration

The runner now requires explicit VIX protocol dispatch for augmented payloads,
validates registered dataset/preprocessing and the complete fixed request before
constructing an environment or model, restores the pinned shared initialization,
and retains its checksummed archive with initial/selected/final model records.
Records include actual policy and Adam-state fingerprints, coupled-Adam semantics,
seed and descriptive global sampling-state hashes. Initial Torch/NumPy/Python
states must match their pins; full/masked initial records match exactly. Loading
a trained SB3 checkpoint does not promise global RNG restoration.

90 focused checks pass, including wrong-protocol/request rejection before model
construction, all-arm initial-state agreement, real recurrent save/load prediction
parity and manually populated Adam serialization. Fixtures perform no learning or
optimizer steps and do not establish learned Colab acceptance. New module/tests
pass Ruff. Existing market, TQC and frozen recurrent regressions remain passing.

Version34 job/fleet/archive integration, physical learned-artifact acceptance and
bounded runtime projection remain before actual512-transition Colab smokes.
No session was allocated or new trading return produced by this milestone.


## Metered job and archive integration

Version34 fixed CPU plans, preallocation/remote payload checks, original-ledger
fleet wiring, a persistent two-unit VIX baseline and smoke-backed deadline
projection are implemented. Archive validation checks complete model/Adam records,
input initializer bytes, selected/final checkpoint identity, execution and
training-year sampling sidecars, all26 development symbols and frozen weights,
normalization/source lineage. Full plans require actual same-protocol accepted
smokes; no screen is authorized by static fixtures.

118 focused VIX runner/job/archive, existing critic/market archive and shared
fleet/budget checks pass. They include retry-baseline preservation, projected
study-overrun stop, payload packaging and semantic tampering rejection. New
modules/scripts/tests pass Ruff. All fixtures are non-optimizing; no Colab
allocation, new provider request or trading return occurred. Physical learned
recurrent/optimizer/frozen restoration and actual smoke acceptance remain next.


## Physical restoration and smoke acceptance implemented

Latest/selected/frozen recurrent and Adam-state restoration,78 causal sequence
predictions per arm, reset/repeat/caller-state checks, mask/schema rejection and
immutable model/preprocessing/data/files are implemented. Caller global RNG is
restored on success and failure. The smoke acceptance script also revalidates
archives, clean source, exact owned session/task mapping and development report
accounting before writing runtime-gating evidence.

93 combined VIX/source/frozen regression checks pass; two further acceptance
boundary checks reject incomplete or missing owned runs before data access.
Real model files with manually populated Adam fixtures establish serialization
only; no optimizer step, actual learned smoke, new return or allocation is claimed.
Next export the clean source, recheck official account usage, inspect each actual
Colab kernel Python and run the three bounded512-transition CPU smokes.


## Actual Colab smoke acceptance complete

[Actual acceptance](smokes.json) verifies three512-transition CPU runs from clean
33dbb56 source, each default Colab Python3.13.16 checked before dependency sync.
All actual initial tensors/Torch/NumPy/Python starts match their pins; full/masked
initial records are exact. Actual latest/selected/frozen recurrent and learned Adam
state restore;234 frozen sequence predictions match exactly, with reset/repeat,
caller-state, mask/schema, normalization/data/file and development accounting
checks passing. No local optimizer step or retired-test row was used.

Training/evaluation durations: legacy134.65s, masked144.89s, full127.00s. The fixed
conservative full-screen projection is3471.70s, including1200s reserve, below7200s.
This authorizes preparing the predefined100k seed11 screen, not confirmation,
profitability or promotion. Runtime projection remains a heuristic.

[Budget/cleanup evidence](smoke-budget.json): conservative smoke increment0.027909
units, cumulative project39.182288/150, working allowance90.817712 to130. Shared
VIX allowance remaining1.972091/2. Official post-cleanup account348.88, zero rate
and assignments; all three owned sessions are verified stopped. Original ledger
and baseline persist. Retrieved complete models, initializer snapshots, curves,
reports and source export remain under ignored `.runtime/vix-source-v1/` artifacts;
protected file hashes/paths are retained in acceptance evidence. Duplicate cleanup
waits for final study release verification. No cleanup recovery is required.

Next prepare the fixed100k three-arm CPU screen with10k curves and actual final
post-update evaluation, independently verify legacy historical reproduction, then
apply unchanged selected/final/stock/ETF/risk gates. A failure closes this fixed
family without extension or tuning; only a pass permits11/23/37 confirmation.
