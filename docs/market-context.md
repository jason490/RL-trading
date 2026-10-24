# Compact broad-market context

The current recurrent source has seven inputs from the traded symbol only.
The failed sector study supplied a large proxy/rank feature set, with optional
issuer classifications. This proposed family instead has three fixed price-only
fields: SPY20-session log return, SPY20-session realized volatility, and the
traded symbol's20-session log return minus SPY's. No sector label, return-ranked
universe or future declining-stock label is needed. Benefit is unproven.

## Coverage and causal preflight

[Fixed audit protocol](../configs/experiments/market_context_audit.json) pins the
original111/26 source and cached Yahoo snapshot. Both raw windows must contain
the exact same21 sessions and no known missing session; all contributing bars
must be available by the source decision bar's recorded availability. Never carry
stale context, backfill it, or use the last historical window for earlier rows.
Use forward-only split continuity and price returns; dividend accounting stays
in the simulator. A delayed or missing benchmark row makes the context unavailable.

Audit every original train/development row and prove its source bar identity.
No training, inference, normalization fit, validation-performance scoring,
provider request or Colab allocation. The snapshot ends before2024; the retired
test remains unopened. Source coverage acceptance does not prove predictive power.
Revised Yahoo prices, assumed availability and survivor-universe limitations remain.

Only a passing audit permits a separate dataset/training preregistration, with
training-only normalization for additional fields, preserved original seven-field
statistics and matched full/masked/legacy controls. Existing failed session,
clipping, capacity, L2 and squashed-action families remain closed.

## Implemented feature and audit

`features/market/context.py` groups the available series once, applies each split
only when effective and precomputes each completed20-session window. It uses log
price differences for numerically stable log ratios. Only SPY needs volatility;
other symbols need their own momentum and matched availability/session windows.
`at` returns three fields or unavailable, never a stale substituted value. Explicit
missing-session metadata is required for every input series. Generic bar validation
rejects unsorted/duplicate/mixed-feed or invalid inputs.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
  scripts/audit_market_context.py --output artifacts/market-context/coverage.json
```

The fixed audit verifies original/raw/protocol/implementation SHA-256 identities,
source calendars and every original row. Actual results:298,946/298,946 training
rows and19,058/19,058 development rows available, with no exclusions or source
changes.60 focused checks pass. [Evidence](../results/market-context-2026-10-23/README.md).
No new preprocessing fit, dataset/model or performance result was produced.
A matched feature study remains to be specified before any Colab allocation.

## Fixed matched study preregistered

[Preregistration](../configs/experiments/market_context_preregistration.json)
fixes legacy7-feature/21-observation and masked/full10-feature/24-observation
arms. Legacy has61,513 parameters; augmented arms63,049. Full/masked share exact
untrained tensors, with original common parameters/LSTM columns copied and only
the three new columns generated without data. Every arm restores the original
per-seed CPU RNG. Shape differences prevent claiming exact legacy/augmented
floating-point trajectory equality; the same-dimension masked control is essential.

Original seven normalization statistics stay exact. Only full's extra fields fit
means/scales on training rows; masked uses exact zero extra fields and identity
scaling, with an explicit schema/builder honoring the mask during frozen inference.
No new row exclusion, input cap, sector labels, distribution/reward/optimizer change.
Default dataset/bundle behavior must remain unchanged.

Three100k seed11 runs,10k cadence; full must beat BOTH controls by2pp at selected
and final, avoid final stock/ETF deterioration and>2pp drawdown worsening, and meet
selected/final stock/ETF/median−2pp versus holding gates. Only a pass permits paired
11/23/37 then prospective evidence. Failure closes benchmark/window/subset grids
and extensions. Shared2-unit baseline38.90354116934003 and original project150/130,
account150/170,7200s limits retained. Three512 CPU Colab smokes and exact measured
runtime acceptance required first. Immutable dataset/normalizer/initialization
identities and dedicated training integration remain pending; no allocation yet.

## Immutable payload and frozen-input acceptance

`data/generation/market_context.py` appends the three fields to a deep copy of the
original source. All episode keys, bars, ATR, original features and first-seven
means/scales remain exact. Full fits only extra-field means/population scales on
298,946 training rows; validation never contributes. Masked fields/means are zero
and scales one, without fitting. No input clipping or default generator change.
The feature builder rejects unavailable full context and emits exact masked zeros.

| Arm | Immutable dataset | Normalizer SHA-256 |
| --- | --- | --- |
| Masked |`a392ba935e3026ab737ad539122d584242994e27a0e02bcad96d3868914913a5`|`f3bf1d780c675c7da397a08437f9628bb6929fab083e01663b57e31fb3b14c7d`|
| Full |`f439b8f4ed367573998998db30af4100bcd3cb167843efeab322b060729447db`|`e10423ea538967364d99a790a3265dbf9b00e3ab0c299d5f6b8eb3fe72a35081`|

Both payloads preserve298,946 training and19,058 development rows. Actual
content-addressed write/read/reconstruction checks pass; all transformed inputs
are finite in float32 and every masked extra channel is exactly zero. Original
source and preregistration/implementation hashes stayed unchanged. Full's extra
training statistics are fitted preprocessing, not policy training.

Frozen loading validates the explicit mode/schema/ordered names, original
normalizer lineage and masked identity scaling. Predictions reject any nonzero
masked extra channel, including direct normalized-observation calls. Full and
masked untrained recurrent save/load/action/hidden/cell carry/reset parity pass,
with caller state and normalization unchanged.48 focused feature/payload/frozen/
legacy checks pass (two existing Gymnasium unbounded-Box warnings).

156 actual prior native/squashed saved-versus-frozen predictions also pass on the
current host. This preserves legacy inference behavior; it does not establish
full equity-path parity with original Colab output. No local policy learning,
optimizer updates, provider requests, retired-test access or allocation.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
  scripts/prepare_market_context.py --output artifacts/market-context/payloads.json
```

[Payload evidence](../results/market-context-2026-10-23/payload-preflight.json),
[legacy restoration](../results/market-context-2026-10-23/legacy-restoration.json).
Next generate/verify the preregistered shared untrained initialization assets,
freeze the final protocol with identities and wire model/checker/Colab acceptance.
Three512 Colab smokes still required before100k. No new return result.

## Shared untrained initialization and final protocol

`training/market_initialization.py` maps the pinned original64-unit policy into
24 inputs. Original feature columns0:7 stay in0:7; portfolio columns7:21 move to
10:24. Actor and critic LSTM columns7:10 retain the native24-input seeded draw.
All other tensors copy exactly. Full/masked use the same complete mapped tensors;
legacy retains its original21-input artifact. All three restore the original
per-seed CPU RNG. The generator checks zero steps/empty Adam, finite matching
schema and exact common/new columns, writes immutable NPZ and verifies readback.

Three mapped snapshots, seeds11/23/37, are pinned in
[the initialization manifest](../configs/initializations/market_context_v1/manifest.json).
Smoke uses the same full64 tensors as the screen. No raw dataset, provider,
trained checkpoint, performance score or optimizer update was used. Shape changes
can alter floating-point summation; no exact21/24-input trajectory parity is claimed.

[Final protocol](../configs/experiments/market_context.json), canonical SHA-256
`f9045fe85e0e756b3397cb27d4231a82ced8645b1a41c1b20cba093160d7cbad`,
binds the unchanged preregistration/gates/settings, all dataset/normalizer IDs
and the new manifest raw SHA-256
`a417c68fd42a3cb69acd78124418d25bb0c369e18f08c8764bb49ca160481dc8`.
Legacy61,513 versus matched63,049 parameters; additional capacity is explicitly
controlled, not attributed to the features alone.

54 focused mapping/payload/feature/legacy protocol checks pass, including all
actual seed assets and original generator/source hashes. Mapping corruption in
old/new/account columns or common parameters rejects; readback into a different
seed fresh model restores exact tensors/RNG without optimizer state. The earlier
native-draw fingerprint is captured before any common-column mapping.
[Evidence](../results/market-context-2026-10-23/initialization.json).

Dedicated training/worker/artifact/report wiring, actual immutable-dataset model
checkers and three512 Colab smokes/runtime acceptance still precede100k. No new
allocation or return result; project38.90354117 unchanged. All failed families
remain closed and robust-return objective unachieved.

## Version32 Colab integration acceptance

`training/market_context.py` validates the final protocol, each immutable payload/
normalizer/schema, mapped initial tensors and actual optimizer/model records.
Legacy21 versus masked/full24 inputs are explicit. Initialization asset validation
rechecks the original-to-augmented column mapping and common per-seed RNG; only
full/masked entire tensors match. Each run retains its exact NPZ snapshot and
initial/selected/final records, execution counts and measured year sampling.

Version32 plans bind schema/names/statistics/data and enforce original account
150/170/project150/130 reserves. Worker and retrieved artifact checks reject
mismatched identities, normalization, model dimensions, checkpoint evidence or
sampling. Strict reports restore actual models/Adam and compare full against
BOTH legacy and masked with unchanged holding paths and fixed gates. Long plans
require exact accepted smoke bytes and the three-arm runtime projection.

Actual immutable datasets pass all three Gymnasium/SB3 checkers/full64 recurrent
forwards.53 initial new protocol/artifact/report checks,156 separate shared and
legacy checks, and an updated121-check focused suite pass (overlap between runs).
The frozen verifier now checks nonzero masked-market rejection on the actual
bundle; smoke acceptance requires that proof,234 exact sequence/reset predictions,
saved model/Adam restoration, mapped tensors/RNG and identical actual sampling.
[Construction](../results/market-context-2026-10-23/construction.json).

Three512 CPU smoke tasks prepared; no policy learning/optimizer updates locally.
Official preallocation account349.10, zero rate/assignments, project38.90354117.
Original2-unit baseline and all reserves/deadlines retained. Commit/freeze and
actual Colab smoke acceptance still required before100k.


### Actual market-context Colab smokes accepted — 2026-10-24 UTC

Three512 CPU/defaultPython3.13.16 runs from clean88fb98c… complete.234 exact frozen
sequence/reset predictions and actual masked nonzero-input rejection pass; saved
models/Adam, mapped initial tensors/RNG and actual symbol/year sampling restore.
80 actual updates/40 nominal epochs each, legacy61,513/augmented63,049 parameters.
All three sessions retrieved/stopped. Durations138.0922/138.7362/136.6898s give
fixed3406.0216s projection including1200s reserve, below7200s. Exact accepted bytes
bind prepared three100k seed11 plans, not allocated. Project38.93466769/account
349.08 zero rate/assignments; original2-unit allowance and project/account limits
unchanged. No smoke-return tuning or retired-test access. [Evidence](../results/market-context-2026-10-23/README.md).
