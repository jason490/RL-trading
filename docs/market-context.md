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
