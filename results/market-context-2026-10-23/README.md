# Compact market-context coverage audit

Status: fixed cached-data coverage/availability preflight passed. No new dataset,
normalization fit, policy inference, model training, validation-performance score
or Colab allocation. [Protocol](../../configs/experiments/market_context_audit.json),
[method](../../docs/market-context.md), [complete evidence](coverage.json).

| Existing split | Rows checked | Context available | Unavailable |
| --- | ---: | ---: | ---: |
| Training,111 symbols |298,946|298,946|0|
| Development,26 symbols |19,058|19,058|0|

Each original source row matches its cached raw bar exactly. All contributing
own/SPY windows contain the same21 sessions, avoid known gaps, and are available
by that decision bar's recorded availability. Covered source calendars are
verified with no missing sessions. The immutable raw snapshot ends before2024;
retired-test bars were not opened. Source/raw/protocol/implementation hashes are
recorded and unchanged. No row, date or stock was selected by future returns.

Three fixed candidate fields: SPY20-session price log return, its20-return
population volatility, and own-minus-SPY20-session price log return. Forward-only
split continuity; dividends excluded from these price features and retained in
simulation cash accounting. The test does not establish economic alpha or vendor
vintage correctness. Revised Yahoo/survivorship/assumed-availability limitations
remain explicit. All existing seven-field preprocessing and source payloads stay
unchanged; the audit fits nothing.

60 focused feature/core/audit checks pass, including causal prefix/future-suffix
parity, warmup, delayed publication, session mismatch, known-gap rejection and
split/dividend conventions. No new Colab units: original project38.90354117,
last official account349.10 zero rate/assignments; previous sessions stopped.

Next: separately preregister a bounded full/masked/legacy-control study, build
immutable opt-in payloads with training-only extra-field normalization and
preserved original seven statistics, validate matched rows/initializations and
frozen inference, then run Colab smokes before any research training. Coverage
acceptance alone does not select a model or authorize automatic allocation.
Closed failed feature/optimizer/distribution families remain closed. Robust-return
objective remains unachieved.

## Immutable full/masked payloads prepared

Actual write/read/reconstruction preflight passed. Masked dataset
`a392ba935e3026ab737ad539122d584242994e27a0e02bcad96d3868914913a5`;
full dataset`f439b8f4ed367573998998db30af4100bcd3cb167843efeab322b060729447db`.
Both keep every original bar/ATR/row/original feature and first-seven statistics.
Full's three added statistics fit298,946 training rows only; masked fits none,
with exact zero fields/means and identity scales. No row exclusion;19,058
development rows preserved. All normalized inputs finite for float32, masked
channels exactly zero. Source/preregistration/implementation hashes unchanged.

48 focused checks pass: source immutability, validation isolation, unsafe source/
payload rejection, explicit frozen contracts and untrained recurrent save/load/
action/hidden/cell carry/reset/caller-state/normalization parity. Direct frozen
predictions reject nonzero masked channels.156 actual prior native/squashed
saved/frozen predictions pass on current host; not original-Colab full-path parity.
[Preflight](payload-preflight.json), [legacy restoration](legacy-restoration.json).

No policy learning/optimizer update, provider request, test access or Colab
allocation. Next shared untrained initialization assets/final identity protocol
and dedicated training integration/checkers/three512 Colab smokes before100k.
Original project38.90354117/last official account349.10 unchanged. No return claim.

## Initialization assets and final identities frozen

Three untrained mapped24-input snapshots generated for11/23/37. Full/masked
complete tensors are identical; original common parameters/market/account columns
remain exact, new three columns retain the native seeded draw. Original per-seed
CPU RNG for every arm. Original assets unchanged, no dataset opened by generator,
no fitting/optimizer updates or checkpoint selection. Legacy61,513/augmented63,049
parameters; no21-versus24-input floating-point trajectory equality claim.

54 focused mapping/payload/feature/legacy checks pass, including all actual assets
and generator/source hashes. [Evidence](initialization.json),
[manifest](../../configs/initializations/market_context_v1/manifest.json),
[final protocol](../../configs/experiments/market_context.json). Canonical final
SHAf9045fe85e0e756b3397cb27d4231a82ced8645b1a41c1b20cba093160d7cbad;
preregistered gates/settings unchanged. Dedicated training/worker/artifact/report
integration, actual source/model construction and three512 Colab smokes/runtime
approval remain before100k. No new units or return result; project38.90354117.

## Version32 integration and actual construction

Training/plan/worker/runner/retrieved-artifact/report/smoke-acceptance integration
implemented. Each immutable dataset and actual full64 environment passes both
Gymnasium/SB3 checkers, finite recurrent forwards and fixed model/Adam validation.
Legacy21 versus masked/full24 inputs and common-column/RNG proofs explicit; only
full/masked complete tensors identical. [Construction](construction.json).

53 initial new contract checks,156 separate shared/legacy checks and updated121
focused checks pass (overlap across invocations). Actual masked frozen-bundle
nonzero-input rejection is mandatory for smoke acceptance, alongside234 exact
predictions/restored optimizer/models and matched sampling. Strict final reports
require both controls and all predefined gates, never retired-test data.

Three512 CPU smoke plans prepared; no Colab allocation yet. Official account349.10,
zero rate/assignments and project38.90354117 unchanged. No local policy learning
or optimizer updates. Actual three-arm smoke/runtime acceptance precedes100k.
