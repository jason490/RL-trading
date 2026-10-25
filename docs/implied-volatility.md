# Historical implied-volatility context

This optional information family is **disabled in model inputs**. The first
source/coverage audit is complete:3,774 pre2024 observations and100% coverage of
3,433 original-dataset decision dates. No trained-model return or feature benefit
is established by coverage. It is distinct from the failed
SPY realized-price market-context and SAC/TQC method studies.

## Source and method

[Cboe historical-data documentation](https://www.cboe.com/tradable_products/vix/vix_historical_data)
publishes daily VIX closing history and distinguishes the SPX-based2003
methodology from the older OEX/VXO series. VIX is broad option-implied volatility,
not an individual stock's option surface, OI/dealer inventory or realized price
volatility. This audit starts2009, after the2003 change; later methodology and
revision/publication limitations must remain recorded.

The installed optional yfinance adapter requests only `^VIX`, inclusive2009-01-01
and exclusive2024-01-01. Raw current-vintage close/date records are stored in a
content-addressed immutable snapshot under `data/context/vix/`, separately from
executable equity bars. No trades, dividends, quantities, stock returns or broker
orders derive from these index records. A single top-level history call has a
30-second request timeout, with180-second worker bound, no feed fallback and
no automatic restart. Library metadata/session HTTP calls are not counted as
additional research-history requests. Original provider values stay local.

FRED's [official observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
has documented observation bounds and requires a registered key. Its public
graph CSV filtering is not treated as a documented date-bound guarantee here.
The full current Cboe CSV is not downloaded, preserving retired-price isolation.
No new key, dependency installation or Colab allocation is required for this
Yahoo research audit; unofficial personal-use/revised-history limitations apply.

## Lag and coverage contract

Only two fixed coordinates are prepared: log previous-session VIX close and
log(previous close /21st-prior-session close), a20-XNYS-session change. Dates
refer to exact XNYS sessions; missing exact endpoints yield uncovered decisions,
never stale filling. An assumed availability timestamp is the following XNYS
session's close plus15minutes. Actual UTC decision timestamps must reach that
availability before context is exposed. Same/future-session source values cannot
affect a prior decision. Early closes/calendar dates use exchange-calendars.

This is a deliberately conservative **availability reconstruction**, not proof
of historical publication timestamps or original vintages. A lag cannot repair
a later historical revision. The source snapshot records retrieval/revision
identity and flags historical publication vintages as unverified. Any later
research must explicitly retain that limitation and require prospective evidence.

The fixed protocol `configs/experiments/vix_source_audit.json` compares readiness
with the immutable original26-symbol train/development-only dataset. Coverage
must reach99.5% overall and in2010–14/2015–20/2021–23 eras, and99% each calendar
year. No source2024+ observations, stock-return scoring, normalization fitting,
training or data-family activation. Passing coverage is only feasibility under
the stated assumption; a separate matched experiment/protocol and acceptance are
needed before model inputs change. Failure does not loosen these thresholds.

Four offline tests validate exact lag/lookback indices, no stale filling,
same-day future perturbation isolation, UTC availability gating, invalid source
schemas and pre-network request bounds. The default pipeline remains unchanged.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/audit_vix_source.py
```

The journal `.runtime/vix-source-progress.json` preserves complete/failure status
and source recovery identity. An existing journal requires review; no implicit
retry or provider substitution. Actual audit evidence belongs in
`results/vix-source-2026-10-25/`.


## Actual coverage acceptance

[Audit evidence](../results/vix-source-2026-10-25/README.md) records100% readiness
over the original26-symbol source's3,433 unique train/development dates, all
three eras and every assessed year. Independent pandas arithmetic/source hashes
verify; dataset/preprocessing unchanged, worker exited0, no new Colab units.
Current-vintage/availability assumptions remain unverified historically; model
inputs stay disabled. Next assess the retained source against the accepted
111-symbol training dataset, then freeze a separate matched feature experiment
with unchanged controls, initial tensors/RNG, absolute gates and prospective
requirements. No performance inference, tuning extension or retired-test access.


## Expanded-consumer coverage preflight

`vix_consumer_audit.json` pins the accepted111-training-symbol/26-development
source and original seven-feature preprocessing identity, plus the retained VIX
source audit/snapshot. `scripts/audit_vix_consumer.py` makes no provider requests.
The shared alignment helper checks original feature schema, UTC availability,
split/symbol boundaries, aligned bars/features/ATR and finite baseline inputs;
it does not change rows or fit preprocessing. Readiness uses the earliest actual
decision timestamp across symbols on each session, so no later symbol timestamp
can hide an unavailable early decision. Unique-date overall/era/year gates are
unchanged; row-weighted counts and exact symbol counts are reported separately.

Six offline tests pass, including immutable consumer alignment, missing context
and split/row rejection. The worker uses a separate durable journal and180-second
deadline; existing journals reject automatic restart. Actual consumer outcome
is pending in `results/vix-source-2026-10-25/consumer.json`.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/audit_vix_consumer.py
```


## Expanded consumer passes; matched experiment fixed

[Consumer evidence](../results/vix-source-2026-10-25/consumer.json) verifies all
298,946 train/19,058 development rows (111/26symbols) ready on3,433 unique dates.
Independent pandas coordinates and source/preprocessing hashes pass. No source
request/fitting/Colab allocation; worker exited0, journal complete. Historical
vintage/publication assumptions remain explicit.

[Preregistered experiment](../results/vix-context-2026-10-25/README.md) fixes
legacy21-input, masked/full23-input native RecurrentPPO controls,100k seed11/10k
curves, two zero-initialized new LSTM columns, original common tensors/RNG,
training-only extra preprocessing and unchanged both-control/absolute/risk gates.
Legacy must reproduce fixed historical19.6245/18.4844% selected/final scores
within0.01pp mean/0.05pp per-symbol. Only passing smokes can authorize screens;
only passing screens can proceed to11/23/37/prospective.2-unit incremental
allowance inside unchanged project/account/deadline limits. Nine focused tests
pass; no new policy, dataset augmentation, training or performance result yet.


## Immutable payload and frozen-schema implementation

`data.generation.volatility_payload` builds full/masked9-field market datasets
from the accepted original7-field consumer and retained index snapshot. Both
require ready as-of context at every original decision. Deep copies preserve
equity bars, ATR, row keys, splits and original features/statistics. Only full's
two new coordinates are fitted on training rows; masked's fields/means are0
and scales1. No original coordinate is refitted, clipped or replaced. A fresh
exact reconstruction must match the complete persisted payload digest.

Explicit lineage records source/context/episode/preprocessing hashes, provider,
feed, bounded request and retrieval time,1-session lag/20-session change,
training-only fit counts and unverified publication-vintage status. FrozenPolicy
requires this contract for VIX schemas/names, rejects false declarations and
rejects nonzero masked input coordinates7:9. Existing non-VIX contracts remain
unchanged. Source VIX values remain local; no executable index is introduced.

60 focused payload/source/protocol, prior-market, TQC and recurrent verifier
checks pass. Tests cover immutable copies, development-value isolation, missing
context, future-row rejection, normalizer/lineage/feed/bounds/vintage/schema
rejection, exact masked normalization and untrained recurrent prediction/state
parity with immutable caller state.23-input native construction has62,537
parameters; no optimizer step is performed. These random untrained serialization
fixtures do not establish the future shared mapped initializer or learned smoke
acceptance.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/prepare_vix_payloads.py
```

The local producer has a600-second bound and durable per-mode progress journal,
refuses an existing journal, revalidates reloaded immutable data, checks finite
float32 inputs and preserves all source hashes. It performs only the declared
training-row preprocessing fit, no policy optimization, provider requests or
Colab allocation. Actual generated identities belong in
`results/vix-context-2026-10-25/payloads.json`; execution remains pending at this
implementation milestone.
