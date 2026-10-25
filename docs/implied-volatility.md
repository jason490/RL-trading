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


## Zero-column untrained initializer implementation

`training.vix_initialization` validates the pinned original full64 recurrent
manifest and per-seed tensor/RNG archives, then embeds21 inputs into23: original
market columns0:7 stay exact, account columns7:21 move to9:23, and new VIX
columns7:9 are zero in both actor and critic LSTMs. Every other policy tensor
and dtype stays exact. Parameter counts61,513/62,537 are checked. Mapping uses
`load_state_dict` into existing Parameters, preserving optimizer ownership and
empty state. Used, non-CPU or already mapped models and incompatible/nonfinite
source tensors reject; complete validation precedes mutation.

Five initializer tests verify11/23/37 mappings, owner/tensor identity, zero
columns, unchanged Torch/NumPy/Python streams during mapping, malformed-shape
rejection before mutation and refusal after timestep use/double application.
The producer restores the pinned original CPU RNG before serialization and
checks NumPy/Python seeded states, without reading a dataset or stepping an
optimizer. Output must be a new portable `configs/initializations/` directory.
Full/masked share the exact same per-seed archive; smoke11 reuses the full64
seed11 architecture/assets, only collection settings differ.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/build_vix_initializations.py \
  --output configs/initializations/vix_v1
```

Actual source-data payload generation completed separately: masked
5184823a388f4cc02197a111fe29cde7f80edae14aa451aa139f791891f903c2 and full
b63683a2b78e5aaa7652069b75b5c95acc32748b8fdf2d4c8dcff825ef5d756a.
The70.04-second local producer reloaded/reconstructed both exactly, preserved
all source hashes and original coordinates, verified finite float32 inputs and
fitted only full's298,946 training rows. Independent pandas coordinates and NumPy training statistics pass in
`payload-verification.json` (coordinate error1.11e-16, mean/scale errors below
1e-10); every original episode field/seven-coordinate value remains exact. Initializer asset generation remains pending;
no training, external data request or Colab allocation.


## Data and untrained initializer binding accepted

[Actual initializer evidence](../results/vix-context-2026-10-25/initializers.json)
verifies12 registered arm/seed/phase entries with3 new portable23-input NPZ
archives plus the3 original legacy archives. Full/masked share each complete
per-seed file; smoke11 reuses the full64 seed11 files. Every common tensor and
original input/account column remains exact, new columns zero, CPU RNG identical
to the original per-seed asset, and NumPy/Python seeded fingerprints verified.
Original source assets remain unchanged. Generation began from clean3e01284
source; it read only protocol/initializer assets, not datasets or providers.

The binding `configs/experiments/vix_context_binding.json` pins preregistration,
three dataset identities/contracts, payload evidence and initializer manifest.
`training.vix_assets` pins its logical SHA256
5ee939d3c96a100365ad5bf9f19bfa656be763bf432e9e0ba511f09bc0e37f85,
checks generator implementation hashes, portable paths, actual tensor/RNG
archives and all three seeds. Unknown phases/seeds, changed bindings, escaped
paths and inconsistent proofs reject. No model is created by asset validation.

70 focused mapping/asset/payload/frozen/source/protocol and previous-family
regression checks pass. Actual local payloads and untrained initializers are
accepted; **learned optimizer/runtime acceptance is not established**. No policy
optimization, new provider request or Colab allocation occurred. The next work
is explicit VIX-family training/task/archive/report integration and actual
Colab smokes under the shared budget. Legacy parameters/data overlap older
controls, so family dispatch must be explicit and preserve old behavior; do not
route this study through the old market-context job contract.

### VIX runner integration

The explicit registered VIX protocol now gates runner requests before environment
or policy construction. Data identities, frozen preprocessing, seeds, sampling,
execution, reward, transitions and checkpoint cadence are fixed. Shared initial
weights/RNG and initial/selected/final actual model/Adam records are retained with
a checksummed initializer snapshot. Trained global RNG restoration is not claimed.
See the [study evidence](../results/vix-context-2026-10-25/README.md) for current
checks. Job/fleet/archive integration and actual Colab smoke acceptance remain;
there are still no trained VIX performance results.

### Colab task and retrieval contracts

Version34 plans bind each legacy/masked/full dataset, initializer and fixed
512-transition smoke or100k screen settings. Full plans require a checksummed,
same-protocol actual smoke acceptance file and a conservative deadline projection.
The projection uses twice the historical1055.604-second control duration times
the slowest smoke-to-legacy ratio, with1200 seconds reserved, and rejects totals
above7200 seconds. This is a heuristic; live deadline enforcement remains required.

The central fleet automatically persists `.runtime/vix-study-budget.json` beside
the resolved original usage ledger. Its original baseline survives smoke/screen
controllers and retries; the allowance cannot exceed2 units. Project150/130 and
account150/170 safeguards still apply. VIX tasks use a dedicated fleet so another
study cannot replace that baseline. Source uploads include all bound initializer
files. Preflight and the remote worker validate immutable payload identities.

Retrieved archives require actual initial/selected/final model and Adam records,
initializer bytes, checkpoint identities, execution/sampling sidecars, training-only
year counts, all26 development symbols and frozen schema/normalization/source
lineage. Complete selected weights must equal the frozen model. These checks do
not replace physical learned-model restoration or report accounting acceptance.

Prepare smoke tasks without provisioning:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_vix_jobs.py \
  --smoke --output .runtime/vix-smoke-jobs.json
```

Actual learned-artifact acceptance is still pending; do not run screens until the
three completed, owned-cleaned Colab smokes pass it. Inspect each actual default
kernel Python before dependency synchronization. No VIX return is available yet.

### Learned checkpoint acceptance

`training/vix_restoration.py` restores latest, selected and frozen recurrent
models and compares actual tensor/Adam/provenance records exactly, excluding
unrestored global sampling streams. It checks three causal prefix decisions for
each of26 development episodes (at least78 predictions), carried/reset hidden
states, deterministic repeats, caller-state immutability, schema/feed/nonfinite
rejection and nonzero masked-coordinate rejection. Models, normalization, source
files and dataset identity must remain unchanged; caller RNG is restored on both
success and failure. Verification performs no optimizer steps.

After the owned fleet is complete and stopped, `scripts/accept_vix_smokes.py`
revalidates retained archives, clean training-source identity, exact task/session
mapping, report accounting against each immutable dataset, actual saved models
and cross-arm initial-state agreement. Its output belongs at
`.runtime/vix-smoke-acceptance.json`; it is the input to full-job runtime gating.
Serialization fixtures validate the checker, not actual learned acceptance.

Actual three-arm Colab smoke acceptance now passes:512 transitions per arm,
234 exact frozen sequence predictions and actual learned Adam restoration.
All default kernels were3.13.16; all owned runtimes are stopped. The conservative
3471.70-second screen projection fits the7200-second allocation deadline. See
[accepted evidence and usage](../results/vix-context-2026-10-25/README.md).
Only the fixed100k screen is next; no performance benefit is established.

### Fixed screen reporting

`scripts/report_vix_study.py` accepts only all three complete, retrieved and
owned-cleaned100k seed11 tasks. It revalidates archive semantics, clean training
source, immutable payloads, actual latest/selected/frozen Adam/recurrent state,
matched holding paths and selected/final report accounting. Cross-arm initial
states and actual symbol/group/year sampling counts must agree. Declining-stock
and action-execution diagnostics remain reporting-only.

Historical legacy reproduction checks the preregistered source metadata and both
report checksums, every checkpoint mean within0.01 percentage points and both
selected/final per-symbol returns within0.05 points. A mismatch invalidates
interpretation/promotion without widening tolerances or automatic retry. Numeric
performance gates are exactly those in the original preregistration; no saved
checkpoint is reselected by the reporter. A standalone interactive chart accompanies
the immutable summary. Passing development gates still requires multi-seed and
prospective confirmation; failure closes the fixed VIX family.
