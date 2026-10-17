# Frozen forward shadow campaign

The user authorized persistent shadow observation of the saved promising model.
The [recipe](../../configs/shadow/forward_validation.json) fixes nine selected
bundles: combined SAC seeds 11/23/37 (including the historical 67.24% validation
checkpoint), basic PPO seeds 11/23/37, and basic SAC seeds 11/23/37.

The campaign uses a fixed 126-settled-session research horizon, unchanged
simulation costs and a buy-and-hold/cash comparison. Each model/stock has an
independent $100,000 simulated account. It has no broker-order path and cannot
automatically become paper-eligible. All policies and seeds remain reported;
forward data must not select a replacement checkpoint.

[Software acceptance](software-acceptance.json) checks the actual nine bundles
against previously inspected validation inputs: features/ATR match the stored
research data and observations match the training environment for all 26 symbols.
Normalization is unchanged and frozen actions are finite. These checks do not
produce fresh performance claims. Full local acceptance passed 156 tests.

The [immutable registration summary](registration.json) and [operating status](status.json)
record the actual campaign. The first
eligible signal must use a bar closing after registration; older bars are warmup.
There are no forward returns at initialization. The retired 2024–September 2026
interval is not reopened as an independent audit. Historical membership and
delisting coverage remain unverified.

See [operation, reports and recovery](../../docs/shadow-inference.md). Runtime
journals, raw data and process logs remain private/ignored. A persistent host is
required for daily uptime; installing the provided service template is an
operator deployment step, not something the template performs automatically.

## Actual registration and current blocker

Campaign `0747a6730378e055ccfa2ac6fed2a1be9b58991f0a3039c4373947478f543965`
was registered at **2026-10-16 22:55:19 UTC**, using clean source `e6db469`.
Its local state is under `artifacts/shadow/CAMPAIGN_ID/`; `report.html` currently
contains no forward outcomes. The one-shot scheduler correctly waits for a
post-registration close. A read-only fresh-feed preflight, however, failed on
Yahoo's October 16 SPY daily bar: high was approximately one cent below open.
A separate fresh request confirmed the same invalid geometry. Prices were not
clamped, repaired, substituted or used in simulated fills.

**Unattended polling was not started.** No forward decisions/returns exist and
no Colab runtime was allocated. The cumulative ledger remains 35.389379 units.
The immutable campaign is retained while the provider problem is reviewed.

After upstream correction, run this read-only collection/overlap audit before
starting the scheduler (substitute the full local campaign path):

```python
from rl_stocks.inference.shadow.collect import collect
from rl_stocks.inference.shadow.protocol import read_protocol
from rl_stocks.inference.shadow.runner import merged_bars
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.core.contracts import utcnow

receipt = collect(campaign, "data/shadow")
merged_bars(read_protocol(campaign), [str(receipt.resolve())], timestamp(utcnow()))
```

Then run the documented `run CAMPAIGN --once` command and inspect its report;
only after a successful feed cycle should the persistent process be enabled.
The first possible signal for this registration is the **October 19 close**, and
the first possible simulated execution is the **October 20 open**. Those dates
are opportunities, not recorded decisions. If data readiness delays the first
signal, record the later actual start and keep all skipped dates unscored; no
historical signals may be backfilled. Once decisions begin, a missed execution
session halts the campaign for review. No model or gate should be changed to
repair a data-feed issue.

## Approved correction and successful preflight

On October 17 the user authorized correcting the small SPY high/open error.
The [correction preflight](correction-preflight.json) now passes all 37 symbols,
with exactly one audited correction. The raw response is preserved, high is
raised to the reported open, and all other fields retain their source values.
The rule requires the exact observed OHLC checksum and caps the change at $0.011;
other invalid rows still fail. Default Yahoo ingestion is unchanged.

The initial campaign and its blocked status are retained under [history](history/initial-registration.json).
A replacement registration pins the new policy and implementation with unchanged
models, seeds, costs and review horizon; see the current [registration](registration.json)
and [status](status.json). No historical study result was recomputed or improved
by this correction. Full software acceptance passes 160 tests.
