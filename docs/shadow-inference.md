# Persistent forward shadow inference

## Purpose and limits

This workflow freezes existing research policies and observes their decisions on
new daily bars. It runs locally or on a persistent server, using CPU inference
and the existing trade simulator. It does not train, allocate Colab, connect to a
trading account or submit orders. Short positions are simulation assumptions;
borrow availability is not verified. This is distinct from Alpaca paper trading.

The supplied `configs/shadow/forward_validation.json` registers nine selected
bundles: combined-input SAC seeds 11/23/37, basic-input PPO seeds 11/23/37, and
basic-input SAC seeds 11/23/37. The combined SAC seed 37 includes the historical
67.24% validation checkpoint. Every arm retains its own frozen normalizer, action
contract and training simulator settings. The original retrospective 26-symbol
universe is fixed; this does not resolve historical membership/survivorship bias.

The 2024–September 2026 interval remains retired. Campaign registration records
its actual UTC timestamp, and scored bars must close strictly after that timestamp
and after September 2026. Earlier bars can initialize indicators but generate no
forward returns. Registering a campaign after today's close means the first
eligible signal is after a later session closes; the first scored execution is
the following session's open. No historical backfill becomes prospective evidence.

## Freeze the campaign

Use the existing Python 3.13 environment matched to the last inspected Colab
kernel (3.13.16). Before dependency changes or new training, inspect Colab again
and follow [setup](setup.md); this workflow needs no new dependencies or Colab VM.
Resolve the downloaded frozen directories and the original immutable snapshot:

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.shadow prepare \
  --recipe configs/shadow/forward_validation.json \
  --bundles artifacts/colab \
  --snapshot data/raw/fced493827a124d4fb499d1c0e40009dc2420e19cd8712441ef65372d589457a \
  --output .runtime/forward-shadow-config.json

PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.shadow init \
  .runtime/forward-shadow-config.json --output artifacts/shadow
```

The command prints the content-addressed campaign directory. Set `CAMPAIGN` to
that exact path for later commands. `prepare` only resolves bundle IDs; `init`
verifies checksums, identity, feed, seed, common universe/bootstrap and simulator
settings through actual frozen-policy loading. A changed manifest, normalization,
feature implementation or protocol is rejected. The registration stores an
implementation fingerprint and Git provenance. Never edit a registered protocol
to replace an underperforming policy. A reviewed implementation update requires
a new campaign; the old campaign's positions/evidence must remain archived.

The locked review horizon is 126 settled exchange sessions. No gate is shown
before that horizon. At the horizon the process stops for research review; it does
not extend automatically or authorize orders. Each family needs at least three
registered seeds, positive average excess return over buy-and-hold, positive
excess in at least two seeds and mean maximum drawdown no more than two percentage
points worse than buy-and-hold. Passing is only a research-review result: these
seeds share one future market path and do not provide independent market samples.
Daily visibility must not become a checkpoint-selection or early-stopping search.

## Daily operation

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.shadow run "$CAMPAIGN" --once
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.shadow report "$CAMPAIGN"
```

The read-only Yahoo collector runs only after a new XNYS session has closed plus
15 minutes, and only for post-registration sessions. It requests the required
tradable/context symbols from the last observed date minus seven days through
today. Bars not yet complete/available are excluded. Original raw responses and
normalized receipts are stored immutably under `data/shadow/`; source snapshots,
request range and receipt times remain recoverable. No provider substitution is
allowed. Yahoo is unofficial and can revise history; the adapter's publication
lag is reconstructed, while new bars also use their actual retrieval timestamps.

For a previously collected canonical immutable snapshot, run without a provider:

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.shadow tick \
  "$CAMPAIGN" data/shadow/SNAPSHOT_ID
```

`tick` rebuilds causal features from the pinned bootstrap plus accepted receipts.
It never fits a normalizer. Incoming prices/actions must agree with previously
observed history wherever they overlap; availability/retrieval/revision metadata
may differ. The collector overlap checks recent history, not every possible
remote revision outside its request window. A new Yahoo split requires manual
adjustment review because Yahoo restates split-adjusted prices; continuing with
unadjusted old positions would produce incorrect P&L. Dividend and raw-action
accounting reuse the simulator. Unverified feed revisions must not be ignored.

All symbols and context ETFs must have the same completed current session,
correct exchange open/close timestamps, continuous forward context and fresh
observations. Historical warmup can retain the original archive's documented
coverage limitations; it is not silently relabeled as newly verified data.

## Decisions, simulated fills and restarts

At the first eligible close, each stock account starts with $100,000 cash and no
holdings. The policy produces a four-component action, decoded proposal, protective
geometry and intended next-session execution. Every decision records the bundle
ID, actual recording time, source snapshot, target session and action. Fills are
not assumed on that decision bar. A later completed bar settles only a signal
already journaled before that bar's open for that exact session.

Settlement reuses whole-share sizing, exposure caps, long/short costs, gaps,
protective exits, partial targets, reversals, corporate actions and the 20-session
holding cap from training. It persists cash, quantity, basis, age, costs,
protection, partial-exit state and pending action. Model observations include the
same five account-state values as the training environment. Recurrent models
add explicit hidden/cell state per arm/symbol; it is persisted transactionally,
starts at zero and is never shared across stocks or reset merely on a trade exit.

A SQLite `BEGIN IMMEDIATE` transaction and unique session key commit the entire
multi-model batch together. A process failure rolls back positions, memory,
receipt references, fills and decisions. Retrying a committed session creates no
new fills or decisions. Parallel processes serialize on the journal. Immutable
input receipts can remain after a failed transaction; they do not prove that a
signal was committed. Re-run safely only while the decision deadline still permits.

Missing a session halts processing. The system neither carries an expired signal
across an unobserved execution session nor fabricates a new signal using hindsight.
Review outages, restore all input receipts and preserve the journal. Once a
necessary pre-open decision window has passed, do not backdate a decision; archive
the interrupted campaign and register a new one, with separate results. No process
can reconstruct an actually missed prospective decision after the fact.

## Tracking and visual reports

Open `CAMPAIGN/report.html` for an interactive net-return chart of every registered
arm, buy-and-hold and cash. `report.csv` stores compact per-arm/session returns;
`report.json` includes arm/family means, seed variability, drawdowns, costs, all
recorded decisions, simulated fills and rejections. `journal.sqlite3` is the
canonical state; reports can be regenerated. Keep its WAL/SHM files with the
journal while active; use a SQLite-aware backup or stop writers before copying.

Returns average separate stock accounts equally; this is not a shared-capital
portfolio, annualized performance or executable allocation. Buy-and-hold uses the
same initial equity, fee/impact assumptions and first execution open, then holds
without stop/holding-limit constraints. Cash remains flat. Intermediate statistics
are descriptive; an empty chart means no forward outcomes, not a zero-return
strategy result. Frozen inference does not change training weights or normalization.

## Persistent process and failure recovery

For a foreground process on an always-on machine:

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.shadow run \
  "$CAMPAIGN" --poll-seconds 900
```

It waits between polls, avoids repeated provider requests for already processed
sessions, and exits at the review horizon. Interrupting the process safely leaves
the last complete transaction. On an operational/data error it records
`failures.jsonl` and exits; it does not restart or skip the problem automatically.
Do not expose raw Yahoo/CLI logs or private state through GitHub releases.

A service template is supplied in `configs/shadow/rl-stocks-shadow.service.example`.
Replace paths with the actual checkout and frozen campaign, then install it as a
user service on the host you intend to keep online. The template has `Restart=no`
so a data ambiguity cannot silently resume. Enabling a service does not keep a
sleeping laptop, stopped container or Colab VM alive. This repository cannot
provide unattended uptime by itself. Observe the first daily cycle and inspect
its report/failure log before relying on unattended scheduling.

## Verification and remaining work

Offline tests cover exact simulator parity across restarts, recurrent memory
isolation/persistence, transaction rollback, duplicate processing, timing,
revision/feed/protocol rejection, missing context, stale/future bars, fixed-horizon
review and report generation. The registered actual bundles are checked separately
with CPU inference; synthetic exercises are labeled as software checks, never
forward market evidence.

Historical membership/delisting expansion, live borrow eligibility, quote-level
fills, broker reconciliation, shared-capital portfolio execution and an Alpaca
order adapter remain separate milestones. A future paper-order implementation
needs its own explicit authorization and verified lifecycle; this shadow module
has no broker-order entry point.
