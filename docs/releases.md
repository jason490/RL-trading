# Research model releases

## Matched training-universe expansion

[universe-expansion-2026-10-17](https://github.com/jason490/RL-trading/releases/tag/universe-expansion-2026-10-17)
contains six Colab CPU PPO models comparing 22 versus 55 training stocks, plus
four ETFs, on the same original validation panel. Mean selected/final returns
were control 11.77%/10.49% and expanded 10.53%/5.60%, versus buy-and-hold 30.60%.
The expansion gate failed; these are development results, not an independent test.

All 13 release asset sizes and SHA-256 digests match local copies. Open
`universe-comparison.html` for interactive return and percentage-point-change
charts. `learning-curves.tar.gz` preserves each seed's JSONL/CSV/HTML histories;
`research-models.tar.gz` contains all complete frozen bundles and selected/final
weights. Six bundles passed 468 exact prediction comparisons. Raw provider bars
and credentials are excluded. See [results and acceptance](../results/universe-expansion-2026-10-17/README.md),
[verified release hashes](../results/universe-expansion-2026-10-17/release-verification.json)
and [retention inventory](../results/universe-expansion-2026-10-17/cleanup.json).

Large research models are published as GitHub release assets. Source, reviewed
experiment protocols and compact results remain in Git. Original provider bars,
replay buffers and private runtime logs remain local.

## Initial four-ETF baseline

[research-baseline-2026-10-14](https://github.com/jason490/RL-trading/releases/tag/research-baseline-2026-10-14)
contains five completed CPU/CUDA models from the original A100 session, including
PPO 50,000 transitions, seed 11, on SPY/QQQ/IWM/DIA. Their training source revision
is recorded in metadata; the release tag identifies the later packaging tools.

| Asset | Contents |
| --- | --- |
| `research-models.tar.gz` | Exact frozen bundles, selected/final weights, run metadata and checkpoint histories |
| `validation.html` | Derived selected-model equity versus buy-and-hold and metrics; original PPO chart is a CPU replay |
| `model-catalog.json` | All five completed model records |
| `SHA256SUMS` | Checksums for the three assets above |

All four server-reported release asset digests were checked against local
SHA-256 values after upload. Extract models into a separate directory and retain
each frozen bundle's complete content-addressed directory; adding/removing files
inside a bundle fails inference integrity checks. Use [FrozenPolicy](inference.md)
with the declared schema/feed and reviewed dependencies.

These are research candidates with `paper_eligible=false`. Validation selected
the checkpoints. This release contains no held-out test evaluation and no
expanded-stock, supply/demand or sector-rotation trained model. Subsequent study
releases will identify those datasets explicitly; see [experiments](experiments.md)
and [verification](verification.md).

## Expanded baseline PPO — interim

[expanded-baseline-ppo-2026-10-14](https://github.com/jason490/RL-trading/releases/tag/expanded-baseline-ppo-2026-10-14)
contains three 50,000-transition PPO runs for seeds 11/23/37 plus four startup
smoke models on 26 separate symbol episodes. This baseline uses seven daily
market inputs; supply/demand and sector rotation are not enabled. Selected
cumulative validation returns are 16.50%, 13.05% and 9.36%; mean 12.97%, sample
seed standard deviation 3.57 percentage points. The final test is untouched.
All four release asset digests match local SHA-256 values.

[Compact results](../results/expanded-baseline-ppo-2026-10-14/summary.json) include
per-symbol metrics, cash/buy-and-hold references, selected checkpoints, bundle
identities and training provenance. The completed comparison and separate held-out audit are published in the final study release below.

## Packaging later study assets

Run `PYTHONPATH=src .venv/bin/python scripts/package_research.py ARTIFACT_ROOT...
--output artifacts/public/STUDY_NAME` to prepare allowlisted frozen models,
run weights/metadata and derived validation charts. After a complete held-out
audit, add `--audit artifacts/audits/AUDIT_NAME` to export `test.html` and
`audit-summary.json`. Price/quote rows, replay files and runtime logs are not
included. Inspect the package, generate asset checksums and verify the uploaded
release digests before recording publication as complete.

Large studies use a small `validation.html` index plus one HTML page per run.
Extract `validation-reports.tar.gz`, retaining its `validation/` directory beside
the index. This avoids loading all symbols/seeds into one very large browser
document. `test.html` contains the three preselected family seeds' derived audit
charts. Older releases retain their original standalone validation pages.

## Completed expanded-universe study

[sector-study-2026-10-14](https://github.com/jason490/RL-trading/releases/tag/sector-study-2026-10-14)
contains all 36 research models plus four startup-smoke models. The index and
per-run pages are in `validation-reports.tar.gz`; `test.html` contains the frozen
selected family's three-seed audit charts. The catalog records actual devices
and source/dataset identities. Protocol, immutable validation selection, audit
summary and compute summary are included. All ten uploaded asset sizes and
SHA-256 digests were independently verified.

The selected combined-SAC family averaged 19.79% on validation and 10.83% on the
held-out interval, versus 30.60% and 48.46% for buy-and-hold. Candidate seed 23
returned 13.90% on test. These are separate symbol episodes; substantial seed
variation, retrospective universe selection and mixed training hardware limit
interpretation. The release is a prerelease research artifact, not a profitable
strategy or permission to submit orders.

See [full results](../results/sector-study-2026-10-14/README.md),
[verified release hashes](../results/sector-study-2026-10-14/release-verification.json)
and [cleanup inventory](../results/sector-study-2026-10-14/cleanup.json).

## Algorithm and recurrent-memory comparison

[algorithm-stability-2026-10-16](https://github.com/jason490/RL-trading/releases/tag/algorithm-stability-2026-10-16)
contains all nine 100k SAC/PPO/RecurrentPPO research models, explicit endpoint
weights, compact learning curves, standalone `comparison.html`, per-stock
validation reports, protocol and sanitized usage evidence. No new test audit
was performed. All gates failed; no model is paper-eligible. See
[results](../results/algorithm-stability-2026-10-16/README.md) for interpretation
and recorded publication/cleanup verification.

## PPO initialization screen and confirmation — 2026-10-19

[Download the release](https://github.com/jason490/RL-trading/releases/tag/action-initialization-2026-10-19). All 17 server asset hashes and sizes are verified.
Eight 50k CPU research models plus four smokes are cataloged with initialization
variant, seed, source and normalization. Open `confirmation.html` after downloading
for paired checkpoint returns, incremental changes and drawdowns; `screen.html`
retains all four initial arms. `research-models.tar.gz` contains frozen bundles
and selected/final weights; `learning-curves.tar.gz` contains per-run JSONL, CSV
and HTML curves. The validation archive contains derived equity reports.

Mean final return improves from 10.49% to 13.89%, still below holding 30.60%.
Declining-stock performance worsens and short-profit requirements remain unmet.
The release is development evidence; it does not replace the registered shadow
campaign or enable broker orders. Full limitations and checksums are recorded in
[the results](../results/action-initialization-2026-10-19/README.md).

Public validation indexes identify each model by algorithm, research variant,
dataset prefix, seed, actual transitions and smoke/research status. The model
catalog also records `run_name` to locate its archive/report folder. This keeps
small smoke runs and models with different training universes distinguishable
when a release contains multiple cells of the same algorithm and seed.

## Causal sampling and universe screen — 2026-10-19

[Download the release](https://github.com/jason490/RL-trading/releases/tag/causal-sampling-2026-10-19). All 17 server asset hashes/sizes are verified.
The four seed-11 50k arms failed their ETF promotion constraint; no confirmation
or longer runs followed. Catalogs distinguish four research models from four
512-step smokes, including sampler variant and training dataset. Open `screen.html`
for all four learning curves and incremental changes; `directions.csv` records
requested/executed direction and side P&L. The learning-curve archive includes
final sampling counters. Manifests, frozen checks, training-only forward analysis
and the synthetic execution-observability witness are also included.

These are negative development results, not a new winning model or independent
performance evidence. [Full results](../results/causal-sampling-2026-10-19/README.md)
retain the gate, source identity, budget and cleanup receipts.

## Recurrent cash-choice screen — 2026-10-22

[Verified release](https://github.com/jason490/RL-trading/releases/tag/recurrent-cash-2026-10-22) publishes eight accepted smoke/screen models and15 assets. Cash candidate failed selected/final return gates; binary’s34.38% selected checkpoint deteriorated to−1.25% final. No robust/prospective winner. Open `comparison.html` for checkpoint curves; extract `validation-reports.tar.gz` for derived equity reports and `research-models.tar.gz` for frozen/selected/latest models. Server sizes/digests all match; original data and private logs excluded. [Full study](../results/recurrent-cash-2026-10-22/README.md).

Recurrent fitted-period charts published: [release](https://github.com/jason490/RL-trading/releases/tag/recurrent-fit-2026-10-22), six server sizes/digests verified,6,722,998 duplicate bytes removed and30 retained hashes unchanged. Original complete paths retained, no model duplication or new Colab charge. [Evidence](../results/recurrent-fit-2026-10-22/README.md).


Training-history publication verified:17 GitHub asset sizes/digests match, four
frozen bundles and interactive reports public. Only two duplicate publication
archives removed (35,931,249 bytes),148 retained hashes unchanged. Original
transports/models/source/both datasets/ledger/plan.html retained. All four sessions
stopped, project38.68329334 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/training-history-2026-10-23).


Recurrent L2 publication complete:16 server assets verified, four model bundles
and interactive reports public. Only two duplicate archives removed (35,744,305
bytes),145 retained hashes unchanged; original source/data/models/checkpoints/
ledger/plan.html preserved. Strict summary reproduces byte for byte from retained
transports with actual models/Adam/coupling restored. All four sessions stopped;
project38.74617914 unchanged. Fixed L2 family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-regularization-2026-10-23).
