# Research model releases

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
