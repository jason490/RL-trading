# Research artifacts and retention

## Universe expansion cleanup — 2026-10-17

All seven owned allocations stopped and all six research bundles passed frozen
acceptance before cleanup. After verifying all 13 GitHub asset SHA-256 digests
and sizes, cleanup removed 48 redundant files totaling 413,585,593 logical bytes
(394.43 MiB). Observed free space increased by 413,687,808 bytes. All 146 retained
files were rehashed unchanged, including selected/final weights, explicit 50k
post-update checkpoints, full frozen directories, curves and detailed reports.
Smoke evidence, raw datasets, source manifests, cumulative ledger and session
recovery records remain. Earlier studies and unrelated files were untouched.
See [inventory](../results/universe-expansion-2026-10-17/cleanup.json) and
[release verification](../results/universe-expansion-2026-10-17/release-verification.json).

Research outputs are separated from source code so models can be inspected and
reused without committing large data snapshots or private runtime state.

## What to retain

| Output | Purpose | Publication |
| --- | --- | --- |
| `results/STUDY/` | Declared validation selection, per-seed metrics, inference checks and held-out audit summary | Git |
| `artifacts/public/STUDY/` | Allowlisted model archive, derived charts, catalog and checksums | GitHub release assets |
| Retrieved `artifacts/runs/RUN/` directories | Best and latest weights, metadata, checkpoint scores, detailed reports and final replay buffers | Local; only allowlisted files enter a release |
| Retrieved `artifacts/frozen/BUNDLE_ID/` directories | Exact weights, schemas, normalization, dependency manifests and file checksums | Local and release archive |
| `data/raw/`, `data/sectors/`, `data/datasets/` | Immutable provider snapshots, provenance and reproducible datasets | Local; original provider data is not redistributed in releases |
| `.runtime/usage-ledger.json` | Cumulative metering, attempts, deadlines and cleanup evidence | Private local state; sanitized summary may enter results |
| `.runtime/` session/authentication files | Runtime recovery and existing CLI authentication | Private local state only |

Keep every file in a frozen bundle together. Renaming its content-addressed
directory, adding files, editing normalization, or dropping a dependency
manifest can invalidate inference checks. Preserve the model catalog and the
recorded source revisions even after temporary copies are removed.

Final SAC/TD3 replay files support continued learning. They are distinct from
frozen inference weights, and do not by themselves restore exact environment or
RNG state. A release intentionally excludes replay and original provider bars;
it is not a complete replacement for local research evidence.

## Cleanup order

1. Retrieve the final artifacts before stopping each owned runtime. Verify
   termination and resolve any cleanup failures before deleting recovery data.
2. Verify all required research cells, frozen checksums and deterministic
   save/load predictions. Record failures or interruptions rather than hiding
   them. Preserve the committed validation selection separately from its audit.
3. Publish the allowlisted assets and compare every server-reported SHA-256
   digest with the local files. Keep a local verified copy of the release.
4. Inventory candidate temporary files with paths, sizes and hashes. Confirm
   each retained run still has its selected weights, metadata, normalization,
   dataset identity and reports. Keep final replay buffers when continuation
   remains useful.
5. Remove redundant downloaded `.tar.gz` copies, intermediate checkpoint weights,
   duplicate checkpoint replay files, and incomplete transient outputs only
   after their retained replacements are verified. Preserve final run outputs
   and required evidence referenced by the documentation.
6. Recheck retained bundles, source status and the runtime ledger. Record the
   removed file count and measured space recovered in the study results.

Do not use a blanket deletion of `artifacts/`, `data/` or `.runtime/`. A usage
ledger must survive retries and hardware changes. Do not follow symlinks into
another workspace or remove unrelated user files. Cache cleanup must preserve
the active Python environment, interpreter and installed official Colab CLI.

## Opening reports and models later

Download the release assets and verify `SHA256SUMS` before extraction. Open
the index in `validation-reports.tar.gz` for checkpoint-selected validation
simulations (extract it with its `validation/` folder) and `test.html`
for the separate held-out audit. The reports contain derived equity and metrics;
they do not include the raw provider feed or imply a single investable portfolio.

Use the full frozen bundle with the matching feature schema and feed described
in [inference](inference.md). For exact data reconstruction and further research,
retain the local immutable dataset snapshots and the hash-locked dependencies.
See [releases](releases.md), [verification](verification.md) and
[Colab recovery](colab.md) for publication evidence and runtime status.

## Indicator follow-up cleanup

The completed 2026-10-16 indicator release has 14 GitHub assets whose server
SHA-256 digests and sizes match local copies. Only after that verification,
cleanup removed 37 temporary/duplicate files totaling 1,942,764,002 logical bytes
(about 1.81 GiB). Observed free space increased by 1,942,892,544 bytes. All nine
new frozen policies, 27 post-update milestone weights, best/latest weights,
final replay buffers, compact curves and detailed validation outputs were kept;
144 retained files were rehashed after cleanup. The unique interrupted-attempt
checkpoint, immutable data and cumulative ledger/recovery records remain. Earlier
study artifacts were untouched. See the tracked [cleanup record](../results/indicator-followup-2026-10-16/cleanup.json)
and [release verification](../results/indicator-followup-2026-10-16/release-verification.json).

## Algorithm stability cleanup

The completed nine-cell algorithm/memory release has 13 assets with verified
server SHA-256 digests and sizes. Cleanup then removed 44 duplicate/temporary
files totaling 678,561,987 logical bytes (about 647 MiB), including final transfer
archives, rolling checkpoint duplicates and isolated owned upload payloads.
Observed free space increased by 678,645,760 bytes. All 129 retained files were
rehashed: nine frozen bundles, best/latest weights, nine explicit 100k milestone
models, final SAC replay buffers, curves and detailed validation remain. Smoke
outputs and unique failed-preflight checkpoints, datasets, cumulative ledger and
recovery records were preserved. Earlier studies were untouched. See
[cleanup](../results/algorithm-stability-2026-10-16/cleanup.json) and
[release verification](../results/algorithm-stability-2026-10-16/release-verification.json).

## Initialization-study cleanup — 2026-10-19

All 12 owned sessions stopped before cleanup. Seventeen GitHub assets passed
server SHA-256/size checks. Each final download archive matched its extracted
file set byte-for-byte. Cleanup then removed 52 task-owned upload/download/source
archives and ready markers, including duplicate compressed publication assets:
642,046,525 logical bytes (612.30 MiB). All 228 retained files were rehashed
before and after deletion with no changes. Selected/final/intermediate weights,
frozen bundles, full reports, compact curves, source export, raw data and the
project ledger remain. Authentication/configuration and unrelated studies were
preserved. The release can recover the removed public compressed assets.

Exact paths and before/after evidence are in
[cleanup.json](../results/action-initialization-2026-10-19/cleanup.json); server
asset receipts are in
[release-verification.json](../results/action-initialization-2026-10-19/release-verification.json).
No cleanup failure or unresolved owned session remains.

## Causal-sampling study cleanup — 2026-10-19

All eight owned sessions stopped, and 17 published assets passed GitHub server
SHA-256/size checks. All final archives matched their extracted file sets.
Cleanup removed 36 owned upload/download/source archives, ready markers and
redundant public compressed copies: 537,871,153 logical bytes (512.95 MiB).
All 160 retained artifact hashes are unchanged. Raw snapshots, selected/final
weights, frozen bundles, full reports, curves, source export and the shared
ledger remain; authentication, unrelated studies and user files were preserved.
No unresolved ownership or cleanup failure remains. Exact paths and hashes are
in [cleanup evidence](../results/causal-sampling-2026-10-19/cleanup.json) and
[server receipts](../results/causal-sampling-2026-10-19/release-verification.json).

## Reward-objective study cleanup — 2026-10-20

Four owned CPU sessions stopped; all13 release assets passed server SHA-256/size
checks. Final archives matched all extracted files and upload payloads matched
preserved source/data/plans before removal. Cleanup removed15 duplicates totaling
124,682,332 logical bytes (118.91MiB), with272 retained hashes unchanged. Three
earlier snapshot archives differed from final extraction and remain. Original
models, full reports, curves, source export, datasets and shared ledger remain.
See [cleanup](../results/reward-objectives-2026-10-20/cleanup.json) and
[release receipts](../results/reward-objectives-2026-10-20/release-verification.json).

## Trend-context study cleanup — 2026-10-20

All6 owned CPU sessions stopped and14 release assets passed server SHA-256/size
verification. Final archives matched extracted files; payloads matched preserved
source/data/plans. Cleanup removed21 proven duplicates totaling196,449,793 logical
bytes (187.35MiB), with316 retained hashes unchanged. Six earlier snapshots were
kept because they are not byte-identical to final output. Original models, full
reports/curves, source export, raw snapshot,3 datasets and ledger remain.
[Cleanup](../results/trend-context-2026-10-20/cleanup.json),
[server receipts](../results/trend-context-2026-10-20/release-verification.json).

## Supervised feasibility cleanup — 2026-10-20

One owned CPU session stopped;12 forecast-only release assets passed server
SHA-256/size checks. Final transfers matched all extracted files, upload payloads
matched retained source/data/plans, and6 staged model copies matched the frozen
bundle. Cleanup removed11 duplicates (10,857,969 logical bytes;10.36MiB), with233
retained hashes unchanged. Unique snapshots, probe predictions, fit logs, prepared
data, source export and ledger remain. [Cleanup](../results/supervised-signal-2026-10-20/cleanup.json),
[server receipts](../results/supervised-signal-2026-10-20/release-verification.json).

## Cash-choice publication cleanup — 2026-10-22

All15 server asset sizes/digests verified before removing two duplicate publication archives totaling58,832,238 bytes. All249 retained hashes unchanged; original transports, complete model/frozen/checkpoint/report/counter evidence, source/failure archives, immutable dataset and ledger retained. Post-cleanup strict summary reproduces byte for byte. All12 owned sessions stopped. [Inventory](../results/recurrent-cash-2026-10-22/cleanup.json), [reproduction](../results/recurrent-cash-2026-10-22/reproduction.json).

Recurrent fitted-period charts published: [release](https://github.com/jason490/RL-trading/releases/tag/recurrent-fit-2026-10-22), six server sizes/digests verified,6,722,998 duplicate bytes removed and30 retained hashes unchanged. Original complete paths retained, no model duplication or new Colab charge. [Evidence](../results/recurrent-fit-2026-10-22/README.md).


Capacity publication verified:15 GitHub assets match server sizes/SHA-256,
four accepted model bundles and interactive reports retained. Only two duplicate
publication archives removed (45,880,597 bytes),142 retained hashes unchanged.
Original transports/source/data/models/checkpoints/ledger/plan.html preserved.
All four sessions stopped; project38.60166524 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-capacity-2026-10-22).


Training-history publication verified:17 GitHub asset sizes/digests match, four
frozen bundles and interactive reports public. Only two duplicate publication
archives removed (35,931,249 bytes),148 retained hashes unchanged. Original
transports/models/source/both datasets/ledger/plan.html retained. All four sessions
stopped, project38.68329334 unchanged. Fixed family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/training-history-2026-10-23).


History-model fitted-period charts published:7 server assets verified,104 equity
plots on four pages, no model duplication or raw bars. Duplicate chart archive
removed (3,373,270 bytes),19 retained hashes unchanged; full paths retained. No
additional Colab units; project38.68329334 unchanged.
[Release](https://github.com/jason490/RL-trading/releases/tag/history-fit-2026-10-23).


Recurrent L2 publication complete:16 server assets verified, four model bundles
and interactive reports public. Only two duplicate archives removed (35,744,305
bytes),145 retained hashes unchanged; original source/data/models/checkpoints/
ledger/plan.html preserved. Strict summary reproduces byte for byte from retained
transports with actual models/Adam/coupling restored. All four sessions stopped;
project38.74617914 unchanged. Fixed L2 family closed.
[Release](https://github.com/jason490/RL-trading/releases/tag/recurrent-regularization-2026-10-23).


Distributional-critic publication/cleanup verified:12 GitHub release asset hashes
and sizes match;18 duplicate transports/upload payloads removed (520,262,328
bytes),191 retained files rehashed at cleanup. All models/replay/checkpoints,
source/data, curves and owned recovery records retained. Subsequent README
publication update is explicitly recorded in cleanup evidence. Six owned sessions
stopped, zero assignments; no additional units or training. Fixed critic study
failed and remains closed; next work is bounded saved-policy/execution sensitivity
diagnosis, not extension or promotion. Release:
https://github.com/jason490/RL-trading/releases/tag/distributional-critics-2026-10-24
