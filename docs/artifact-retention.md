# Research artifacts and retention

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
