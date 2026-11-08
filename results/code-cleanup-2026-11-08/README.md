# Code cleanup — 2026-11-08

Reviewed644 tracked files under scripts/src for references and exact duplicate
function bodies. No source module was proven disposable by that scan; standalone
historical verification/report tools remain supported and were retained.

Removed five identical retained-report loader bodies from the critic, CrossQ,
GAE, Treasury and VIX report scripts. They now import one shared implementation
from `reporting/retained_reports.py`, preserving their commands and return
contracts. Removed an unused sampling-validation import from the recency checker.

Standard source-payload builders now exclude generated Python cache directories
and standalone pyc/pyo files via `runtime/colab/source_archive.py`. This changes
upload contents only; source, dependency locks, configurations and data payloads
remain included. The dedicated TRPO upload already excludes caches.

Deleted664 generated pyc/pyo files under scripts/src, totaling4,291,513 logical
bytes. Empty cache directories were removed. Source Python files were hashed
before/after deletion and matched. Cache files may regenerate during normal
Python use; they are no longer uploaded by the patched builders.

[Cleanup inventory](cleanup.json) records each removed path, size/hash and
unchanged shared ledger/plan.html hashes. No raw data, dataset, model, retained
source snapshot, credential, budget record or recovery tool was deleted. No
Colab allocation or training occurred.

Validation:58 focused report-loader, upload/fleet and critic/GAE/VIX job checks
pass. Ruff passes on the consolidated loader, archive filter, new tests and five
report scripts; the legacy payload builder passes its established E701/E702
exceptions. This is not a claim that the full repository lint/test suite ran.
