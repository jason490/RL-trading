# Task-conditioning preflight — 2026-10-31

The cached source audit confirms107 stock annotations and four broad ETF
annotations on the pinned111-symbol training payload. Dataset/normalizer identity
matches the fixed protocol. Dataset bytes, protocol bytes and the cumulative
usage ledger are unchanged by the audit. No fitting, development/test evaluation,
provider request or Colab allocation occurred.

[Evidence](type-evidence.json) deliberately supplies no `known_from` dates:
the universe configuration is a retrospective snapshot and does not establish
historical type availability. It is not a valid historical classification
manifest. Training admission remains pending historical primary-source records
covering the exact training universe, with reviewed availability at or before each
first training session. Existing survivor/revised-history limitations persist.

The fixed [protocol](../../configs/experiments/task_conditioning.json) compares
masked shared, tagged shared and tagged routed SAC. All use full22 observations,
64x64 trunks and equal100k-transition screens. Smokes retain research64x64
geometry while reducing replay batch/warmup. Checkpoints are selected on the
fixed training window; no development checkpoint reselection. Final training
admission must pass before any development scoring. Multi-seed development and
prospective evaluation remain necessary; this preflight is not a performance
result or runtime-profile acceptance.

The factory has no-optimization tests for full SB3 SAC save/load parity, replay
tag retention and role mismatch rejection. No worker or ordinary factory was
changed to start this experiment. Replay-update behavior, frozen-bundle metadata,
production manifest and centrally metered Colab worker integration remain pending.
