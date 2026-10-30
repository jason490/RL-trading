# Reviewed continuation after the recurrent profile stop

## Current status and scope

The original CPU profile stopped at its working deadline. Its claim is consumed,
owned shutdown is verified, and the original project ledger remains stopped.
The user explicitly approved the corrected profile on2026-10-30. It completed,
passed physical verification and ended with verified owned-session cleanup.
No100k training was included.
[Readiness evidence](../results/recurrent-offpolicy-2026-10-27/corrected-profile-readiness.json)
records the earlier prepared-state input/start acceptance.
[Completed-run evidence](../results/recurrent-offpolicy-2026-10-30/run.json) records
the approved execution; the old stopped attempt and claim remain preserved.

The user-supplied operating rule prohibits automatic restart after a budget stop.
[Colab operations](colab.md) says: “It never automatically restarts after a budget
stop.” The existing `Ledger.review_resume` also requires an explicit reviewed
authorization, the exact stopped attempt, verified cleanup and healthy idle
metering. A generic worker completion or a fabricated authorization is insufficient.

This review covers one corrected engineering profile, not performance training:

| Item | Bound |
| --- | --- |
| Hardware and concurrency | One Colab CPU session |
| Allocation deadline | 1,200 seconds, including setup and retrieval |
| Teardown reserve | 420 seconds; begin teardown by 780 seconds |
| Profile allowance | 0.1 units shared across original and corrected attempts |
| Project ceiling / teardown | Original150 /130 units |
| Account minimum / teardown | Original150 /170 units |
| Attempts | One corrected attempt, no automatic retry |
| Performance selection | Disabled |

The original two-unit study allowance and its start remain unchanged. The
longer6/3/2-hour CPU option remains a separate proposal; this review neither
increases the study cap nor admits any100k cell or confirmation seed.

## Fixed inputs and corrective work

[Execution review configuration](../configs/experiments/recurrent_sac_execution_review.json)
has canonical digest
`fc4ff33df54fc28641056231822b0026381f3d281a5077060a7c80ab192de491`.
It references the original research protocol, exact stopped fleet/child, original
profile claim, CrossQ integrity-stop hash and original study baseline. Those
records must remain intact. The original failed profile does not become accepted.

Use the same immutable111-training/26-development dataset and the same nine
unfitted portable starts. No provider requests, new features, changed models,
rewards, transition budgets, checkpoint rules or performance gates are included.
The corrected engineering work uses lossless gzip1 transport, checkpoint-first
publication, the observed256MiB archive envelope and effective native
parameter/constant-schedule checks. Native profiling now includes the same
finite-gradient/weight hooks as the performance worker, with actual64/600
optimizer limits. It does not perform extra updates to improve profile scores.

## Preparation and explicit authorization

Prepare the task without allocating:

```bash
MPLCONFIGDIR="$PWD/.runtime/matplotlib" PYTHONPATH=src \
  .venv/bin/python scripts/build_reviewed_recurrent_profile_jobs.py \
  --output .runtime/recurrent-corrected-profile-jobs-v1.json
```

The builder refuses to overwrite an existing task file. Prepared jobs do not
clear the ledger stop. Fleet construction rejects missing reviewed authorization.

Only after an explicit user approval for this scope, record it with
`scripts/review_recurrent_profile.py --authorization <review-reference>
--cli-python <existing-official-cli-interpreter>`. The script locks the original
allocation ledger, reads official usage, settles delayed debits on the exact
stopped fleet, and records the user's authorization through `Ledger.review_resume`.
It preserves charges, attempts, claims, historical stop reasons and the account
floor; it does not allocate a session. Approval-record publication precedes the ledger resumption. If publication fails,
the ledger stays unchanged and stopped. Allocation requires both the record and
its matching persisted ledger review; either record alone is insufficient.

The agent must not invent approval text or execute this authorization command
solely because the prepared task exists. The configured review is a proposal,
not evidence that the user approved it. The authorization command ran only after the user explicitly said “approve the
profile.” Its persisted review preserves the original charges and stopped attempt.

## Allocation, acceptance and failure

Version42 routes directly to the guarded profiling worker. It packages portable
starts explicitly, requires the exact persisted review, and consumes a separate
single corrected-allocation claim before provisioning. The original version41
claim remains consumed. No execution retry, fleet restart or deadline extension
is automatic. Concurrency1 applies to this profile only.

The standard fleet remains responsible for aggregate minute metering, integrated
between-reading estimates, bounded retrieval and verified owned shutdown. The
existing strict archive verifier restores actual models/moments and checks
complete CPU/default-kernel/source/data evidence. Missing completion evidence
still rejects; no file or provenance is fabricated from an earlier attempt.

A completed profile may legitimately fail the original two-hour projection. That
blocks the existing performance plan. A separately reviewed execution proposal
would still need accepted physical profiling, current rates and unchanged
performance gates before any full training. All unsuccessful attempts remain
part of the shared project charge.

## Independent longer-runtime foundation

Opt-in budget/controller support for the later
6/3/2-hour CPU proposal was implemented and tested before the corrected profile. Existing plans keep the
original two-hour default. The longer API requires explicit per-task duration
slots and persisted account protection; completed slots cannot become automatic
retries. No registered task opts into that API yet. This work does not broaden
this review's20-minute profile scope, modify the original study envelope, clear
the stop or authorize any performance training.

## Accepted corrected-profile result — 2026-10-30

Session `rl-stocks-cpu-1b35c54f7de7`, clean738a573, actual default Python3.13.18,
CPU AVX2. The complete engineering task finished in548.72 seconds including
setup, retrieval and verified shutdown. Native finite-gradient/weight audits and
physical restoration validate64 GRU-SAC updates,64 SAC updates and3,000 PPO
transitions/600 Adam steps. Full CPU/source/data/report completion evidence passes.
A CLI reply timeout was recovered by retrieval/probing the same worker; no restart.

Required allocation projections, including reserves, are4.67h GRU-SAC,1.95h SAC
and1.21h RecurrentPPO. The original two-hour three-arm performance plan still
fails because of GRU-SAC. A separately admitted longer-CPU execution plan is
required; the approved engineering profile does not itself authorize it.

All scientific models/reports/inputs and both attempts remain retained. Verified
duplicate staging cleanup removed45,795,864 bytes;53 protected files are unchanged.
See the dated [result directory](../results/recurrent-offpolicy-2026-10-30/README.md)
for measured/estimated charging, exact timings and cleanup evidence.
