# TRPO engineering acceptance — 2026-11-08

The separately reviewed corrected CPU allocation completed all four fixed cases.
Independent physical acceptance and post-cleanup reverification pass. This is
engineering feasibility only: no economic paths/development returns were scored,
no candidate was promoted and retired2024+ history remained closed.

| Case | Transitions | Learn seconds | Actual Adam steps | Actual HVP calls | Accepted actor searches |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO smoke |1,024|6.803|320|0|—|
| TRPO smoke |1,024|3.130|320|134|8/8|
| PPO profile |2,000|8.961|400|0|—|
| TRPO profile |2,000|7.466|400|33|2/2|

Both algorithms have11,465 parameters and exactly matched full initial policies
within each pair. TRPO Adam state belongs only to critic parameters; its actor
uses native constrained line search. Independent maximum accepted Gaussian KL
is0.00683742 smoke and0.00568785 profile, below0.01. These means follow the native
new||old convention across samples and action coordinates. Timing includes
observer overhead and is a tiny CPU profile, not a robust performance/speed claim.

[Completed receipt](completed.json) binds clean source0280143, Python3.13.18,
original dataset/normalizer, all counters and saved acceptance/transport hashes.
Native save/load/repeat checks, restored actor endpoint/continuity/KL, complete
Adam ownership/moments, exact reference actor re-export and all32 saved reference
probe points pass. Full frozen preprocessing/bundle and economic-path parity
remain separate requirements before a performance screen. Original failed
families remain closed; future performance registration must include strong SAC,
matched holding, fixed selected/final gates and seeds11/23/37.

Owned session `rl-stocks-cpu-53324f7dd3a6` stopped and termination verified before
offline acceptance. Successful attempt measured0, conservatively charged0.012151119
units. Original failed charge0.001057903 remains: study total0.013209022/0.25,
original baseline40.21369700555672 unchanged. Project40.226906028/150, remaining
109.773093972, project teardown130; fresh account544.61, rate0, assignments0.
Account minimum150/teardown170 remain unchanged. No recovery is needed.

Retained artifacts: `artifacts/colab/rl-stocks-cpu-53324f7dd3a6/` contains the
source upload, kernel record, complete transport, extracted initial/final models,
per-update tensors, reference actors, progress, completion, summary and physical
acceptance. [Cleanup](cleanup.json) removed22,259 bytes of superseded diagnostic
progress copies after prefix/record verification against complete retained logs
and results. Protected files rehash unchanged; physical matrix acceptance passes
again after cleanup. The original failed claim, inputs and records remain intact.
Both allocation claims are consumed; do not replay either command.

# Original failed allocation

The first fixed four-case Colab CPU allocation failed before dependency setup
or learning. No PPO/TRPO smoke or profile result exists, and no performance
experiment is admitted.

Owned session: `rl-stocks-cpu-4f2bbd67c096`; clean source3c42cc4. Actual default
kernel Python3.13.18 passed inspection. Post-allocation reported rate0.08 units/hr,
initial/final account544.61. The dedicated bootstrap incorrectly treated notebook
kernel argv (`-f` with a kernel connection file) as user arguments and rejected
execution before extraction. The CLI completed without a useful training export;
bounded artifact download attempts failed, then owned shutdown was verified.
The original controller process terminated with a retrieval error. Kernel
inspection, source upload, controller logs, claim and summary remain retained.

[Failure receipt](initial-failure.json) binds the claim, input archive and kernel
record. Measured debit0; conservative estimated/charged0.001057903 units. Project
usage40.214754909/150, remaining109.785245091. Original project teardown130 and
account floor150/teardown170 are unchanged. Fresh post-cleanup usage reports
544.61 balance, rate0, assignments0. No recovery command is required.

The fix separates the explicit `--worker` subprocess entry from the bootstrap;
notebook kernel arguments no longer reject bootstrap. A pre-filesystem regression
test reproduces notebook argv;18 transport/controller checks pass, with Ruff and
diff checks. The original single-use claim is consumed and has not been replaced.
No automatic retry occurred. Before a separately reviewed execution-only attempt,
preserve this charge and claim and retain the original0.25-unit study baseline
40.21369700555672. Do not rerun the consumed command or reset shared budgets.

## Local frozen preprocessing acceptance

The four learned actors now have separate checksummed frozen research bundles,
including the original normalizer, feature/action/state schemas, data provenance,
simulation defaults and dependency lock. All32 raw-feature/state observations
match the real environment exactly, and all resulting frozen predictions match
their direct reference actor exactly. Original actors and dataset remain unchanged.
[Bound receipt](frozen-preprocessing.json); detailed local proof and bundles are
under `artifacts/trpo-frozen-preprocessing-v1`. This is local inference acceptance,
not cross-machine full-path or trading-return evidence. No new Colab units used.

## Complete local economic-path acceptance

All four frozen actors pass exact direct-reference versus bundled-preprocessing
replay over the full fixed26-symbol training window. Each case has19,136 decisions:
**104 complete episodes /76,544 decisions total**. A separate saved-path verifier
reconstructs every observation, recorded action execution, fill, cash/holding/
protection state, reward, final metric and directional attribution. All pass.
The114 declared numerical import files still match original replay revision3607704.
[Complete local receipt](local-full-paths.json) binds the local result, provenance,
case path digests and independent acceptance artifact.

This establishes local engineering parity, not return superiority, a prospective
result or readiness to trade. No local fitting, checkpoint selection, development
scoring, retired-test access or additional Colab allocation occurred. Complete
lossless paths, case/root results and independent proof remain under
`artifacts/trpo-full-path-local-v1`. Source provenance records the original clean
revision before later transport/checker additions; calculation code was unchanged.

[Local path cleanup](local-path-cleanup.json) removes five superseded progress
summaries after reconciling them against complete retained results. Every protected
path/proof/source/ledger/plan hash remains unchanged. Full remote replay is next,
under the already fixed single-claim CPU inference runtime and shared budgets.
