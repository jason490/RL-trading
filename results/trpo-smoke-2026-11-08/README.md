# TRPO engineering allocation — 2026-11-08

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
