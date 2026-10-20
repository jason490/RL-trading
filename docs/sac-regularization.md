# SAC fixed regularization experiment

The final100k SAC policies beat holding on a fixed fitted window but fail
repeated development. [Fitted evidence](../results/sac-diversity-fit-2026-10-20/README.md)
is consistent with a generalization gap, not proof of its cause. Test one fixed
regularization intervention rather than adding steps or retuning many weights.

## Fixed comparison

[Protocol](../configs/experiments/sac_regularization.json) compares fresh
expanded111-symbol controls/candidates, same original26 development episodes,
original immutable normalization,21 full-state inputs,64×64 networks, net-log
rewardscale1, initial learned alpha1, and100k transitions per cell. Seed11 screen;
seeds11/23/37 confirmation only after all screen gates pass. Two1024 Colab CPU
smokes precede screening. All fitting stays in Colab.

Control uses Adam weight_decay0; candidate1e-4. In installed SB3 2.7.0,
policy optimizer_kwargs applies coupled L2 to actor and critic Adam, including
bias parameters. This is not decoupled AdamW. Critic target is updated by Polyak
averaging and has no optimizer. The independently constructed entropy-coefficient
Adam retains weight_decay0. Reward, costs, entropy target and alpha schedule
remain unchanged; regularization changes the optimization objective.

Actual fresh optimizers and initial actor/critic/target tensors are inspected,
not inferred from requested config. Both initial networks/alpha must match;
optimizer records must show exactly the declared actor/critic decay and zero
entropy decay. No local optimizer updates are needed to verify construction.
Future Colab smoke must verify training, restoration and actual saved optimizer
settings before screens.

## Sampling and gates

Both arms use identical expanded data, stock/ETF group cycles and seeded full512
windows. Require actual per-symbol/group counts to match, and every symbol to
receive transitions. No outcome-based sampling. Retain curves every10k and both
selected/final results; final must be post-update, not cached callback score.

Selected and final candidate gains must each be at least2pp versus control;
final group gains nonnegative and mean drawdown worsening at most2pp. Both
phases require stock/ETF and median-stock excess at least−2pp against holding.
Failure stops this fixed family: no coefficient sweep, extra steps, confirmation
or promotion. Report joint positive total/short P&L separately; no future decline
labels enter learning or checkpoint selection. Multi-seed development success
still requires prospective evidence. Retired2024–September2026 stays unopened.

## Budget and current status

Use existing aggregate fleet metering, cumulative150-unit ceiling/130 teardown,
7200-second allocation deadlines and maximum2 incremental units for this study,
including setup/idle/failure/downloads. Existing ledger36.263895/150 is preserved.
CPU preference follows measured small-network workload; no GPU is allocated.
Save/retrieve models, replay, curves and reports before verified owned shutdown;
verify durable publication before duplicate cleanup.

Protocol plus optimizer-construction acceptance is prepared:6 focused existing/
new tests pass, including actual optimizer settings and equal initial tensors.
No training, provider call, dependency change or allocation has occurred for this
study. Job/payload/fleet/archive/report integration remains required before any
Colab allocation. Keep this distinction explicit; construction tests do not
validate learning or performance.

## Runtime integration acceptance

Version20 fixed plans, controller/worker immutable expanded-payload checks,
initial/final/selected actual optimizer records, checksummed archive contracts
and strict report restoration are implemented. The report reloads both latest
and selected SAC models and compares their actual optimizer/tensor fingerprints
with recorded evidence. Actual symbol/group sampling must match across arms.
Forty focused optimizer/save-load/job/archive/gate/report/recovery checks pass;
actual immutable datasets validate for both smoke tasks. Scripts compile.
No local training was used. Colab smokes remain unexecuted at this acceptance.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_regularization_jobs.py \
  --smoke --output .runtime/regularization-smoke-jobs.json
```

Freeze committed source before running these tasks through the existing metered
fleet. Retain initial construction fingerprints separately from learned final/
selected alpha values. Actual entropy optimizer decay must stay zero in all
three records; invalid or nonfinite final alpha is rejected.

## First smoke failure and configuration fix

The first attempt's control completed1024 transitions, but strict completion
validation detected an unexpected `use_sde=False` inside the recorded plan.
SB3 mutates its supplied policy_kwargs; factory override merging retained a
reference to the caller's nested dictionary. This contaminated both the job
plan and subsequently resolved metadata. The validator correctly rejected the
archive. Candidate was interrupted; both owned sessions stopped and verified.
No screen ran. Failed archives/checkpoints remain retained.

The factory now deep-copies overrides before model construction. A real SAC
construction regression verifies SB3 receives its own mutable dictionary,
caller settings remain unchanged and resolved metadata has no injected field.
Twenty relevant actual-optimizer/job/archive tests pass, no local fitting.
Account350.31, rate0 and zero assignments verified after shutdown. Conservative
project36.277597/150 includes this attempt. Retry must use a newly frozen fixed
source, new owned sessions/deadlines and the original study starting budget;
this was not a budget stop. Do not relax archive validation or overwrite failed
evidence. [Failure record](../results/sac-regularization-2026-10-20/smoke-attempt-1.json).

## Replacement Colab smoke acceptance

Two1024-transition CPU smokes pass on actual Python3.13.16, worker6792d2b.
Actual initial networks/alpha and sampling counts match. Selected/final saved
optimizer records restore exactly;156 frozen prefix predictions pass with
immutable normalization/schema rejection. Both replacement sessions stopped;
all4 attempt sessions have verified cleanup. Project36.297597/150 includes the
failed attempt; account350.29 zero rate/assignments. Fixed100k screen next from
same frozen source and original2-unit allowance.
[Evidence](../results/sac-regularization-2026-10-20/README.md).
