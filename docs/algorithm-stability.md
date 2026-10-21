# Diagnosing deterioration and comparing algorithms

## Question and observed behavior

A high validation-selected checkpoint does not imply that later training improves
the policy. Compare selected weights with the actual post-update endpoint, keep
all seeds, and distinguish validation selection from independent performance.
The 2024–September 2026 audit interval remains retired.

The indicator follow-up showed large selected-to-final declines. In sector-only
seed 23, net return fell 58.34 percentage points while additional modeled costs
were 6.71 points of initial equity. Time short rose from 8% to 33%; exposure-action
saturation rose from 35% to 61%. Combined-input seeds declined 45–53 points with
similar or lower costs. Costs matter but cannot account for most deterioration.
All nine indicator cells are complete; their source-verified results are
[published separately](../results/indicator-followup-2026-10-16/README.md).

These are different policies evaluated on the same historical validation episodes,
not a live account losing the selected checkpoint's profits. The best model file
is retained separately. Direction, timing and exposure changes explain the saved
path differences more directly than a blanket cost penalty. Calling the training
cause “overfitting,” critic error or entropy collapse requires further evidence.

Checkpoint selection itself also creates a gap: the selected result is the best
of many validation checkpoints, whereas the endpoint is one checkpoint. More
steps add selection opportunities. That alone makes selected results optimistic;
a selected-to-final decline is not proof of overfitting. The new experiment fixes
the evaluation count for all arms and requires improvements in endpoint return
and drawdown as well as checkpoint retention.

## Reproduce the diagnosis

```
PYTHONPATH=src python scripts/diagnose_learning.py ARTIFACT_ROOT... --output artifacts/diagnosis.json
```

Each root contains `runs/RUN/metadata.json` and both saved validation JSON reports.
The command performs no training or new model evaluation. It rejects mismatched
symbol sets, non-validation episode keys, inconsistent initial equity and invalid
actions. It records source-file checksums and reports:

- Selected/final net return, costs, turnover, long/short/flat fractions and fill behavior.
- Four-component action means and saturation fractions (`abs(action) >= 0.95`), plus target-sign switches.
- Annual validation returns with positions and equity carried across year boundaries.
- Changes in recorded-path net P&L, modeled costs and non-cost P&L, all measured against matched initial equity.
- Available training losses, SAC entropy coefficient and PPO update diagnostics.

Adding modeled costs back to a saved path is an accounting decomposition. It is
not a backtest with costs removed: different fills and position quantities would
change that counterfactual. Training loss and entropy are diagnostic proxies and
cannot prove the cause of validation losses. Annual slices are descriptive and
must not select a different checkpoint for each year.

## Bounded PPO, RecurrentPPO and SAC comparison

The [committed protocol](../configs/experiments/algorithm_stability.json) fixes nine
new CPU cells: PPO, RecurrentPPO and fresh SAC controls at 100k transitions, seeds 11/23/37, using
basic inputs. Data, costs, reward, hidden-layer widths, learning rate and discount
stay fixed. Existing algorithm-specific defaults stay intact; no extra reward,
new stock selection, KL tuning or learning-rate search is included.

PPO uses recent on-policy rollouts and a clipped update objective; SAC uses replay
and entropy regularization. This makes PPO a reasonable stability hypothesis,
not a guaranteed fix. The earlier basic-input 50k study recorded PPO mean selected
return 12.97% (seed standard deviation 3.57 points), SAC 15.60% (5.95 points), and
TD3 4.18% (8.90 points). Those historical baseline families all used CUDA and are
context only. The new comparison uses CPU for both algorithms and fresh controls.
See the versioned [PPO](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html)
and [SAC](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/sac.html) documentation.

The fresh SAC controls avoid reusing a different optimization trajectory: this
experiment uses one uninterrupted 100k learn call, with an explicit post-update
endpoint. The earlier learning study used segmented 50k/100k calls. Expanded
training diagnostics also change the package identity; both new algorithms must
share the new source and dependency proofs.

Every 10k steps, save compact validation curves. PPO records approximate KL,
clipping fraction, entropy/value/policy losses and explained variance when present;
SAC retains its actor/critic/entropy diagnostics. Never compare their raw loss
magnitudes as though the objectives were identical. Final model weights and best
weights are separate. The predeclared gate requires mean selected and final return
gains of at least 2 points versus paired SAC, final improvements in at least two
seeds, no mean drawdown worsening above 2 points, and at least a 2-point reduction
in the mean selected-to-final return gap. A flat but stable model cannot pass just
by avoiding the stronger model's swings.

No new session starts until the indicator study is finished, retrieved, published,
stopped and its temporary duplicates cleaned. The same cumulative ledger, minute
metering, two-hour allocation deadline and 130-unit working stop apply. Complete
only the nine declared cells; do not extend training because a result disappoints.

## RecurrentPPO memory contract

The user proposed RecurrentPPO before this new experiment allocated a session.
The third arm uses `sb3-contrib==2.7.0`, matching SB3 2.7.0 and the inspected Colab
3.13.16 interpreter. The hash lock adds only the reviewed contrib dependency;
existing dependencies retain their pins. [RecurrentPPO reference](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/ppo_recurrent.html).

The LSTM keeps a learned compressed state, not an exact record of every past bar.
Actor and critic have separate single-layer 64-unit LSTMs and 64x64 MLP heads.
PPO rollout, minibatch, epochs and clipping match the standard PPO arm. The
recurrent arm adds parameters, so improved results would support this model
configuration, not isolate memory from parameter count. A dedicated recurrent
versus standard PPO comparison uses the same stability gate. No post-result
architecture tuning is permitted.

Evaluation carries hidden/cell state through each stock's ordered sessions and
starts with zero memory at each episode boundary. It resets at a new symbol or
episode, not at each trade exit. No state is carried across training/validation
splits, and no future observations initialize memory. Separate evaluation streams
prevent one stock's history from affecting another stock.

Frozen recurrent inference is explicit:

```python
state = None
for index, observation in enumerate(ordered_observations):
    action, state = frozen.predict_recurrent(
        observation, state=state, episode_start=(index == 0))
```

Callers own a separate state pair per stream. A restart must reset and replay a
causal history or restore a validated matching state; silently continuing with
missing memory is rejected. Stateless `predict()` rejects a recurrent bundle.
Shape/nonfinite state rejection, unchanged input memory, multi-step saved-model
parity, reset parity and reversed-symbol-order evaluation are acceptance checks.
Persistent live-state recovery and broker orchestration remain later milestones.

## Report the fixed comparison

```
PYTHONPATH=src python scripts/report_stability.py ARTIFACT_ROOT... --output artifacts/algorithm-stability.html
```

The report requires clean Git source proofs, identical learning implementation and
dependencies, the declared algorithm parameters, identical costs and passive
benchmarks, every 10k validation point, and a post-update 100k endpoint. Partial
families expose their seed curves without a family aggregate or gate. Complete
three-seed groups report selected/final returns and seed standard deviations,
drawdowns, and the selected-to-final gap. Gates are shown only for complete pairs.
The standalone HTML, JSON and CSV record the protocol and do not select on test.

## Completed results, 2026-10-16

All nine cells finished on Colab CPU. Selected/final mean validation returns were
SAC 19.70%/16.50%, PPO 18.69%/16.90%, RecurrentPPO 13.27%/6.13%; buy-and-hold
was 30.60%. All declared gates failed. PPO reduced the selected-to-final gap
from 3.20 to 1.79 points but missed the required return and gap improvements.
The recurrent configuration had a 7.14-point gap and lost to both controls at
the endpoint in every seed. No architecture tuning or additional steps followed.

Saved-path attribution varies: extra costs represent 3.77 of PPO seed 37's
4.57-point decline, but only 2.41 of recurrent seed 23's 11.39-point decline.
The latter also increased exposure saturation from 8% to 47%. This supports
separate turnover and policy-behavior hypotheses, not a universal reward fix or
proof of overfitting. All nine frozen bundles pass sequence/reset parity and
immutable normalization/schema checks. Full local acceptance passed 146 tests.
Every owned session stopped; cumulative conservative usage is 35.389379 units.

See [complete results and next steps](../results/algorithm-stability-2026-10-16/README.md)
and [downloadable graphs/models](https://github.com/jason490/RL-trading/releases/tag/algorithm-stability-2026-10-16).
Do not reuse the retired final-test interval to rescue a failed comparison.

## Recurrent input and memory audit

Six retained100k recurrent selected/final models use legacy12 inputs, omitting9
explicit execution fields.96 fixed synthetic sequence observations show exact
direct-call/stream parity and reset replay; resetting each observation changes
actions, so memory is active. One regression passes, weights/updates/source
hashes unchanged; no fitting/dataset/test/Colab access. Missing explicit state
is a distinct testable hypothesis, not proof of the previous performance cause.
Next freeze a matched recurrent masked21/full21 comparison with feed-forward
control and unchanged absolute/relative gates before smoke acceptance.
[Audit](../results/recurrent-contract-2026-10-21/README.md).
