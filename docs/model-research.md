# Research on model and learning alternatives

Reviewed2026-10-26 using original papers, authors' implementations and official
library documentation. This is a literature review and proposal queue, not a
registered training experiment or evidence that any candidate beats holding.
The existing Treasury payload work continues separately. No dependency change,
local optimization, new Colab allocation or retired-test evaluation occurred.

## Reported paper outcomes and project status — 2026-10-27

| Paper | Reported result | Transfer limit |
| --- | --- | --- |
| [CrossQ, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf) | Matched or exceeded strong continuous-control methods' sample efficiency with one critic update per interaction; reported wall-clock gains up to4× over comparison methods including REDQ/DroQ. | Benchmark control tasks and larger networks; no evidence of stock returns or4× speedup over our small SAC baseline. |
| [Recurrent model-free RL, ICML2022](https://proceedings.mlr.press/v162/ni22a.html) | Carefully configured recurrent off-policy baselines beat specialized methods in sample efficiency and asymptotic performance on18/21 environments. | Partially observed benchmark tasks; recurrent SAC remains a distinct, untested project candidate. |
| [Deep Momentum Networks](https://arxiv.org/html/1904.04912v3) | On88 continuous futures contracts, the Sharpe-trained LSTM exceeded the best traditional signal's cost-free Sharpe by over2× and retained an advantage through2–3bps costs. | Volatility-scaled futures portfolio and risk-adjusted objective; not twice the stock profit, or validation under our exposure/holding/execution constraints. |
| [Tabular trees, NeurIPS2022](https://arxiv.org/abs/2207.08815) | Across45 datasets, tree-based models remained strong on medium-sized tabular problems around10k samples. | General tabular evidence; does not establish financial predictability or rescue our failed forest feasibility study. |

These are authors' reported outcomes, not independently reproduced project
results. Our small CrossQ implementation passed two4096-transition Colab smokes;
the subsequent screen stopped at a mandatory historical-SAC reproduction failure.
No valid CrossQ trading comparison or promotion follows. Recurrent SAC and direct
cost-aware momentum remain proposals. The separate bounded SAC numerical probe
addresses the reproduction investigation, not returns. See
[CrossQ failure](../results/crossq-integrity-2026-10-27/README.md) and
[numerical diagnostic](sac-first-update.md). Earlier sections below retain the
original2026-10-26 proposal snapshot; their implementation status is historical.

## What the project evidence requires

The latest fixed recurrent control returns19.62% selected/18.48% final versus
30.60% matched holding on repeatedly used2021–2023 development history. Lambda1
worsens the result; its raw-target variance audit does not establish gradient
noise or causation. Prior MLP/random-forest predictors and distributional TQC
also failed their own fixed gates. See [results](../results/README.md).

A useful next technique must address an untested mechanism while preserving
causal data, explicit costs, four action components, whole shares,100% exposure
and the20-session holding limit. Bigger networks or extra steps are not evidence
of a better signal. New algorithms and new rate features must be isolated in
separate comparisons so their effects can be assessed.

## Candidate map

| Candidate | Distinct mechanism | Fit to this system | Main limitation / next evidence |
| --- | --- | --- | --- |
| CrossQ | Off-policy critic batch normalization/renormalization with no target critic networks | Practical next algorithm feasibility candidate; class exists in installed SB3-Contrib2.7.0 | Different mechanism from failed TQC; no stock evidence; capacity/runtime/frozen-statistics checks first |
| Recurrent SAC or recurrent TD3 | Sequence-based off-policy replay with separate actor/critic memory | Tests memory plus replay, which feed-forward SAC/TD3 and recurrent PPO do not isolate | Higher integration cost: sequence replay, hidden-state burn-in, reset/timeout masks, checkpoint acceptance |
| Gradient-boosted signal baseline | CPU-friendly nonlinear tabular predictor, potentially a separately defined ranking task | Could test whether useful signal exists before another RL allocation | Prior random forests/MLPs failed; no automatic tree-count/feature/target grid or warm-start rescue |
| Direct cost-aware momentum model | Learn a position signal directly against an economic objective | Possible independent benchmark for signal versus controller quality | Futures/volatility-scaled results do not transfer to our single-stock constraints or raw-return goal |
| Better uncertainty reporting | Paired seed comparisons, intervals and evaluation accounting | Necessary for interpreting any future passing screen | Does not improve a strategy by itself; three seeds and correlated stocks remain weak evidence |

## CrossQ: strongest practical next algorithm proposal

Bhatt et al., ICLR2024, introduce joint current/next-state critic batch statistics,
remove target networks and report sample-efficient continuous-control results at
update-to-data ratio1. Their evidence is continuous-control benchmarks, not equities.
[Original paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf).

**Project interpretation:** this tests a different critic-learning mechanism from
TQC's truncated quantiles. Our saved-critic diagnostics do not prove the specific
normalization problem studied in that paper; current SAC has no such batch layers.
CrossQ is a candidate to test, not a diagnosed repair or a way to guarantee shorts.

The installed SB3-Contrib2.7.0 contains CrossQ, but our algorithm registry does not
yet enable it. Official documentation supports continuous Box actions and states
that recurrent policies are unsupported. Default critic layers are1024×1024,
compared with2048×2048 in the paper; installed actor defaults are256×256. This is
substantially larger than our64-wide research networks. Low update-to-data ratio
relative to other papers does not imply lower credits than our current PPO.
[Version-matched documentation](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/crossq.html).

Before training: fix capacity-matched SAC controls and the comparison budget;
profile suitable Colab hardware; verify actual batch-renormalization counters,
optimizer state, replay restoration and immutable evaluation statistics. Frozen
prediction must not update running means/variances. Do not add Treasury inputs
or change reward/execution at the same time. No CrossQ parameters or run are
registered by this review.

## Recurrent off-policy learning: memory with experience reuse

Ni, Eysenbach and Salakhutdinov, ICML2022, study recurrent model-free baselines in
partially observed tasks. Their implementation choices include separate actor/
critic recurrent encoders and off-policy learning. The study finds strong results
across its POMDP benchmarks; these are not trading tests.
[Paper](https://proceedings.mlr.press/v162/ni22a.html),
[authors' implementation](https://github.com/twni2016/pomdp-baselines).

**Project interpretation:** recurrence alone has already been tested, but combining
it with SAC/TD3 replay is a distinct hypothesis. Our recurrent PPO already uses
separate actor/critic LSTMs, so that architectural advice is not a new change.
A future candidate would need contiguous sequence replay, causal previous-action/
reward semantics, correct initial/burn-in memory and independent target-network
memory. Randomly treating transitions as independent LSTM starts is insufficient.

This is higher engineering risk than CrossQ. Old research repositories must not
force an older interpreter or dependencies into the current Colab setup. Require
serialization/frozen-sequence parity, replay/timeout/boundary tests and actual
Colab smoke acceptance before any performance screen. No context-length grid is
proposed, and a failure would close the fixed candidate.

## Boosted trees: a signal baseline, not an assumption of an edge

Grinsztajn, Oyallon and Varoquaux, NeurIPS2022, find strong tree-based performance
on medium-sized tabular benchmarks. Their dataset selection excludes time-series/
streaming tasks, so it does not establish performance on this market history.
[Paper](https://arxiv.org/abs/2207.08815).

Microsoft's Qlib offers a finance-oriented research workflow and official
LightGBM benchmarks. Those examples provide implementation/evaluation references,
not validated US-stock predictions for this simulator.
[Qlib paper](https://arxiv.org/abs/2009.11189),
[official LightGBM examples](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/LightGBM).

**Project interpretation:** gradient boosting differs from our failed random
forest, but that distinction alone does not justify training. A separate,
chronologically purged predictive feasibility study would need a stated reason
for its features/target, fixed baselines and completed labels. If ranking is
studied, changing the objective must be declared beforehand; positive rank IC
cannot retroactively override the old failed MSE gate. A portfolio-ranking result
also cannot replace the required matched individual-stock/ETF return comparison.

## Direct momentum objectives and turnover

Lim, Zohren and Roberts,2019, introduce Deep Momentum Networks, jointly learning
trend and sizing using a Sharpe objective and studying turnover regularization.
Their experiments use88 continuous futures contracts in a volatility-scaling
framework. This differs from our assets, accounting and objective.
[Original paper](https://arxiv.org/abs/1904.04912).

**Project interpretation:** a direct signal-to-economic-objective baseline could
help distinguish forecasting weakness from controller optimization. Sharpe is not
cumulative return, however, and volatility targeting may demand leverage our100%
cap forbids. A future benchmark must retain costs/short borrow/holding constraints.
Net-equity rewards already subtract costs; an extra turnover penalty must be a
separately tested preference, not an unnoticed second subtraction. These papers
do not authorize replacing learned stop/target outputs with fixed exits.

## Evaluation techniques that matter immediately

Agarwal et al., NeurIPS2021, show how few-run RL point estimates can support
unreliable conclusions and recommend interval estimates and robust aggregate
reporting. Their examples are RL benchmarks, not financial confidence guarantees.
[Paper](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).

Bailey et al.'s backtest-overfitting work explains the selection risk from trying
many configurations. It reinforces retaining unsuccessful trials and recording
selection effort.
[Authors' manuscript](https://carmamaths.org/jon/backtest2.pdf).

**Project interpretation:** retain fixed selected/final and multi-seed gates, add
paired uncertainty summaries when a candidate reaches confirmation, and preserve
prospective evidence. Stock returns share market shocks: do not treat26 symbols
as26 independent experiments or naively bootstrap individual daily observations.
Any date-block/cluster resampling rule must be fixed before interpretation.
Three seeds cannot by themselves establish tight confidence or repair repeated
reuse of2021–2023. Never turn the retired test back into a model-selection set.

## Recommended sequence

1. Finish Treasury readiness-matched payload acceptance and its separately fixed
   feature-only comparison; keep the original recurrent algorithm unchanged.
2. Prepare a distinct CrossQ feasibility/cost review on unchanged baseline data,
   with capacity-matched SAC controls and frozen-normalization acceptance.
3. Consider recurrent SAC/TD3 only after its replay/memory implementation is shown
   compatible and its higher engineering cost is justified.
4. Keep boosted/direct-signal studies as separately motivated alternatives, not
   an automatic model/indicator grid after any failure.

This order is a project judgment based on implementation fit and existing negative
results. None of the cited papers proves these choices will beat holding, fix the
short side, or supply the missing independent market evidence. The shared budget,
account floor, Colab-only training and closed-family rules remain unchanged.
