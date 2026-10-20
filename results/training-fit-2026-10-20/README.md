# Fixed fitted-period policy replay

Three already-trained final policies were replayed without fitting, optimizer
calls or Colab. Source revisions and model hashes are fixed in the committed
[protocol](../../configs/experiments/training_fit_diagnostic.json). Each model
uses full execution observations, seven indicators, 50k training transitions
and seed 11. PPO used wide initialization, SAC/TD3 standard; source revisions
differ, so this is not a controlled algorithm ranking.

Training executions cover January 2, 2018 through December 2, 2020, after a
December 29, 2017 decision anchor. The purged cutoff is preserved. There are 26
symbol episodes and 19,136 execution rows per model. No preprocessing refit,
checkpoint selection or retired-test data is involved. All results are average
cumulative net returns across separate episodes, not annual or portfolio returns.

| Final policy | Training stocks / holding | Training ETFs / holding | Training flags |
| --- | ---: | ---: | --- |
| PPO | 58.98% / 58.15% | 30.84% / 47.15% | Fail |
| SAC | 67.06% / 58.15% | 53.86% / 47.15% | Pass |
| TD3 | 32.29% / 58.15% | 25.88% / 47.15% | Fail |

SAC training mean 65.03% beats holding 56.45%, with median stock excess -0.46pp
and mean maximum drawdown -18.93%. On existing 2021–2023 development paths,
the same final model returns 7.93% versus holding 30.60%; stock excess -23.94pp
and ETF excess -15.70pp fail. This fitted/development contrast is consistent
with a generalization gap, not proof of overfitting, memorization, critic error
or any single mechanism. Period regimes and durations differ. Training success
never overrides the original failed development gate or permits promotion.

PPO training median stock excess -5.18pp and ETF excess -16.30pp fail despite
slightly positive mean stock excess. TD3 training stock/ETF excess -25.85pp/
-21.27pp also fails; there is no universal claim that each model only overfits.

On the two declining stocks in this fitted window, joint positive total return
and net short P&L is PPO 0/2, SAC 2/2, TD3 1/2. These are fitted-period,
ex-post reporting cohorts, not evidence of general short profitability or
labels usable for selection, rewards or sampling.

[Full compact evidence](summary.json), [category guide](../../docs/training-fit-diagnostic.md).
Policy tensors (including SAC learned alpha), update counts, normalization and
all 13 protected source files remain unchanged. Explicit `train` labels are
preserved; the standard evaluator still rejects `train`. Twenty focused slicing,
next-session simulation, isolation, accounting and report checks pass. Complete
derived paths/charts are retained under `artifacts/training-fit-2026-10-20` and
published separately; no raw OHLCV dataset or newly fitted model is produced.

Next examine a fixed SAC training-diversity hypothesis using the existing
verified expanded survivor universe, with original normalization and validation
held fixed and a fresh matched original-universe control. Prior failed PPO
universe studies remain negative evidence; SAC's fitted-period result motivates
investigation, not predicted improvement. Audit dataset coverage and actual
sampling before committing any bounded study or allocating Colab. Keep strict
selected/final stock/ETF/median and multi-seed/prospective gates.

Project usage remains 36.149107/150, all owned sessions stopped. The overall
objective remains unmet.

[Download derived training paths and charts](https://github.com/jason490/RL-trading/releases/tag/training-fit-2026-10-20),
[server asset verification](release-verification.json), [duplicate cleanup](cleanup.json).
