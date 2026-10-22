# Recurrent fitted-period replay

The [fixed protocol](../../configs/experiments/recurrent_fit_diagnostic.json) and implementation were committed as6f2e10c before replay. All four accepted cash-study arms, selected and final weights, were replayed on the same original26-symbol eligible training window: decision anchor2017-12-29; next-session executions2018-01-02 through2020-12-02.26 episodes/19,136 execution rows per model,153,088 rows across eight replays.

This is local inference on saved weights, not local training. No fitting, optimizer steps, normalization changes, provider requests, new checkpoint selection, Colab allocation or retired-test access. Cached verified training/development-only data was opened; extra85 training symbols were excluded from this comparison. Recurrent state is carried within episodes and reset per symbol; masked21 inputs, declared exposure transforms, costs, corporate actions and holding rules are unchanged.

## Fitted-period outcome

Each cell reports selected→final net return for separate symbol episodes, not a portfolio. Same-window matched holding is58.15% for stocks and47.15% for ETFs. All eight selected/final policies fail the descriptive requirement that stock/ETF excess and median stock excess each reach−2pp.

| Arm | Fitted stocks, selected→final | Fitted ETFs, selected→final | Fitted mean, selected→final |
| --- | ---: | ---: | ---: |
| Posthoc Gaussian unit | 27.22%→27.22% | 68.43%→68.43% | 33.56%→33.56% |
| Aligned Gaussian unit | −10.69%→0.79% | 1.90%→−4.68% | −8.75%→−0.05% |
| Binary categorical | 3.84%→18.27% | 29.50%→−1.16% | 7.79%→15.28% |
| Cash categorical | 5.02%→19.92% | −6.65%→27.07% | 3.22%→21.02% |

Posthoc has a strong ETF result on this window, exceeding holding by21.28pp, but stocks trail by30.92pp and median stock excess is−27.29pp. Cash final stock/ETF excess is−38.23/−20.07pp, with median stock excess−14.95pp. Binary final stock excess improves versus selected but ETFs deteriorate substantially. These results do not establish fitted-period benchmark success.

## Contrast with development

Cash improves its fitted-window mean3.22%→21.02% while development declines29.24%→6.63%. Both fitted stock and ETF returns improve, even while development group returns worsen. That contrast is consistent with a generalization tradeoff, but weak fitted-period benchmark results remain. Binary fitted mean improves7.79%→15.28% while development34.38%→−1.25%; its fitted stock/ETF behavior is mixed. Aligned Gaussian also improves fitted mean while development declines. Thus development deterioration is not evidence that optimization simply worsened every period.

The posthoc control does not uniformly succeed across groups/periods either: fitted stocks trail holding heavily while fitted ETFs beat it; development stocks beat holding and ETFs nearly match it. Different regimes, durations, initial states and exposure paths prevent raw-return comparisons across periods from proving an optimizer/capacity/overfit cause. Eligible training history is not proof every bar was sampled by the original512-step random training episodes.

No training flag overrides failed development gates or selects/promotes/confirms a model. The cash-choice family remains closed. Longer training alone is not justified by these observations. A distinct bounded policy-update safeguard remains an unproven possible hypothesis, retaining the strongest control and all absolute/multi-seed/prospective gates; no new training is allocated here.

## Integrity and artifacts

[summary.json](summary.json) includes group/per-symbol results, accepted model identity, full Adam fingerprints and all input/source hashes. [groups.csv](groups.csv) provides16 compact rows. Each model's tensors, update counters, full optimizer state and source normalization are unchanged before/after prediction; all protected source files remain unchanged and holding paths match exactly across models. Existing models/metadata match the retained publication hash inventory.

Full derived training JSON/HTML remains in `artifacts/recurrent-fit-2026-10-22/ARM/PHASE/training.{json,html}`. Source models are in the [cash-study release](https://github.com/jason490/RL-trading/releases/tag/recurrent-cash-2026-10-22). Reproduce into new paths with the command in [the category guide](../../docs/training-fit-diagnostic.md).32 focused checks cover explicit panel slicing, future-row/split rejection, memory/reset, execution mapping, optimizer fingerprints and existing contracts.

All12 owned Colab study sessions remain stopped. This replay adds no Colab charge: project38.309006/150; last official account349.56 with zero rate/assignments. Original source/model/failure artifacts and cumulative ledger are retained. No robust/prospective objective achievement.

## Derived charts published

[Diagnostic release](https://github.com/jason490/RL-trading/releases/tag/recurrent-fit-2026-10-22) contains six verified assets: standalone
interactive group-excess comparison,208 derived equity charts in an archive,
compact CSV/JSON, fixed protocol and checksums. Models remain in the original
cash-study release and were not duplicated; raw provider bars/private files
excluded. All server sizes/digests match. Removed only the duplicate6,722,998-byte
report archive after verification; all30 retained hashes unchanged, including
full original derived paths and extracted charts. No additional compute or
model promotion. [Release checks](release-verification.json), [cleanup](cleanup.json).
