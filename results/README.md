# Research results index

Returns below are mean net total returns from separate matched symbol episodes
on repeatedly reused2021–2023 development data. They are not annualized returns,
a combined portfolio, or independent prospective evidence. No model has passed
the full required multi-seed and prospective acceptance process.

## Latest completed history comparison

[Full/recent history](training-history-2026-10-23/README.md): full selected/final
19.62/18.48%, recent20.95/5.60%, holding30.60%. Selected improvement below2pp;
final and absolute gates fail. Fixed family closed, no confirmation/promotion.
Measured2020 stock exposure1.98%→6.64% confirms intended composition change.
312 frozen checks including smokes; all four sessions stopped. Project38.683293,
account349.27 zero rate/assignments. 17 release assets verified;35.93MB duplicate archives removed with148 retained
hashes unchanged.

## Recent completed comparisons

| Study / arm | Selected | Final | Interpretation |
| --- | ---: | ---: | --- |
| [Cash-study posthoc Gaussian control](recurrent-cash-2026-10-22/README.md) |32.80%|32.80%|Strong prior one-seed control; absolute ETF/other gates remain unmet |
| [Update-guard fresh control](recurrent-guard-2026-10-22/README.md) |19.62%|18.48%|Fresh pinned initialization; matched guard control |
| [Update-guard candidate](recurrent-guard-2026-10-22/README.md) |28.07%|24.44%|Relative improvement, absolute gates fail; family closed |
| Matched buy-and-hold |30.60%|30.60%|Same26 symbol episodes and costs |

Different studies can have different untrained tensor starts despite the same
seed label. Compare candidates with their own matched controls, not as if every
row were an identical initialization. Best interim checkpoint results do not
establish stable final or multi-seed performance.

## Latest completed feature experiment

[Session-context study](session-context-2026-10-22/README.md) completed three
matched100k recurrent PPO runs. Full selected/final26.42/16.08%, masked27.66/10.76%,
legacy19.62/18.48%, holding30.60%. Both relative comparisons and absolute targets
fail; no confirmation/promotion. All six smoke/screen sessions stopped,468 frozen
predictions pass. Publication and checksummed duplicate cleanup are verified; models/reports are linked by the study README.

## Latest completed capacity experiment

[Recurrent capacity](recurrent-capacity-2026-10-22/README.md): both100k seed11
runs complete. Control64 selected/final19.62/18.48%; larger12828.15/13.43%;
matched holding30.60%. Larger selected improvement fails to persist; final and
absolute gates fail. Fixed family closed, no extension/confirmation/promotion.
312 total frozen checks pass; all four sessions stopped. Project38.601665/150,
account349.34 zero rate/assignments. Strict evidence/curves retained;15 release assets verified and45.88MB duplicate
publication archives removed with142 retained hashes unchanged.

## Recent diagnostics

- [Saved history optimizer/parameter audit](history-optimizer-2026-10-23/README.md): finite restored Adam states; displacement and KL are descriptive, not causal proof.

- [Training/development saved action comparison](history-fit-actions-2026-10-23/README.md): similar final short exposure, different payoffs; descriptive only.

- [Full/recent saved-model common training-window replay](history-fit-2026-10-23/README.md): all four benchmark flags fail; no fitting or new units.

- [History-study saved actions and costs](history-actions-2026-10-23/README.md)

- [Capacity selected-to-final actions and costs](capacity-actions-2026-10-23/README.md)

- [Saved actions and costs](guard-actions-2026-10-22/README.md)
- [Exit and inherited-position partitions](guard-exits-2026-10-22/README.md)
- [Exact overnight/intraday P&L bridge](guard-bridge-2026-10-22/README.md)
- [Overnight gaps by preceding exits](guard-overnight-2026-10-22/README.md)

## File organization

Each dated study folder contains a README, compact machine-readable results,
verification and limitations. Category methods live in [docs](../docs/README.md),
fixed numeric definitions in [experiment configs](../configs/experiments/), and
reusable code in [the package](../src/rl_stocks/). Large models, raw data,
transports and derived HTML stay in ignored artifact/data directories and, where
published, checksummed GitHub releases linked by the study README. Historical
studies remain in their dated directories to preserve source and artifact links.

The detailed chronological decision record is [the research log](../docs/research-loop.md).

[Training-history coverage](training-history-2026-10-23/README.md) audits date-only
availability and expected window year mass. Both full/recent sources support111
symbols; expected2020 mass2.4% versus7.3%. No training or performance claim;
actual year logging and a frozen matched protocol remain prerequisites.

[Prepared recurrent PPO regularization](recurrent-regularization-2026-10-23/README.md):
fixed Adam decay0/0.0001, same full-history model/data/steps.138 integration checks
and actual untrained construction pass. Actual Colab smokes accepted with exact saved optimizer/frozen proof;
runtime approval passes,100k screen next. No research performance or promotion.
