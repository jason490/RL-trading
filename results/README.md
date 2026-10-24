# Research results index

Returns below are mean net total returns from separate matched symbol episodes
on repeatedly reused2021–2023 development data. They are not annualized returns,
a combined portfolio, or independent prospective evidence. No model has passed
the full required multi-seed and prospective acceptance process.

## Current research status

[Latest compact-market-context screen](market-context-2026-10-23/README.md) failed:
full selected/final7.70/−3.77%, masked18.73/18.73%, legacy19.62/18.48%, holding
30.60%. Both-control/absolute/risk gates fail; fixed family closed.468 frozen
checks including smokes, six owned sessions stopped. Project39.014378/150,
last official account349.02 zero assignments. 16 release assets and cleanup verified; saved-path diagnosis complete. Robust-return objective remains unmet.

## Latest completed history comparison

[Full/recent history](training-history-2026-10-23/README.md): full selected/final
19.62/18.48%, recent20.95/5.60%, holding30.60%. Selected improvement below2pp;
final and absolute gates fail. Fixed family closed, no confirmation/promotion.
Measured2020 stock exposure1.98%→6.64% confirms intended composition change.
312 frozen checks including smokes; all four sessions stopped. Project38.683293,
account349.27 zero rate/assignments. 17 release assets verified;35.93MB duplicate archives removed with148 retained
hashes unchanged.

## Latest completed optimizer comparison

[Recurrent coupled L2](recurrent-regularization-2026-10-23/README.md): control
selected/final19.62/18.48%, candidate30.45/18.42%, holding30.60%. Selected overall
near holding but ETF/median targets fail; final and relative gates fail. Fixed
family closed, no confirmation or coefficient sweep.312 frozen checks including
smokes, all four sessions stopped; project38.746179/150, account349.23 zero
rate/assignments. 16 release assets verified;35.74MB duplicate archives removed with145 retained
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

- [Rejected native crossover control](l2-crossover-2026-10-23/README.md): direct and wrapper match, source Colab sub-cent differences exceed fixed tolerance; no mixed outcomes.

- [Saved recurrent L2 actions and costs](l2-actions-2026-10-23/README.md): stock deterioration with similar exposure and lower costs; crossover diagnostic pending.

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

[Completed recurrent PPO regularization](recurrent-regularization-2026-10-23/README.md):
fixed Adam decay0/0.0001, same full-history model/data/steps. Candidate selected/
final30.45/18.42% versus control19.62/18.48%; absolute and final gates fail.
Family closed. Publication/verified duplicate cleanup and strict reproduction
complete; no confirmation, sweep or promotion.

[Feature-scale audit](feature-scale-2026-10-23/README.md): outcome-free frozen
training/development z distributions and source arithmetic. Large training tails,
no model fitting or new units. The fixed cap5 preprocessing protocol and immutable
payload are now prepared in [input-clipping](input-clipping-2026-10-23/README.md);
The completed fixed clipping comparison failed its gates; see its dated result below.


## Fixed input clipping completed

[Clipping](input-clipping-2026-10-23/README.md) failed: selected/final24.70/18.57%
versus control19.62/18.48% and holding30.60%. No cap sweep or extension.312 frozen
checks including smokes, four sessions stopped, project38.816469/account349.17.

[Clipping saved-action diagnosis](clipping-actions-2026-10-23/README.md):
increased short exposure/costs and deteriorating net short contribution at final;
104 saved episodes, no causal claim, fitting or additional Colab units.


[Gaussian action-boundary audit](action-boundaries-2026-10-23/README.md): exact
conditional any-component out-of-bounds mass~79% on fixed flat-account training
prefixes; model/Adam/RNG/normalization unchanged, no fitting or new Colab units.
Supports a distinct preregistered bounded-policy hypothesis, not causal proof.

[Completed squashed-action comparison](squashed-actions-2026-10-23/README.md):
explicit transformed Gaussian sampling/likelihood, unchanged full64 tensors/data/
optimizer. Completed100k screen failed; details and verified cleanup in the dated result.


## Fixed squashed-action comparison completed

[Squashed Gaussian](squashed-actions-2026-10-23/README.md) selected/final15.47/5.14%
versus native19.62/18.48% and holding30.60%. Fixed gates fail; family closed with
no extension or transform/noise grid.312 frozen checks, four sessions stopped;
project38.903541/account349.10. Publication and saved-path diagnosis next.


[Squashed saved-action diagnosis](squashed-actions-diagnosis-2026-10-23/README.md):
both long/short contributions deteriorate at final while short exposure and costs
increase;104 saved episodes, no causal claim, fitting or additional Colab units.

[Compact market-context coverage](market-context-2026-10-23/README.md): SPY return/
volatility and own-relative20-session features are causally available on all
298,946 train/19,058 development rows.60 checks pass; no fitting, new model,
performance score or Colab allocation. Matched dataset/model study remains pending.


[Market-context smokes accepted](market-context-2026-10-23/README.md): three512
CPU Colab runs,234 exact frozen predictions and actual mask rejection, models/Adam/
mapped initialization/sampling restored, three owned sessions stopped. Project
38.934668/account349.08. Fixed100k three-arm plans prepared, no return result yet.


[Market-context saved-action diagnosis](market-actions-2026-10-24/README.md):
full model deterioration mainly reflects reduced long contribution even as short
exposure decreases.156 saved episodes, no causal claim, fitting or extra units.

## Pending distinct information-family audit

[SEC source-schema/publication review](filing-source-review-2026-10-24/README.md)
identified archive reprocessing, segment-aware numeric identity and publication
uncertainty.48 focused checks pass. First2020q1 strict failure retained; original
filing identifies the2024 label as a forecast. Explicit-quarantine inspection
completed3,096,249 numeric keys with3,103 forward-period exclusions. Causal
readiness remains unverified; no financial model inputs or training. This is
feasibility work, not a new return result or a promoted model.
