# Research results index

Returns below are mean net total returns from separate matched symbol episodes
on repeatedly reused2021–2023 development data. They are not annualized returns,
a combined portfolio, or independent prospective evidence. No model has passed
the full required multi-seed and prospective acceptance process.

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
predictions pass. Publication/checksummed duplicate cleanup remains.

## Recent diagnostics

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
