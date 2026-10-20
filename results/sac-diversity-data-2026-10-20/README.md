# Cached SAC training-diversity data audit

No downloads, model fitting or Colab allocation. Immutable original and expanded
sources pass checksum reads and source-preservation checks. Both contain only
training/development episodes with the existing20-session purges. Every original
training episode and all26 original validation episodes match expanded data
exactly, including bars, features, ATR and bar provenance.

| Source | Training stocks | Broad ETFs | Training bars |
| --- | ---: | ---: | ---: |
| Original | 22 | 4 | 70,200 |
| Expanded | 107 | 4 | 298,946 |

The expansion adds85 training symbols. ABBV has a later2013 onset; no earlier
bars are invented. Complete-case survivor selection and revised Yahoo history
remain limitations. Actual historical membership/delisting returns remain
unverified; EA/AVB/EQR and partial Alpaca substitutes stay excluded.

Original and expanded normalizers differ. New immutable dataset
`cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b`
keeps the expanded episodes and records the original frozen training-only
normalizer, both source identities and normalization hash. Source files remain
unchanged. There is no refit or change to the original validation panel.

The fixed next comparison preserves stock/ETF sampling mass22/26 versus4/26
instead of accidentally reducing ETF mass to4/111. Seeded per-group cycles and
full windows are specified, but not implemented/verified yet.100k transitions
per fresh arm provide about196 starts, enough to traverse107 stocks under the
declared group rule;50k cannot. Actual coverage is mandatory, never inferred
from configuration, and failure does not authorize extra steps. Prior failed
PPO expansion and all absolute/multi-seed/prospective gates stay visible.

[Exact audit](summary.json), [protocol](../../configs/experiments/sac_diversity.json),
[category guide](../../docs/sac-diversity.md). Eighteen data/slice/isolation tests
pass. Runtime/sampler integration and Colab acceptance remain pending. Project
usage36.149107/150, all owned sessions stopped; overall objective unmet.
