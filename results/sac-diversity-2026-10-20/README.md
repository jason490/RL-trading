# SAC training diversity — completed, gate failed

Two fixed100k-transition seed11 SAC screens ran on Colab CPU after two1024
smokes, actual Python3.13.16. Worker source5cad3fa stayed fixed; screen
controllerc6f69d4 includes bounded connection recovery. All initial actor/critic/
target hashes, alpha1, original normalization and26 development episodes match.
Four learned actions, costs, reward and all SAC parameters are unchanged.

| Training arm | Selected mean return | Final post-update mean return |
| --- | ---: | ---: |
| Original22 stocks +4 ETFs | 24.35% | 10.38% |
| Expanded107 stocks +4 ETFs | 24.19% | 24.19% |
| Matched holding | 30.60% | 30.60% |

Cumulative net returns across separate2021–2023 development episodes, not
annual returns or an investable portfolio. Original selected peak is60k; its
100k cached on-step score10.15% differs from final10.38%. Expanded selected is
its100k post-update checkpoint. Original older50k studies are not matched controls
for this new sampler/100k comparison.

Expanded final improves13.81pp over original, including stock+13.96pp and
ETF+12.95pp; mean final maximum drawdown improves1.17pp to-32.88%. But selected
loses0.16pp versus original, missing the required2pp improvement. Expanded
stocks26.06%/ETFs13.88% trail holding32.63%/19.47%, with median stock excess
-5.49pp. Both absolute phases fail. No confirmation seeds, extensions, sampler/
weighting rescue or promotion follows this protocol. The late gain is useful
negative/positive evidence together, not robust benchmark success.

## Coverage and sampling

All26/111 training symbols receive positive transitions. Seeded group decks
cover each member before repetition; future outcomes are never sampling labels.
Intended ETF probability is4/26 in both arms, but observed ETF transitions are
15,360/100,000 (15.36%) original and19,968/100,000 (19.97%) expanded. Stock
transitions84,640/80,032, respectively. Empirical exposure differs despite
matched probabilities; do not call this exactly matched realized ETF mass or
attribute all return differences solely to added names. Source survivor/vintage
limitations and ABBV's later onset remain explicit.

## Short side

Expanded earns joint positive total return AND positive net short P&L on0/3
falling stocks: AMZN-12.54% with+$2,355.68 short contribution, DIS-47.16% with
+$5,175.21, NEE-17.78% with-$172.75 per$100k initial equity. Positive short
contribution with a negative total return fails the user's objective. Original
selected succeeds1/3, final0/3. Future decline cohorts are reporting-only,
never observations/rewards/sampling/checkpoint criteria.

## Artifacts and operations

[Exact gates/group/symbol records](summary.json), [10k curves](learning-curves.csv),
[screen frozen restoration](restoration.json), [smoke evidence](smoke-summary.json),
[compute](compute.json), [protocol](../../configs/experiments/sac_diversity.json),
[guide](../../docs/sac-diversity.md),
[interactive comparison](https://github.com/jason490/RL-trading/releases/download/sac-diversity-2026-10-20/comparison.html),
[models and derived reports](https://github.com/jason490/RL-trading/releases/tag/sac-diversity-2026-10-20),
[release verification](release-verification.json), [cleanup](cleanup.json).

All312 smoke/screen selected-bundle prefix predictions are exact, normalization
unchanged and incompatible inputs rejected. All4 owned sessions stopped and
cleanup verified. A smoke CLI failure required one authenticated-absence-proven
same-session retry, without a new allocation or deadline reset. The controller
fix is separately recorded; worker source stayed unchanged. No screen restart.

Study conservative increment0.11478852 units; cumulative36.26389527/150,
remaining113.73610473. Separate account350.31, zero reported rate/assignments
after shutdown. Ledger never reset; setup/idle/retrieval/failures/retries count.
Retired2024–September2026 data stays unopened. No prospective confirmation or
objective completion is claimed.

Post-cleanup report reproduction is byte-identical: [proof](reproduction.json).
Duplicate cleanup removed16 files/248,893,520 bytes, preserving643 hashes.
Fourteen release assets have verified server digests. Saved cost/exposure/short
path diagnosis is next; no further training under this failed protocol.
