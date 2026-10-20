# SAC training diversity

The [fitted-period replay](../results/training-fit-2026-10-20/README.md) found
that one saved full-state SAC policy approaches holding on fitted history but
fails development. A broader training pool is a generalization hypothesis, not
a promised remedy or proof of overfitting. Earlier PPO expansion failed; retain
that negative evidence and the survivor/revised-history limitations.

## Cached data audit

Original dataset 672a2162… contains 26 training episodes/70,200 bars; expanded
c2e6c58c… contains 111 training episodes/298,946 bars (107 stocks plus four broad
ETFs). All original training and 26 validation episodes are exactly equal in
expanded data, including bars/features/ATR and provenance. Existing purge and
verified-source flags remain enforced; no test split or post-2023 bars enter.
Some listings start later: ABBV begins March 14, 2013 after feature warmup.
Do not claim every symbol has a complete 2010 history.

The source normalizers differ. New immutable dataset cdb93a33… retains expanded
episodes/provenance and replaces preprocessing with the original training-only
normalizer, recording both source identities and normalizer hash. Neither input
is modified, and no fitting occurs. EA/AVB/EQR exclusions and separate partial
Alpaca histories remain excluded; no new download or broker call is required.
Current sector counts are annotations, not historical classifications or inputs.
[Full audit](../results/sac-diversity-data-2026-10-20/summary.json).

## Fixed comparison and sampling

[Protocol](../configs/experiments/sac_diversity.json) fixes two fresh original/
expanded SAC cells: full21 observations, standard initialization, initial alpha1
with automatic adaptation, existing64×64 parameters/reward/costs, original
normalization and exact original development episodes. Only the stock training
pool differs. Original control is freshly trained; older50k results are context.

Uniform episode sampling would change ETF probability from4/26 to4/111. Instead
choose stock/ETF groups with fixed22/26 and4/26 probabilities in both arms, then
use a seeded shuffled episode deck per group. Exhaust a group before reshuffling
it; no return labels or future outcomes enter. Choose a start uniformly among
full512-transition windows (128 for smoke). Record actual per-symbol and group
starts/transitions; probability does not imply exact empirical ETF proportions.

Two100k-transition screens are necessary for this declared coverage:50k offers
only about98 full-window starts and cannot traverse107 stocks. At100k about196
starts plus group cycles can cover every stock; both arms receive identical
budgets. Verify every training symbol actually starts before interpreting the
screen, rather than inventing exposure from a configured ticker list. Failure
of coverage is a failed acceptance result, without automatic extra steps or
sampling changes. This does not automatically authorize longer training for
other failed models.

Two1024 CPU smokes must pass actual network, sampler, preprocessing, artifact
and inference checks. Then seed11 screens evaluate every10k and report selected/
final separately. Require2pp selected/final gain over fresh original, no group
final deterioration, at most2pp extra drawdown, and both stock/ETF and median
stock excess at least-2pp in both phases. Failed gate stops this family; no
weighting/sampler/temperature rescue or confirmation. Only a pass permits23/37
matched confirmation, followed by frozen prospective evidence. Future declining
stock labels remain reporting-only, with joint positive total/short results.

## Operations and current status

Existing central fleet, original cumulative150-unit ceiling/130 working stop,
2-hour owned deadlines and one2-unit study allowance including smokes/setup/
idle/downloads/failures/retries. Check actual default Colab Python and balance
before allocation. Retrieve artifacts and verify owned shutdown on every path;
checksum publication before duplicate cleanup. Never use retired test data.

Data audit/build and18 slice/isolation/data tests pass; immutable source files
are unchanged. Sampler/runtime integration and actual Colab smokes/screens are
pending. No new allocation or fitting; project remains36.149107/150.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_sac_diversity.py \
  --output artifacts/sac-diversity/new-data-audit.json
```

The output path must be new. The existing content-addressed dataset is reused
only when exact bytes match; source caches are never overwritten.

## Version19 preflight implemented

`stock_etf_cycle_windows` now uses fixed group probabilities, per-group shuffled
decks and uniform complete-window starts. Explicit seeded resets clear decks;
ordinary episode resets continue the current cycles. Counts use deep copies,
including zero-transition automatic reset starts, so parity checks cannot mutate
saved diagnostics. Screen coverage requires positive transitions, not just starts.
Local Gymnasium/SB3 checks pass without fitting.

Version19 jobs, preallocation/worker payload checks and returned archives pin
data, original normalizer, SAC settings, initial network/alpha records and
sampling summaries. Per-symbol totals and group-cycle balance must reconcile;
every screen symbol must actually trade through at least one transition.
The strict reporter independently validates complete saved accounting, matching
holding paths, initial-network equality and selected/final scores before gates.
Only dataset identity differs in the matched contract.

Eighty-two focused data/sampling/job/archive/gate/fleet/budget/report tests pass.
Both actual immutable smoke payloads validate, and scripts compile. Fixture
coverage/noise/network records do not replace real Colab evidence. Source freeze
and two1024 CPU smokes are next; no new allocation at this preflight checkpoint.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_diversity_jobs.py --smoke \
  --output .runtime/diversity-smoke-jobs.json
PYTHONPATH=src .venv/bin/python scripts/report_diversity.py \
  --state .runtime/diversity-screen-state.json \
  --output artifacts/diversity/screen-report.json
```

Account balance350.42, zero rate and assignments verified before source freeze.
Project36.149107/150. The new study allowance2 units includes smokes and screen,
starting at the unchanged current cumulative ledger; it never resets the project.
