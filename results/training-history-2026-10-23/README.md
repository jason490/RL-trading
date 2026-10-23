# Training-history availability and window-mass audit

Date-only audit of the immutable expanded111 Yahoo training panel. No market
outcomes, model inference/fitting, preprocessing changes, provider calls or Colab
allocation. Existing2021–2023 development values are unread; retired2024+ absent.
Dataset file hash and unchanged-source proof are in [coverage.json](coverage.json).

## Findings

| Candidate source | Stocks / ETFs | Minimum bars per symbol | Expected2020 execution share |
| --- | ---: | ---: | ---: |
| Existing full history |107 /4|1,946 stocks;2,700 ETFs|2.45% stocks;2.43% ETFs|
| Fixed2016 onward subset |107 /4|1,239 each|7.32% each|

Every symbol supports a complete512-transition window in both sources. Full
source spans2010–2020, with different listing/readiness starts. The subset uses
2016-01-01 onward and the existing purged2020-12-02 endpoint; no new bars or
labels. This is a coverage candidate, not an approved training dataset.

A uniformly drawn start does not imply uniformly visited dates. Middle dates
belong to many possible windows; dates near the history ends belong to few.
Expected shares are exact enumeration-equivalent probabilities for a complete
window, assigning each transition to its next-bar execution date. The first bar
is observation-only; the last bar can execute a transition. This follows the
actual sampler's exclusive upper bound `len(bars)-512`. Group means assume
uniform episode frequency within each shuffled stock/ETF deck, not uniform
weight across all111 symbols. Current group probability remains22/26 stocks,
4/26 ETFs. Partial final windows and finite realized samples can differ.

Existing training logs contain symbol/group starts and transitions, not actual
year counts. These expected masses cannot reconstruct what an individual run
visited. No claim that poor2020 visitation caused deterioration or that a more
recent window will improve returns. A recent subset also discards older market
conditions, reduces unique observations and increases repeat exposure.

## Next experiment prerequisites

Before allocating, add measured decision/execution year counts to training logs
with sum/identity checks. Freeze a single full-history versus2016-onward training
comparison, unchanged64-unit architecture/actions/costs/100k budget, common
untrained starts and original26-symbol development dates. Verify immutable
subset construction, original training-only normalization held constant to
isolate composition, feature warmup inherited causally, train-only cutoff and
exact unchanged development episodes. Set numeric selected/final relative,
stock/ETF/median/risk gates and shared allowance before training. Failure closes
that fixed comparison; passing alone only permits multi-seed and prospective
confirmation. Do not choose dates by observed profits or vary years/model/reward
simultaneously. Capacity family stays closed.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_training_history.py \
 --dataset data/datasets/cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b/dataset.json \
 --output .runtime/training-history-coverage.json
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_history_coverage.py
```

11 checks pass: exact brute-force window expectation, boundary/off-by-one cases,
invalid/nontraining inputs, outcome-free sources, unchanged input and unread
development episode. Project38.60166524/150 unchanged; last official account
349.34, zero rate/assignments. No sessions allocated; all prior owned sessions stopped.
