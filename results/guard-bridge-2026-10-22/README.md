# Exact saved-fill overnight/intraday reconciliation

Method committed as96dddd9 before actual immutable-bar diagnostic. Four accepted
phases/104 episodes; strict dates/prices/metrics/holding/directional accounting
checks pass. Every policy and holding session reconciles; worst error
1.28e-10 dollars. Exact accepted source hashes remain unchanged. No fit, new
prediction, provider calls, Colab or retired-test evaluation.
[Method](../../docs/pnl-bridge.md), [full compact totals](summary.json),
[group components](groups.csv). No raw bars in these artifacts.

## Final guarded excess versus holding

All numbers are percentage points of original capital, equal-symbol means.

| Component | Stocks | ETFs |
| --- | ---: | ---: |
| Inherited overnight price move |−1.4255|−7.6351|
| Inherited intraday price move |−0.0370|−1.2040|
| Actual fill-to-close inventory adjustment |−0.2247|−1.5044|
| Dividends |−0.3170|−0.4841|
| Fees |−0.9332|−0.8095|
| Impact |−1.8664|−1.6190|
| Borrow |−0.0456|−0.0980|
| Net excess |−4.8494|−13.3541|

Guard selected ETF overnight gap−1.2071pp worsens to−7.6351pp final. Yet
control final overnight−0.3202pp is less negative than guard despite weaker total
performance: control fill-to-close correction−10.3289pp versus guard−1.5044pp.
An isolated favorable component does not define a better strategy. Final stock
control overnight+3.4473pp also does not offset other losses/costs.

The residual noncost ETF gap−10.8276pp now reconciles overnight−7.6351,
inherited intraday−1.2040, fill adjustment−1.5044 and dividends−0.4841. These
are identities, not independent trading effects: inherited intraday and fill
adjustment offset changes in actual inventory. No chronological OHLC path is
invented. Fee/impact/borrow signs use actual fills; splits/dividends process
before fills, and fractional cash-in-lieu is included in overnight identity.

## Interpretation and next action

The ETF miss is not principally rising costs or frequent final protective fills.
Actual overnight exposure differences are substantial. This does not prove that
relaxing20-session holding exits or banning shorts would fix the model. Next
partition the exact overnight difference by inherited side and preceding actual
exit reason, preserving all fixed symbols/phases. That descriptive step can
distinguish repeated flat-after-holding gaps from short-direction errors before
a distinct preregistered intervention. No model/feature/reward/rule change or
failed-family extension is authorized by this diagnostic alone.

40 focused checks passed before committed method. All sessions remain stopped;
project38.39201565/150, last official account349.51 unchanged.
