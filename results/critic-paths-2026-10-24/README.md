# Saved selected/final TQC path audit

The audit completed on all26 accepted development episodes, with no models,
training, new simulations or Colab allocation. [Result](summary.json),
[independent verification](verification.json), and
[fixed reviewed protocol](../../configs/experiments/critic_path_diagnostic_v2.json)
retain source identities and the first failed attempt. The critic family remains
closed; this is descriptive follow-up, not a new return or promotion result.

## Reconciled findings

The observed final-minus-selected mean return difference is−2.004248pp. Exact
daily equity-increment differences telescope to each symbol's saved endpoint,
then to stock/ETF groups and the overall accepted result.

| Paired-path measure | Stocks (22) | ETFs (4) |
| --- | ---: | ---: |
| Mean final-minus-selected net return |−2.7041pp|+1.8447pp|
| Mean fraction of sessions with different closing quantity |72.94%|63.49%|
| Mean fraction with different closing side |2.97%|2.87%|
| Mean fraction with different protective-reason sequence |2.24%|1.95%|
| Mean fraction with different fill-reason sequence |9.51%|11.17%|
| Long net contribution change |−2.5417pp|+2.6265pp|
| Short net contribution change |−0.1624pp|−0.7818pp|

Every symbol has changed requested/closing quantities somewhere.19/22 stocks
and all4ETFs have at least one different closing side;18/22 stocks and allETFs
have a different stop/target reason sequence somewhere. All26 per-symbol first
action/quantity/side/reason dates and largest absolute daily P&L differences are
retained, rather than choosing favorable examples. First differences can occur
well apart: for example DIS first differs in protective reasons2021-10-14 and
closing side2022-04-26; GOOGL changes closing side without any protective-reason
sequence change. These examples do not select or justify a policy.

Stock bridge buckets with matching same-day protective reasons sum−2.6403pp;
those with different protective reasons sum−0.0638pp. ETF counterparts sum
−0.9895/+2.8342pp. These are mutually exclusive additive **observations**, not
causal effects. Same-day labels do not capture earlier exits' downstream effects;
a low event-mismatch frequency cannot rule out protection's influence. Once the
account paths diverge, same-date actions also receive different execution-state
inputs. This does not show a removable fee loss, bad fill, exploding update or
a reason to disable required stops. No counterfactual fills were generated.

## Validation and retained failure

The initial attempt rejected a legitimate `gap_invalid_geometry` case with
`approved_target_quantity=None`. [Failure evidence](preflight-failure.json) and
the original protocol/journal remain preserved. Reviewed v2 accepts a null
approved request only when a rejection is recorded. Closing positions still
require whole shares and exact fill/equity chains. All26 episodes, definitions,
measurements and deadlines remain unchanged; the original journal was not reset.

Ten new numeric/schema tests form part of27 passing action/accounting tests.
They cover telescoping P&L, inherited-side buckets, first event mismatches,
recorded gap rejections and invalid dates/prices/fills/equity/endpoints/baselines/
actions. The actual worker completed in2.01seconds, journal complete and process
exit0; source hashes remain unchanged. Independent NumPy equity differencing and
quantity/side/protection counts reproduce all26 records, with maximum bridge
discrepancy1.07e−14pp. No temporary worker remains.

Current official Colab check:348.89 account units,0.00/hour,0assignments. Project
ledger remains39.154379/150. No additional model training, optimization, provider
request, raw dataset access, retired-test row or broker order occurred.

## Next research decision

The two bounded diagnostics do not justify another clipping, learning-rate,
protection-disable or longer-training rescue of this failed critic study.
A distinct information-family feasibility audit is the next candidate: historical
option-implied broad-market volatility, rather than another architecture/step
sweep. Cboe documents [daily VIX closing history](https://www.cboe.com/tradable_products/vix/vix_historical_data).
This is only a source lead, not accepted model input or evidence of better returns.
Before collecting data, fix date-bounded retrieval through2023, historical
methodology/revision and availability assumptions, conservative lagged joins and
coverage gates. No full-current-history download that exposes retired prices;
no training until that source audit and a separate matched protocol pass.
