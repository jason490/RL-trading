# Saved GAE target variability — complete

Read-only audit of200 accepted actual training buffers/200,000 transitions from
the failed fixed GAE study. No model replay, market-data access, optimization,
new Colab session or retired-test access. This is diagnostic evidence, not a
new trading model or performance improvement.

## Paired target results

Both traces are recomputed on the same stored rewards, values, reset masks and
endpoint within each buffer. Ratios are lambda1/lambda0.95 population variances;
medians summarize100 buffers per source arm. Different source arms have different
trained policies and paths, so the arms themselves are not paired policy outcomes.

| Source buffers | Median advantage variance ratio | Median return variance ratio |
| --- | ---: | ---: |
| Control | 4.408252 | 2.779849 |
| Lambda1 candidate | 4.018140 | 2.422366 |

All100 buffers in each source arm have higher raw advantage variance with lambda1.
Return variance increases in99/100 control and95/100 candidate buffers. No zero
denominator is encountered; the implementation retains explicit nulls if one occurs.

| Fixed block | Control advantage ratio | Candidate advantage ratio |
| --- | ---: | ---: |
| First50 | 4.292271 | 3.946849 |
| Last50 | 4.878825 | 4.177272 |

These are correlated buffers, not independent statistical replicates. The ratio
compares raw advantage variance, not standard deviation, normalized minibatch
advantages, actual gradient noise, profitability or generalization.

## Endpoint dependence

Lambda1 returns decompose into discounted raw rewards, an endpoint critic term
and an explicit stored-float32 reward-adjustment rounding residual. Mean absolute
raw-reward/endpoint components are0.049585/0.011489 for control and0.049389/0.011913
for candidate. Rounding components average below2.72e−10. The largest per-decision
identity residual is6.11e−16. These values are in discounted log-equity reward units.
They are not variance attribution fractions or evidence of critic bias.

Each source arm has195 time-limit endings and zero true terminations. Lambda1
removes intermediate value terms within segments but retains timeout/end-of-rollout
critic values. It therefore does not provide an unbiased complete-horizon return.
The20-session maximum position holding period remains an internal trading rule;
it does not turn each512-session environment episode into a20-step terminal task.

## Verification and limits

[Protocol](../../configs/experiments/gae_variance_audit.json) was committed before
actual audit, SHA256:6dd450bb05d716524a951b929bbcc32be88ec6a954bdcc840ecf9472ad663b68.
[Summary](summary.json) retains every buffer statistic and fixed-block aggregate.
Accepted source summary bytes, all original NPZ/JSONL evidence and source hashes
are checked; original actual target math is independently revalidated. All221
protected files, including the original project ledger, remain unchanged.

[Independent verification](verification.json) reconstructs both traces using
segmented discount matrices, population variance via centered dot products and
Python medians across all200 buffers. Maximum variance discrepancy1.05e−17;
ratio/block-median discrepancy1.60e−14.25 focused diagnostic, original-buffer and
financial-gate checks pass; new modules/script/tests pass Ruff. Fixtures never
optimize. The actual audit and independent worker both completed successfully.

SB3 normalizes advantages within recurrent minibatches. Their exact shuffled
sequence order and masks are not available in these scalar arrays, so this audit
cannot establish actual normalized gradient noise or causally explain the poorer
candidate return. Bootstrap dependence is an identity, not a calibration test.
No lambda/gamma/optimizer/model/reward grid, extension, confirmation or checkpoint
promotion follows. The failed GAE family stays closed.

No new Colab units: cumulative project39.351004/150, last verified account348.74
idle, all owned sessions already stopped. Original source models/data/arrays and
ledger remain retained. Next research requires a distinct saved-evidence review;
this result does not justify another trace setting or blind additional training.
