# Saved critic final-update sensitivity

The bounded diagnostic completed without training, new returns, provider data or
Colab allocation. [Result](summary.json), [independent verification](verification.json)
and [reviewed v2 protocol](../../configs/experiments/critic_update_diagnostic_v2.json)
retain exact source/checkpoint/observation identities. The underlying critic
study remains closed and failed; this is descriptive follow-up, not promotion.

## Fixed comparison

All three pairs have50,000 transitions and48,999 versus49,000 actual optimizer
updates. The same256 evenly spaced normalized observations from the accepted
truncatedTQC training replay are used for every model. No observation or outcome
was selected to maximize a difference. Original frozen8c27af4 package code is
verified; all source files, learned weights and optimizer states stay unchanged.

| Model | Actor relative L2 change | Mean absolute exposure-output change | Mean pre-tanh Gaussian KL |
| --- | ---: | ---: | ---: |
| SAC |0.04077%|0.003045|0.006346|
| Untruncated TQC |0.03490%|0.005800|0.001927|
| Truncated TQC |0.03356%|0.001462|0.001133|

All three have zero exposure-sign changes on these256 common inputs. Truncated
TQC's maximum normalized stop-component change is0.040599, with mean0.005385;
its maximum exposure-component change is0.017960. Values are normalized policy
coordinates, not net returns, actual position changes or physical stop distances.
The1,024-step smoke and50k trading results are not recomputed here.

This probe does **not** show an unusually large truncatedTQC actor update relative
to these controls. It does not rule out development-state differences, short
exposure problems, stop/target/whole-share thresholds or diverging execution
paths. Candidate return had fallen2.0042pp across the final update in the accepted
study; these training-replay action/KL measurements do not establish its cause.
No clipping/learning-rate rescue, longer training or reopening of the failed
truncation family is justified by this diagnostic.

## Verification and failure retention

The first attempt rejected before comparisons because `checkpoint-50000.zip`
contains49,000 updates. [Failure](preflight-failure.json) preserves that result.
Actual archive counters and compact-callback source identify
`latest-checkpoint.zip` as the retained48,999-update file for all three arms.
Reviewed v2 corrects only file selection and pins the failed journal; the original
protocol and failure remain, with no automatic restart or changed sample/metrics.

Three numeric tests cover known norms, geometry/schema/finite guards and exact
Gaussian KL/asymmetry. Twelve related checkpoint/gate/numeric tests pass after
the correction (17 related checks passed before execution). The actual worker
completed in0.95seconds, with all six models' complete records unchanged during
inference. NumPy/Python/Torch RNG is restored on exit; the journal is complete,
worker exited0 and no temporary worker remains.

Independent verification uses Torch vector norms and
`torch.distributions.kl_divergence(Normal, Normal)`, without the new numeric helper.
All three tensor norms, action means and Gaussian KL means agree within declared
1e-12/0/1e-10 tolerances, and protected files rehash unchanged. No optimizer step,
normalizer fitting, new price row, retired-test evaluation or Colab unit occurred.
Cumulative project usage remains39.154379/150; the last verified account reading
was348.89 with no assignments. No broker orders or robust winner.

Next inspect the accepted selected/final development execution paths to locate
where protection/position trajectories diverge, before preregistering another
training hypothesis. Such an audit must preserve reporting-only labels, source
hashes, fixed scope and the existing failed-family gates.
