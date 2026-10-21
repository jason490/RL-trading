# SAC discount horizon — Colab smoke acceptance

Fixed gamma0.99 control versus0.95 candidate; fresh identical expanded111-symbol
training and original26 development episodes. Reward, alpha, optimizer, features,
costs/actions, initialization and sampler remain unchanged. This changes the
discounted objective; it is not an extension of interrupted regularization.
[Protocol](../../configs/experiments/sac_discount.json),
[guide](../../docs/sac-discount.md).

Both1024-transition CPU smokes completed on Colab default Python3.13.16, worker
cf41830751554fee4f6f9944a3a38b33a8f0c6a7. Actual gamma/initial-step records pass;
actor/critic/target/alpha and actual sampling counts match across arms. Actual
selected/final saved-model optimizer/gamma/step records restore exactly. New
checkpoint-state hashes/steps/phases and final post-update marker validate.
156 frozen selected-bundle prefix predictions are exact, normalization immutable
and incompatible schemas/feeds/nonfinite inputs rejected.
[Smoke evidence](smoke-summary.json), [restoration](smoke-restoration.json).

Both owned sessions stopped and cleanup verified. Cumulative conservative
37.80559113/150, remaining112.19440887; study increment0.01837772. Account349.96,
rate0/assignments0 after shutdown. Minimum account balance150, teardown170;
project ceiling150/teardown130 remains independent. Original study starting
units37.78721340 and2-unit allowance include all setup/idle/retrieval/retries.
No CLI installation/upgrade was performed.

Fresh paired100k seed11 CPU screening is next from the same frozen source, with
10k curves and exact actual sampling matching required. Fixed relative/absolute
selected/final stock/ETF gates precede any further seeds; prospective evidence
and joint positive total/short P&L still required. No new return result, promotion,
confirmation allocation, retired test access or broker order is claimed.
