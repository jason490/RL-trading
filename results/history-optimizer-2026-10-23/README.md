# Saved history-policy optimizer and parameter audit

Four accepted selected/final models restored on CPU, exact metadata and published
summary/model hashes verified. All parameters and Adam moments finite, shapes and
ownership correct. No fitting, optimizer step, dataset/provider access or Colab
allocation. Source files remain unchanged. [Summary](summary.json), [updates](updates.csv).

| Arm | Selected Adam updates | Final Adam updates | Selected→final parameter L2 norm | Displacement L2 |
| --- | ---: | ---: | ---: | ---: |
| Full |15,800|20,000|29.2802→29.7052|3.7433|
| Recent |9,800|20,000|28.9986→29.8936|5.8829|

Selected on-step checkpoints precede that rollout's optimizer update. Final100k
has20,000 actual Adam updates and1000 nominal PPO epochs. Different comparison
intervals (4,200 versus10,200 updates) prohibit interpreting larger displacement
as evidence of less stable optimization. Most displacement is in actor recurrent
weights and actor dense heads in both arms. Norm growth is not proof of overfit,
optimality, a capacity limitation or a causal failure mode.

Recent logged approximate KL0.02468 selected→0.03804 final, clip fraction0.2250
→0.2873, Gaussian mean standard deviation0.9170→0.8132. Full KL0.02803→0.03695,
clip0.2609→0.2642, standard deviation0.7672→0.7554. These are sparse last-rollout
logs, not every update or a causal diagnosis. Similar KL growth in both arms
cannot establish why development behavior differs. Prior fixed KL guard failed
absolute gates; its family remains closed with no KL/lr/epoch grid.

A direct recurrent PPO Adam L2 comparison was not found in current experiment
configs. Prior supervised/SAC weight decay studies do not establish its effect
on this on-policy recurrent learner. A single fixed coefficient could test a
new hypothesis, but these norms do not prove it will help. Freeze controls,
initialization, cost/return/risk gates, failure stop and resource budget before
any such training. No failed history/capacity family extension or coefficient grid.

37 focused optimizer/fit checks pass, including exact tensor displacement,
immutability and wrong name/shape/dtype/nonfinite rejection; old fingerprint and
fit tests pass. Legacy `diagnose_cash_optimizer.py` keeps its default cash-study
interface; `--study training-history` selects these accepted sources. Models and
original reports retained; project38.68329334/150 unchanged, last official account
349.27 zero usage and all prior owned sessions stopped.
