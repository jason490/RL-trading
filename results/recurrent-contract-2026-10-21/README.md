# Retained recurrent PPO contract audit

All three original100k seeds11/23/37, selected and final weights: six models.
Actual saved observation spaces are12-dimensional, with7 named market features
and5 legacy portfolio fields, confirmed against checksummed frozen manifests.
They omit9 explicit execution fields now available to full-state policies:
held stop/near/far distances, protection presence, partial-exit state, previous
three protective actions and whole-share granularity. Memory could infer some
state; missing explicit fields do not prove the cause of weak returns.

Actual actor LSTMs have64 hidden units,1 layer and separate critic LSTMs.
For each model,16 fixed synthetic observations produce exact action parity
between the evaluation PredictionStream and direct recurrent calls. Independent
streams reset and replay exactly. Resetting every observation instead changes
actions (maximum absolute delta0.319–1.119 across these models), demonstrating
active memory on these probes, not useful trading signals. Evaluation creates a
new stream per symbol episode. Original frozen sequence acceptance remains in
[the stability study](../algorithm-stability-2026-10-16/bundle-verification.json).

Weights, update counters, model timesteps, metadata and manifest hashes remain
unchanged. No fitting, dataset opening, provider calls, test evaluation, Colab
allocation or credits. One prediction-stream regression test passes; actual
six-model assertions cover96 sequence observations plus reset replays. These
probes do not prove correctness for every observation or prospective performance.

[Exact records and source hashes](summary.json). Reproduce from retained models:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_recurrent_contract.py \
 --output /tmp/recurrent-contract.json
```

The earlier recurrent selected/final means13.27%/6.13% versus holding30.60% remain
failed evidence. No retrospective promotion. A distinct next hypothesis can test
explicit full execution state with fresh masked21/full21 recurrent arms, plus a
matched feed-forward control. Freeze dataset, preprocessing, sampling, PPO rollout,
steps, gates and costs first. LSTM versus feed-forward adds parameters, so it is
not a parameter-count-matched proof of memory. Actual smoke acceptance must
verify21-input save/load/frozen state/reset parity before any fixed screen.
Existing cumulative37.897158/150 and account150/170 safeguards remain unchanged.
