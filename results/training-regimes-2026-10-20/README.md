# Fitted-period decline/rebound decision audit

No new inference or fitting was necessary: existing checksummed2018–2020
training replay paths from the fixed50k full-state PPO/SAC/TD3 policies supply
all observations. These are not the new100k diversity policies. Original fitted
replay protocol and failed development gates remain unchanged.

Fixed diagnostic:20-session reference-close trailing return≤−10%; report next
20-session return≤−10%,≥+10%, or intermediate. Trailing labels are causal;
future buckets are reporting-only, never sampling/reward/selection labels.
Reference-price returns are gross, omit dividends/costs, and overlapping windows
are not independent trades. Both stocks and ETFs are pooled; no causal algorithm
ranking is justified by these differently initialized/source policies.

| Policy | Short requests after past decline | If future decline | If future rebound |
| --- | ---: | ---: | ---: |
| PPO | 2.25% | 0.00% | 0.78% |
| SAC | 30.40% | 40.63% | 31.44% |
| TD3 | 3.23% | 1.04% | 2.34% |

SAC mean approved exposure after past declines+23.51%, continued declines+6.92%,
rebounds+24.02%. It sometimes reduces/reverses exposure on fitted declining
paths; this is not evidence of robust positive short P&L. Mean forward price
return after past declines is+5.18% among SAC accepted decisions, highlighting
why blindly shorting all trailing declines can be misleading.

There are18,096 eligible overlapping windows per PPO/TD3,18,061 SAC. SAC has35
rejected protective decisions excluded because no approved exposure exists;
prices/horizons remain aligned across these gaps. This means SAC cohort means
are conditional on accepted decisions, not identical samples to other models.
SAC counts:987 past declines,96 future declines,353 rebounds,538 intermediate.
[Exact statistics and source hashes](summary.json).

Three numeric/timing/rejection tests pass. No dataset/retired-test access,
provider calls or Colab allocation; cumulative usage36.263895/150. Source files
remain unchanged. Reproduce with `PYTHONPATH=src .venv/bin/python
scripts/diagnose_training_regimes.py` using a new output path (the script refuses
to overwrite its fixed result). This diagnostic cannot promote a checkpoint.

Next apply the fixed fitted-window replay to the latest100k original/expanded
SAC policies before choosing a new optimization or sampling hypothesis. Do not
infer that fitted behavior transfers to repeatedly used development periods.
