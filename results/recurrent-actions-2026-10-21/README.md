# Observed recurrent actions and executed positions

Six accepted saved phases, exact source report hashes, original26 development
panel. Nine action/execution/accounting checks pass; no fitting, dataset opening,
provider calls, retired-test access or Colab allocation. Source reports and
metadata remain unchanged. [Exact audit](summary.json), [categories](categories.csv).

| Final masked recurrent group | Requested long magnitude | End-session long exposure | Same requested/executed direction |
| --- | ---: | ---: | ---: |
| Stocks | 74.06% | 67.04% | 93.93% |
| ETFs | 73.15% | 67.44% | 94.77% |

Long requests occur86.60%/83.13% of stock/ETF sessions, short requests13.40%/16.87%.
Average requested short magnitude6.23%/8.15%, executed short exposure6.05%/8.01%.
Flat closes with nonzero requests are classified by explicit priority: rejection,
protective fill, holding limit, approved zero-share target, other. Stocks show
1.03%/2.09%/2.84%/0.07%/0%; ETFs0.48%/0.92%/3.65%/0.17%/0% respectively.
Protective-fill preemptions in rejected rows stay in rejection. Opposite-direction
closes occur0.03%/0%, and exactly-zero policy requests0%/0%.

Thus the policy frequently asks for less than full long exposure even before
execution. Required exits explain some exposure loss, but these observations do
not establish that protection is the dominant cause or that increasing size helps.
Mean magnitude includes zero for requests of the opposite sign. Positions are
measured at the session close after all fills; requested actions came from the
prior completed bar. Price movement, whole shares and intraday activity also
separate those measurements. They are not interchangeable portfolio statistics.

Full-state recurrent final requested long magnitude58.06%/52.29%, executed51.58%/
46.71%, same-direction94.37%/94.23%. Its poor returns cannot be attributed to
wholesale failure to execute requested signs. No causal information/optimizer
claim follows. The old state protocol remains failed and closed.

Per-category P&L is the observed same-session equity change, including carried
positions, all intraday fills and costs; categories partition total net return.
It is not the isolated value of that decision or a counterfactual removal of a
stop. Categories never provide inputs, rewards, labels or checkpoint selection.
Unknown/corrupt reports and broken equity chains reject.

Reproduce without fitting:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_recurrent_actions.py \
 --output /tmp/recurrent-actions.json
```

Next hypothesis candidate: a single fixed unit-magnitude exposure replay of the
saved masked recurrent policy, retaining predicted sign, all three learned
protective components, costs and20-session cap. It must be preregistered, use
unrestricted-path parity before interpretation, recompute account/memory state
on altered paths, report both selected/final on every symbol, and never tune a
threshold from outcomes. This is sizing research, not a resumption of failed
state training. No replay, promotion or new training has occurred here. Project
remains38.014408/150; no active owned runtimes. Robust confirmation and prospective
performance remain unmet.
