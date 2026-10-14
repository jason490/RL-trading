"""Reproducible fixture/accounting/report audit. No learned policy or network."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from rl_stocks.core.contracts import Bar, Request
from rl_stocks.core.provenance import source_identity
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.accumulation.backfill import accumulate
from rl_stocks.data.generation.datasets import generate, read_dataset
from rl_stocks.environment.execution.simulator import Simulator, SimulationConfig
from rl_stocks.evaluation.metrics import metrics
from rl_stocks.evaluation.runner import baselines
from rl_stocks.reporting.html import report
from rl_stocks.runtime.colab.budget import Ledger


def main():
    snapshot=accumulate(SyntheticProvider(seed=11),Request(),'data/raw',verify_calendar=False)
    dataset_path=generate(snapshot,'data/datasets',allow_unverified=True)
    dataset=read_dataset(dataset_path)
    output=Path('artifacts')/'offline-audit'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    results={}
    config=SimulationConfig()
    for name,episode in dataset['episodes'].items():
        if episode['split']!='validation':
            continue
        bars=[Bar(**b) for b in episode['bars']]
        sim=Simulator(config)
        rows=[]
        for i,(previous,current) in enumerate(zip(bars,bars[1:])):
            # Fixed alternation is an accounting fixture, not a model strategy.
            exposure=.25 if (i//20)%2==0 else -.25
            info=sim.advance([exposure,0.,0.,-.5],previous,current,episode['atr'][i])
            info.update(session=current.session,price=current.close)
            rows.append(info)
        results[name]={'rows':rows,'events':sim.portfolio.events,'metrics':metrics(rows,sim.portfolio.events),
            'baselines':baselines(episode,config),'symbol':episode['symbol'],'split':'validation'}
    report(results,output/'report.html','Synthetic accounting audit — no trained policy; no market evidence')
    metadata={'snapshot':snapshot.name,'dataset':dataset_path.name,'seed':11,'simulation':asdict(config),
        'source_hash':source_identity(),'policy':'Fixed alternating fixture; not trained','split':'validation','dataset_manifest':dataset['manifest'],
        'colab':{'sessions_provisioned':0,'units_allocated':0,'balance_available':False,'cleanup':'No owned session to terminate'}}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    Ledger('.runtime/usage-ledger.json').save()
    print(json.dumps({'snapshot':str(snapshot),'dataset':str(dataset_path),'report':str(output/'report.html')},indent=2))

if __name__=='__main__':main()
