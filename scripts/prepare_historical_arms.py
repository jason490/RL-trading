"""Prepare fixed historical arms after source/overlap acceptance; no training."""
import json
from pathlib import Path
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.data.generation.history import prepare_arms
from rl_stocks.data.storage.immutable import load


def main():
    base_id='672a21621091a56c1c7beb30557464e7edd8df38874494659bab2c83738e1bf9'
    earlier_id='f44c7b0831d8322c029acf0229b754221d53cc20f8fc641a24afa4d3c1fc093b'
    evidence=Path('results/historical-regimes-2026-10-20')
    audit=json.loads((evidence/'audit.json').read_text());overlap=json.loads((evidence/'overlap.json').read_text())
    if audit['snapshot']!=earlier_id or not audit['accepted_source_consistency'] or not overlap['passed']:
        raise ValueError('accepted immutable source and overlap required')
    base=read_dataset(Path('data/datasets')/base_id);bars,_=load(Path('data/raw/historical-regimes')/earlier_id)
    source=dict(base_dataset=base_id,earlier_snapshot=earlier_id,source_audit_sha256=digest(audit),overlap_audit_sha256=digest(overlap))
    arms=prepare_arms(base,bars,source);summary={}
    val=lambda d:{k:v for k,v in d['episodes'].items() if v['split']=='validation'}
    for arm,payload in arms.items():
        assert val(payload)==val(base) and payload['normalizer']==base['normalizer']
        assert all(payload['episodes'][k]==e for k,e in base['episodes'].items())
        identity=digest(payload);path=Path('data/datasets')/identity;write_once(path/'dataset.json',canonical(payload));assert read_dataset(path)==payload
        train=[e for e in payload['episodes'].values() if e['split']=='train']
        summary[arm]=dict(dataset=identity,train_episodes=len(train),train_rows=sum(len(e['bars']) for e in train),validation_identical=True,normalization_identical=True,original_episodes_identical=True,expected_etf_episode_fraction=payload['manifest']['history_augmentation']['expected_etf_episode_fraction'])
    report=dict(arms=summary,training_performed=False,test_included=False,source=source)
    write_once(Path('artifacts/historical-regimes/prepared-arms.json'),json.dumps(report,indent=2).encode()+b'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
