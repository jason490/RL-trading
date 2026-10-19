import json
import runpy
import pytest


def test_short_audit_keeps_same_algorithm_variants_separate_and_requires_paired_seeds(tmp_path,monkeypatch):
    audit=runpy.run_path('scripts/audit_short_side.py')['audit']
    episode={'bars':[{'session':'2021-01-01'},
                     {'session':'2021-01-04','open':100.,'close':100.}]}
    monkeypatch.setitem(audit.__globals__,'read_dataset',lambda _: {'episodes':{'A-validation-0':episode}})
    def run(arm,seed):
        root=tmp_path/f'{arm}-{seed}';root.mkdir()
        meta={'status':'complete','algorithm':'ppo','seed':seed,'dataset':'fixture',
              'requested_steps':50000,'simulation':{'initial_equity':100000.},
              'initialization':{'arm':arm}}
        (root/'metadata.json').write_text(json.dumps(meta))
        report={'A-validation-0':{'symbol':'A','split':'validation',
                'rows':[{'session':'2021-01-04','equity':100000.,'quantity':0}],
                'events':[],'metrics':{'net_return':0.},
                'baselines':{'buy_hold':{'metrics':{'net_return':-.1}}}}}
        for name in ('selected-validation.json','validation.json'):
            (root/name).write_text(json.dumps(report))
        return root
    roots=[run('standard',11),run('wide',11)]
    universe={'instruments':[{'symbol':'A','kind':'stock'}]}
    result,rows=audit(roots,universe,tmp_path)
    assert {s['variant'] for s in result['summaries']}=={'standard','wide'}
    assert len(result['summaries'])==4 and len(rows)==4
    with pytest.raises(ValueError,match='duplicate'):
        audit(roots+roots[:1],universe,tmp_path)
    with pytest.raises(ValueError,match='seed sets'):
        audit(roots+[run('wide',23)],universe,tmp_path)
