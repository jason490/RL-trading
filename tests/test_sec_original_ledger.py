import csv
import hashlib
import importlib.util
import io
from pathlib import Path
from zipfile import ZipFile

import pytest


def module():
    spec=importlib.util.spec_from_file_location(
        'sec_original_ledger',Path(__file__).parents[1]/'scripts/build_sec_original_ledger.py')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fixture(tmp_path,instance='original.xml'):
    quarters=[f'{y}q{q}' for y in range(2009,2024) for q in range(1,5) if (y,q)>=(2009,2)]
    fields=['adsh','cik','form','period','filed','accepted','instance','name']
    entries=[]
    for i,q in enumerate(quarters):
        rows=[]
        if i==0:
            rows=[dict(adsh='0000000001-10-000001',cik='2',form='10-K',period='20091231',
                       filed='20100201',accepted='2010-02-01 12:00:00.0',instance=instance,name='Older issuer'),
                  dict(adsh='0000000001-10-000002',cik='3',form='10-K',period='20091231',
                       filed='20100201',accepted='2010-02-01 12:00:00.0',instance='other.xml',name='ABC lookalike')]
        text=io.StringIO()
        writer=csv.DictWriter(text,fields,delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
        path=tmp_path/(q+'.zip')
        with ZipFile(path,'w') as z:
            z.writestr('sub.txt',text.getvalue())
        entries.append(dict(quarter=q,path=str(path),archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    state=dict(schema='sec-quarter-collection-v1',status='downloaded_only',planned_quarters=quarters,completed=entries)
    discovery=dict(discovery_only=True,historical_mapping_verified=False,source_sha256='fixed',candidates={'ABC':[{'cik':'1'}]})
    specification=dict(source_dataset='dataset',current_candidate_source_sha256='fixed',forms=['10-K','10-Q','10-K/A','10-Q/A'],additional_candidates=[dict(symbol='ABC',cik='2')])
    return state,discovery,specification


def test_explicit_predecessor_is_collected_without_name_matching_or_approval(tmp_path):
    m=module()
    result=m.build(*fixture(tmp_path))
    assert result['accession_count']==1 and result['candidate_cik_count']==2
    row=result['records'][0]
    assert row['cik']=='2' and row['discovery_symbols']==['ABC']
    assert row['source_url'].endswith('/2/000000000110000001/original.xml')
    assert not row['original_period_verified']
    assert not result['financial_features_enabled']
    assert not result['historical_mapping_verified']


def test_traversal_filename_and_partial_scope_rejected(tmp_path):
    m=module()
    with pytest.raises(ValueError,match='safe original'):
        m.build(*fixture(tmp_path,instance='../leak.xml'))
    state,discovery,specification=fixture(tmp_path)
    state['completed'].pop()
    with pytest.raises(ValueError,match='complete'):
        m.build(state,discovery,specification)


def test_source_corruption_rejected_before_candidate_extraction(tmp_path):
    m=module()
    state,discovery,specification=fixture(tmp_path)
    Path(state['completed'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash/size'):
        m.build(state,discovery,specification)
