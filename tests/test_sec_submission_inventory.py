import csv
import hashlib
import importlib.util
import io
from pathlib import Path
from zipfile import ZipFile

import pytest


def module():
    spec=importlib.util.spec_from_file_location(
        'sec_inventory',Path(__file__).parents[1]/'scripts/inventory_sec_submissions.py')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fixture(tmp_path,m,changes=None):
    row=dict(adsh='0000000123-20-000001',cik='123',form='10-K',period='20191231',
             filed='20200201',accepted='2020-02-01 12:01:01.0')
    entries=[]
    for i,q in enumerate(m.QUARTERS):
        records=[dict(row)] if i in {0,1} else []
        if changes:
            changes(i,records)
        path=tmp_path/(q+'.zip')
        text=io.StringIO()
        writer=csv.DictWriter(text,m.FIELDS,delimiter='\t')
        writer.writeheader()
        writer.writerows(records)
        with ZipFile(path,'w') as z:
            z.writestr('sub.txt',text.getvalue())
        entries.append(dict(quarter=q,path=str(path),archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    return dict(schema='sec-quarter-collection-v1',status='downloaded_only',
                planned_quarters=m.QUARTERS,completed=entries,
                protocol_sha256=hashlib.sha256(Path('configs/experiments/filing_coverage_audit.json').read_bytes()).hexdigest())


def test_unique_count_does_not_double_count_cross_archive_duplicates(tmp_path):
    m=module()
    result=m.inventory(fixture(tmp_path,m))
    assert result['counts']['source_rows']==2
    assert result['counts']['unique_nonretired_submissions']==1
    assert result['counts']['cross_archive_duplicate_accessions']==1
    assert result['by_cik']['123']['filing_year_forms']=={'2020:10-K':1}
    assert not result['causal_ready'] and not result['financial_amounts_read']


def test_conflicting_duplicate_reported_not_replaced(tmp_path):
    m=module()
    def change(i,rows):
        if i==1:
            rows[0]['filed']='20200202'
    result=m.inventory(fixture(tmp_path,m,change))
    assert result['counts']['cross_archive_conflicts']==1
    assert result['conflict_examples'][0]['adsh']=='0000000123-20-000001'
    assert not result['historical_mapping_verified']


def test_incomplete_scope_and_changed_source_rejected(tmp_path):
    m=module()
    state=fixture(tmp_path,m)
    state['status']='running'
    with pytest.raises(ValueError,match='terminal'):
        m.inventory(state)
    state['status']='downloaded_only'
    state['completed'].pop()
    with pytest.raises(ValueError,match='complete'):
        m.inventory(state)
    state=fixture(tmp_path,m)
    Path(state['completed'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        m.inventory(state)


def test_retired_submissions_excluded_without_becoming_identity_evidence(tmp_path):
    m=module()
    def change(i,rows):
        if rows:
            rows[0]['filed']='20240101'
    result=m.inventory(fixture(tmp_path,m,change))
    assert result['counts']['retired_SUB_rows_excluded']==2
    assert not result['by_cik']
    assert not result['financial_features_enabled']


def test_nonperiodic_missing_period_is_reported_but_annual_missing_period_rejects(tmp_path):
    m=module()
    def nonperiodic(i,rows):
        if rows:
            rows[0].update(form='S-1',period='')
    result=m.inventory(fixture(tmp_path,m,nonperiodic))
    assert result['counts']['nonperiodic_forms_without_period']==2
    assert not result['by_cik']
    def annual(i,rows):
        if rows:
            rows[0]['period']=''
    with pytest.raises(ValueError,match='date format'):
        m.inventory(fixture(tmp_path,m,annual))
