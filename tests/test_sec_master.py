import hashlib
from zipfile import ZipFile

import pytest

from rl_stocks.data.providers.sec.master import read_master


def archive(tmp_path,rows,member='master.idx'):
    path=tmp_path/'master.zip'
    with ZipFile(path,'w') as z:
        z.writestr(member,'Description: fixture\nCIK|Company Name|Form Type|Date Filed|Filename\n-----\n'+'\n'.join(rows)+'\n')
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_joint_filers_preserve_pairs_and_actual_directory_owner(tmp_path):
    rows=['1|Parent|10-K|2020-02-01|edgar/data/1/0000000001-20-000001.txt',
          '2|Subsidiary|10-K|2020-02-01|edgar/data/1/0000000001-20-000001.txt',
          '3|Lookalike|10-K|2020-02-01|edgar/data/3/0000000003-20-000001.txt']
    result=read_master(*archive(tmp_path,rows),{'1','2'})
    assert len(result['records'])==2
    assert result['records'][1]['cik']=='2'
    assert result['records'][1]['filing_directory_cik']=='1'
    assert not result['financial_values_read']


def test_retired_metadata_excluded_and_nonfinancial_forms_not_counted(tmp_path):
    rows=['1|Parent|10-K|2024-02-01|edgar/data/1/0000000001-24-000001.txt',
          '1|Parent|8-K|2020-02-01|edgar/data/1/0000000001-20-000001.txt']
    result=read_master(*archive(tmp_path,rows),{'1'})
    assert not result['records']
    assert result['retired_selected_rows_excluded']==1


@pytest.mark.parametrize('row',[
    '1|Parent|10-K|2020-02-01|../../secret.txt',
    '1|Parent|10-K|2020-02-30|edgar/data/1/0000000001-20-000001.txt',
    'malformed',
])
def test_bad_selected_rows_reject(tmp_path,row):
    with pytest.raises(ValueError):
        read_master(*archive(tmp_path,[row]),{'1'})


def test_wrong_member_and_corrupt_bytes_reject(tmp_path):
    with pytest.raises(ValueError,match='bounded'):
        read_master(*archive(tmp_path,[],member='../master.idx'),{'1'})
    path,sha=archive(tmp_path,[])
    with pytest.raises(ValueError,match='checksum'):
        read_master(path,'0'*64,{'1'})
