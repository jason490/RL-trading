import csv
import hashlib
import io
from zipfile import ZipFile

import pytest

from rl_stocks.data.providers.sec.concepts import scan_archive


A='0000000001-20-000001'


def fixture(tmp_path,change=None):
    rows=[]
    for tag,q in [('Revenues','4'),('NetIncomeLoss','4'),('Assets','0'),('Liabilities','0')]:
        rows.append(dict(adsh=A,tag=tag,version='us-gaap/2019',ddate='20191231',qtrs=q,
                         uom='USD',coreg='',segments='',value='123'))
    if change:
        change(rows)
    stream=io.StringIO()
    writer=csv.DictWriter(stream,rows[0].keys(),delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
    path=tmp_path/'archive.zip'
    with ZipFile(path,'w') as z:
        z.writestr('num.txt',stream.getvalue())
    item=dict(path=str(path),bytes=path.stat().st_size,archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),quarter='2020q1')
    targets=[dict(accession=A,cik='1',form='10-K',archive_period_label='20191231',archive_filed='20200201')]
    return item,targets


def test_presence_pass_does_not_create_ratios_or_model_readiness(tmp_path):
    result=scan_archive(*fixture(tmp_path))
    row=result['records'][0]
    assert row['structural_presence_pass'] and row['same_flow_qtrs']==[4]
    assert not result['financial_ratios_computed'] and not result['financial_features_enabled']
    assert 'value' not in row


@pytest.mark.parametrize('change',[
    lambda r:r[1].update(tag='ProfitLoss'),
    lambda r:r[1].update(tag='OperatingIncomeLoss'),
    lambda r:r[0].update(tag='SalesRevenueGoodsNet'),
    lambda r:r[0].update(segments='Region=Europe'),
    lambda r:r[0].update(coreg='subsidiary'),
    lambda r:r[0].update(value=''),
    lambda r:r[0].update(uom='EUR'),
    lambda r:r[0].update(ddate='20241231'),
    lambda r:r[0].update(qtrs='3'),
    lambda r:r[0].update(version=A),
])
def test_alternatives_and_incompatible_contexts_never_fill_required_presence(tmp_path,change):
    result=scan_archive(*fixture(tmp_path,change))
    assert not result['records'][0]['structural_presence_pass']


def test_unsupported_names_remain_visible_as_diagnostic_only(tmp_path):
    def change(rows):
        rows[1]['tag']='ProfitLoss'
    result=scan_archive(*fixture(tmp_path,change))
    assert result['records'][0]['unsupported_current_labels']==['ProfitLoss']
    assert not result['records'][0]['current_net_income_present']


def test_corrupted_archive_rejects(tmp_path):
    item,targets=fixture(tmp_path)
    item['archive_sha256']='0'*64
    with pytest.raises(ValueError,match='checksum'):
        scan_archive(item,targets)
