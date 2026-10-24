import csv
import io
from zipfile import ZipFile

import pytest

from rl_stocks.data.providers.sec.archive import REQUIRED, audit_archive


A = '0000000001-20-000001'


def fixture(tmp_path, change=None, extra=None):
    data = {
        'sub': [dict(adsh=A, cik='1', form='10-K', period='20191231',
                     filed='20200203', accepted='2020-02-03 18:01:02', prevrpt='1')],
        'num': [dict(adsh=A, tag='Assets', version='us-gaap/2019', ddate='20191231',
                     qtrs='0', uom='USD', coreg='', value='123')],
        'tag': [dict(tag='Assets', version='us-gaap/2019', custom='0')],
        'pre': [dict(adsh=A, report='1', line='1', tag='Assets', version='us-gaap/2019')],
    }
    if change:
        change(data)
    path = tmp_path / 'archive.zip'
    with ZipFile(path, 'w') as z:
        for table, records in data.items():
            stream = io.StringIO()
            writer = csv.DictWriter(stream, sorted(set(REQUIRED[table]) | set(records[0])))
            # SEC tables are tab-separated.
            writer = csv.DictWriter(stream, writer.fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(records)
            z.writestr(table + '.txt', stream.getvalue())
        if extra:
            z.writestr(extra, 'bad')
    return path


def test_original_and_amendment_retained_and_future_flag_ignored(tmp_path):
    def change(data):
        amendment = dict(data['sub'][0], adsh='0000000001-20-000002', form='10-K/A',
                         accepted='2020-02-04 12:00:00', filed='20200204')
        data['sub'].append(amendment)
    path = fixture(tmp_path, change)
    result = audit_archive(path, '2020q1')
    assert result['counts']['amendments'] == 1
    assert len(result['submissions']) == 2
    assert all('prevrpt' not in r for r in result['submissions'])
    assert result['submissions'][0]['accepted'] == '2020-02-03 18:01:02'
    assert not result['causal_ready']
    assert not result['publication_verified']


@pytest.mark.parametrize('table,key,value', [
    ('sub', 'filed', '20240101'), ('sub', 'period', '20240101'),
    ('sub', 'accepted', '2024-01-01 00:00:00'), ('num', 'ddate', '20240101'),
    ('num', 'adsh', '0000000001-20-999999'), ('pre', 'adsh', 'missing'),
    ('num', 'value', 'NaN'), ('num', 'value', 'Infinity'),
    ('num', 'qtrs', '-1'), ('sub', 'cik', 'nan'),
])
def test_reject_bad_or_retired_records(tmp_path, table, key, value):
    path = fixture(tmp_path, lambda d: d[table][0].update({key: value}))
    with pytest.raises(ValueError):
        audit_archive(path, '2020q1')


@pytest.mark.parametrize('quarter', ['2024q1', '2009q1', '2020q5', '../2020q1'])
def test_reject_out_of_scope_quarter(tmp_path, quarter):
    with pytest.raises(ValueError):
        audit_archive(fixture(tmp_path), quarter)


def test_reject_unsafe_members_and_resource_limit(tmp_path):
    with pytest.raises(ValueError, match='unexpected'):
        audit_archive(fixture(tmp_path, extra='../sub.txt'), '2020q1')
    with pytest.raises(ValueError, match='byte limit'):
        audit_archive(fixture(tmp_path), '2020q1', max_uncompressed_bytes=1)


def test_duplicate_submission_is_not_silently_overwritten(tmp_path):
    path = fixture(tmp_path, lambda d: d['sub'].append(dict(d['sub'][0])))
    with pytest.raises(ValueError, match='duplicate submission'):
        audit_archive(path, '2020q1')


def test_cli_creates_immutable_report(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        'sec_audit_cli', Path(__file__).parents[1] / 'scripts/audit_sec_archive.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = fixture(tmp_path)
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--archive', str(archive),
                                    '--quarter', '2020q1', '--output', str(output)])
    module.main()
    saved = output.read_bytes()
    assert json.loads(saved)['causal_ready'] is False
    module.main()
    assert output.read_bytes() == saved
    output.write_text('{}')
    with pytest.raises(ValueError, match='immutable'):
        module.main()


def test_segments_are_distinct_numeric_keys(tmp_path):
    def change(data):
        data['num'][0]['segments'] = ''
        data['num'].append(dict(data['num'][0], segments='Geography=Europe', value='99'))
    result = audit_archive(fixture(tmp_path, change), '2020q1')
    assert result['numeric_schema'] == 'segments-v2'
    assert result['counts']['unique_numeric_keys'] == 2
    assert result['counts']['segment_numeric_rows'] == 1
    assert not result['issues'].get('conflicting_numeric_values')
    assert 'segments' in result['numeric_unique_key']


def test_legacy_schema_reported_without_claiming_segment_fidelity(tmp_path):
    result = audit_archive(fixture(tmp_path), '2020q1')
    assert result['numeric_schema'] == 'legacy-v1'
    assert not result['causal_ready']


@pytest.mark.parametrize('value,conflicts', [('123.0000', 0), ('124', 1), ('', 1)])
def test_duplicate_values_are_audited_without_silent_overwrite(tmp_path, value, conflicts):
    def change(data):
        data['num'].append(dict(data['num'][0], value=value))
    result = audit_archive(fixture(tmp_path, change), '2020q1')
    assert result['counts']['unique_numeric_keys'] == 1
    assert result['counts']['duplicate_numeric_keys'] == 1
    assert result['issues'].get('conflicting_numeric_values', 0) == conflicts
    assert not result['financial_features_enabled']


def test_official_optional_readme_allowed_without_extraction(tmp_path):
    path = fixture(tmp_path, extra='readme.htm')
    result = audit_archive(path, '2020q1')
    assert result['counts']['sub'] == 1
    assert not (tmp_path / 'readme.htm').exists()


@pytest.mark.parametrize('suffix', ['.0', '.123456'])
def test_acceptance_fraction_preserved(tmp_path, suffix):
    value = '2020-02-03 18:01:02' + suffix
    path = fixture(tmp_path, lambda d: d['sub'][0].update(accepted=value))
    result = audit_archive(path, '2020q1')
    assert result['submissions'][0]['accepted'] == value
    assert not result['timezone_verified']


@pytest.mark.parametrize('value', ['2024-01-01 00:00:00.0', '2020-2-03 18:01:02.0',
                                  '2020-02-03 18:01:02Z', '2020-02-03 18:01:02.1234567'])
def test_ambiguous_or_retired_acceptance_rejected(tmp_path, value):
    path = fixture(tmp_path, lambda d: d['sub'][0].update(accepted=value))
    with pytest.raises(ValueError):
        audit_archive(path, '2020q1')
