import importlib
from pathlib import Path
from zipfile import ZipFile

import pytest


def test_exact_exception_rows_are_verified_without_relaxing_master_scope(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    source = importlib.import_module('collect_task_exception_sources')
    path = tmp_path / 'archive.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('master.idx', 'CIK|Company Name|Form Type|Date Filed|Filename\n'
                         '1|Synthetic spinoff|8-K|2013-01-02|edgar/data/1/0000000001-13-000001.txt\n'
                         '2|Synthetic predecessor|10-K|2009-12-02|edgar/data/2/0000000002-09-000001.txt\n')
    rows = [{'symbol': symbol, 'candidate_cik': cik, 'form': form, 'filed': filed,
             'submission_path': filename} for symbol, cik, form, filed, filename in [
        ('ABBV', '1', '8-K', '2013-01-02', 'edgar/data/1/0000000001-13-000001.txt'),
        ('DIS', '2', '10-K', '2009-12-02', 'edgar/data/2/0000000002-09-000001.txt')]]
    selection = {'master_archive_sha256': {str(path): source.checksum(path)}, 'records': rows}
    assert len(source.verify_candidates(selection)) == 2
    rows[0]['filed'] = '2013-01-03'
    with pytest.raises(ValueError, match='one exact'):
        source.verify_candidates(selection)
