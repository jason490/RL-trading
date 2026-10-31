from hashlib import sha256
import json
from urllib.error import HTTPError

import pytest

from rl_stocks.data.providers.sec.html_source import acquire_filing_html, validate_endpoint

URL = 'https://www.sec.gov/Archives/edgar/data/320193/000119312509214859/d10k.htm'
OPTIONS = {'url': URL, 'cik': '320193', 'accession': '0001193125-09-214859',
           'user_agent': 'synthetic-test contact@example.org'}


def test_endpoint_scope_rejects_foreign_or_retired_source():
    assert validate_endpoint(URL, OPTIONS['cik'], OPTIONS['accession']) == URL
    for bad in (URL.replace('www.sec.gov', 'example.org'), URL + '?x=1',
                URL.replace('d10k.htm', '../d10k.htm')):
        with pytest.raises(ValueError):
            validate_endpoint(bad, OPTIONS['cik'], OPTIONS['accession'])
    with pytest.raises(ValueError, match='retired'):
        validate_endpoint(URL.replace('2009', '2024'), '320193', '0001193125-24-214859')


def test_cache_reuse_checks_bytes_and_never_calls_network(tmp_path, monkeypatch):
    import rl_stocks.data.providers.sec.html_source as module
    raw = b'<html>synthetic historical source</html>'
    sha = sha256(raw).hexdigest()
    target = tmp_path / sha
    target.mkdir()
    (target / 'filing.htm').write_bytes(raw)
    (target / 'receipt.json').write_text(json.dumps({'source_url': URL, 'sha256': sha,
                                                    'bytes': len(raw)}))
    def forbidden(*args, **kwargs):
        pytest.fail('network requested despite cache')
    monkeypatch.setattr(module, 'build_opener', forbidden)
    path, _, cached = acquire_filing_html(tmp_path, **OPTIONS)
    assert cached and path.read_bytes() == raw
    path.write_bytes(raw + b'x')
    with pytest.raises(ValueError, match='corruption'):
        acquire_filing_html(tmp_path, **OPTIONS)


def test_access_rejection_is_single_attempt_and_leaves_no_source(tmp_path, monkeypatch):
    import rl_stocks.data.providers.sec.html_source as module
    calls = []
    class Opener:
        def open(self, *args, **kwargs):
            calls.append(True)
            raise HTTPError(URL, 403, 'blocked', {}, None)
    monkeypatch.setattr(module, 'build_opener', lambda *a: Opener())
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='HTTP403'):
        acquire_filing_html(tmp_path, **OPTIONS)
    assert len(calls) == 1
    assert not list(tmp_path.glob('*/filing.htm'))


def test_primary_html_comes_from_exact_index_row(tmp_path):
    from rl_stocks.data.providers.sec.filing_index import primary_html
    raw = b'''<html><table><tr><td>1</td><td>ANNUAL</td><td><a href="d10k.htm">d10k.htm</a></td><td>10-K</td></tr>
    <tr><td>2</td><td>EXHIBIT</td><td><a href="exhibit.htm">exhibit.htm</a></td><td>EX-13</td></tr></table></html>'''
    path = tmp_path / 'index.htm'
    path.write_bytes(raw)
    assert primary_html(path, sha256(raw).hexdigest(), cik=OPTIONS['cik'],
                        accession=OPTIONS['accession'], form='10-K') == URL
    with pytest.raises(ValueError, match='one explicitly'):
        primary_html(path, sha256(raw).hexdigest(), cik=OPTIONS['cik'],
                     accession=OPTIONS['accession'], form='10-Q')


def test_single_success_is_immutable_and_oversize_rejects_before_read(tmp_path, monkeypatch):
    import io
    import rl_stocks.data.providers.sec.html_source as module
    raw = b'<html>synthetic source</html>'
    calls = []
    class Response(io.BytesIO):
        status = 200
        headers = {'Content-Length': str(len(raw))}
    class Opener:
        def open(self, *args, **kwargs):
            calls.append(True)
            return Response(raw)
    monkeypatch.setattr(module, 'build_opener', lambda *a: Opener())
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    path, receipt, cached = acquire_filing_html(tmp_path, **OPTIONS)
    assert not cached and path.read_bytes() == raw
    assert receipt['sha256'] == sha256(raw).hexdigest()
    assert acquire_filing_html(tmp_path, **OPTIONS)[2] is True
    assert len(calls) == 1

    class Oversize(Response):
        headers = {'Content-Length': '20000001'}
        def read(self, *args):
            pytest.fail('oversized response body read')
    class BadOpener:
        def open(self, *args, **kwargs):
            return Oversize(raw)
    monkeypatch.setattr(module, 'build_opener', lambda *a: BadOpener())
    with pytest.raises(ValueError, match='declared size'):
        acquire_filing_html(tmp_path / 'other', **OPTIONS)


def test_concurrent_collector_rejects_without_network(tmp_path, monkeypatch):
    import fcntl
    import rl_stocks.data.providers.sec.html_source as module
    def forbidden(*args, **kwargs):
        pytest.fail('concurrent collector issued request')
    monkeypatch.setattr(module, 'build_opener', forbidden)
    with (tmp_path / '.html-download.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='busy'):
            acquire_filing_html(tmp_path, **OPTIONS)


def test_joint_directory_requires_both_official_master_rows_and_same_accession(tmp_path):
    from rl_stocks.data.providers.sec.filing_index import primary_html, primary_html_joint
    from test_sec_master import archive
    index = tmp_path / 'joint-index.htm'
    index.write_text('<html><table><tr><td>1</td><td>ANNUAL</td><td><a href="'
                     '/Archives/edgar/data/2/000000000110000001/main.htm">main</a>'
                     '</td><td>10-K</td></tr></table></html>')
    sha = sha256(index.read_bytes()).hexdigest()
    kwargs = dict(cik='1', accession='0000000001-10-000001', form='10-K')
    with pytest.raises(ValueError, match='differs'):
        primary_html(index, sha, **kwargs)
    rows = ['1|Parent|10-K|2010-02-01|edgar/data/1/0000000001-10-000001.txt',
            '2|Joint filer|10-K|2010-02-01|edgar/data/2/0000000001-10-000001.txt']
    path, master_sha = archive(tmp_path, rows)
    result = primary_html_joint(index, sha, **kwargs, master_archive=path,
                                master_sha256=master_sha)
    assert result['cross_directory'] and result['directory_cik'] == '2'
    assert not result['historical_identity_verified']
    for bad in ([rows[0]], [rows[0], rows[1].replace('2010-02-01', '2010-02-02')],
                [rows[0], rows[1].replace('0000000001-10-000001', '0000000001-10-000002')]):
        path, bad_sha = archive(tmp_path, bad)
        with pytest.raises(ValueError, match='both registrants'):
            primary_html_joint(index, sha, **kwargs, master_archive=path, master_sha256=bad_sha)
