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
