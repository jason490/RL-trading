"""Select the explicitly typed primary HTML document from an original SEC index."""
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re

from rl_stocks.data.providers.sec.html_source import validate_endpoint


class _Rows(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.row = None
        self.cell = None
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag == 'td' and self.row is not None:
            self.cell = {'text': [], 'hrefs': []}
        elif tag == 'a' and self.cell is not None:
            href = dict(attrs).get('href')
            if href:
                self.cell['hrefs'].append(href)

    def handle_data(self, data):
        if self.cell is not None:
            self.cell['text'].append(data)

    def handle_endtag(self, tag):
        if tag == 'td' and self.cell is not None:
            self.row.append({'text': ' '.join(''.join(self.cell['text']).split()),
                             'hrefs': self.cell['hrefs']})
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _primary_candidate(path, expected_sha256, *, cik, accession, form):
    with Path(path).open('rb') as stream:
        raw = stream.read(20_000_001)
    if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('bounded SEC index checksum mismatch')
    parser = _Rows()
    parser.feed(raw.decode('utf-8-sig'))
    choices = []
    for row in parser.rows:
        if (len(row) < 4 or row[0]['text'] != '1' or row[3]['text'] != form
                or len(row[2]['hrefs']) != 1):
            continue
        href = row[2]['hrefs'][0]
        if href.startswith('/Archives/'):
            url = 'https://www.sec.gov' + href
        elif href.startswith('https://www.sec.gov/'):
            url = href
        elif '/' not in href:
            url = (f'https://www.sec.gov/Archives/edgar/data/{cik}/'
                   f'{accession.replace("-", "")}/' + href)
        else:
            raise ValueError('unexpected primary document link')
        choices.append(url)
    if len(choices) != 1:
        raise ValueError('one explicitly typed sequence1 primary HTML source required')
    return choices[0]


def primary_html(path, expected_sha256, *, cik, accession, form):
    url = _primary_candidate(path, expected_sha256, cik=cik, accession=accession, form=form)
    return validate_endpoint(url, cik, accession)


def primary_html_joint(path, expected_sha256, *, cik, accession, form,
                       master_archive, master_sha256):
    """Authorize only an index-linked directory corroborated by the same master accession."""
    from rl_stocks.data.providers.sec.master import read_master
    url = _primary_candidate(path, expected_sha256, cik=cik, accession=accession, form=form)
    match = re.fullmatch(r'https://www.sec.gov/Archives/edgar/data/([1-9][0-9]{0,9})/'
                         r'([0-9]{18})/[^/]+', url)
    if match is None or match[2] != accession.replace('-', ''):
        raise ValueError('same-accession original SEC primary endpoint required')
    directory = match[1]
    validate_endpoint(url, directory, accession)
    identities = {cik, directory}
    rows = [r for r in read_master(master_archive, master_sha256, identities, forms={form})['records']
            if r['accession'] == accession and r['form'] == form]
    if ({r['cik'] for r in rows} != identities or len(rows) != len(identities)
            or len({r['filed'] for r in rows}) != 1
            or directory not in {r['filing_directory_cik'] for r in rows}):
        raise ValueError('both registrants and target directory require matching official master rows')
    return {'url': url, 'declared_cik': cik, 'directory_cik': directory,
            'accession': accession, 'form': form, 'cross_directory': directory != cik,
            'index_sha256': expected_sha256, 'master_sha256': master_sha256,
            'master_records': rows, 'historical_identity_verified': False}
