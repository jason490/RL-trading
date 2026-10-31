"""Select the explicitly typed primary HTML document from an original SEC index."""
import hashlib
from html.parser import HTMLParser
from pathlib import Path

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


def primary_html(path, expected_sha256, *, cik, accession, form):
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
        validate_endpoint(url, cik, accession)
        choices.append(url)
    if len(choices) != 1:
        raise ValueError('one explicitly typed sequence1 primary HTML source required')
    return choices[0]
