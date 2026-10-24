"""Read bounded SEC master-index metadata; preserve joint-filer issuer pairs."""
import hashlib
import io
import re
from datetime import date
from pathlib import Path
from zipfile import ZipFile

FORMS = {'10-K', '10-Q', '10-K/A', '10-Q/A'}


def read_master(path, expected_sha256, candidate_ciks):
    path = Path(path)
    if path.stat().st_size > 10_000_000:
        raise ValueError('compressed master limit exceeded')
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != expected_sha256:
            raise ValueError('master source checksum mismatch')
    records, seen, source_rows, excluded = [], set(), 0, 0
    with ZipFile(path) as z:
        if z.namelist() != ['master.idx'] or z.getinfo('master.idx').file_size > 100_000_000:
            raise ValueError('single bounded master.idx required')
        header = False
        with z.open('master.idx') as raw:
            for line in io.TextIOWrapper(raw, encoding='latin1'):
                line = line.strip()
                if not header:
                    if line == 'CIK|Company Name|Form Type|Date Filed|Filename':
                        header = True
                    continue
                if not line or set(line) == {'-'}:
                    continue
                fields = line.split('|')
                if len(fields) != 5:
                    raise ValueError('malformed master row')
                source_rows += 1
                cik, name, form, filed, filename = (f.strip() for f in fields)
                if cik not in candidate_ciks or form not in FORMS:
                    continue
                if not re.fullmatch(r'[1-9][0-9]{0,9}', cik):
                    raise ValueError('valid registrant CIK required')
                if date.fromisoformat(filed).isoformat() != filed:
                    raise ValueError('exact master filing date required')
                if filed >= '2024-01-01':
                    excluded += 1
                    continue
                match = re.fullmatch(r'edgar/data/([1-9][0-9]{0,9})/(\d{10}-\d{2}-\d{6})\.txt', filename)
                if not match:
                    raise ValueError('safe master submission path required')
                directory_cik, accession = match.groups()
                key = (cik, accession)
                if key in seen:
                    raise ValueError('duplicate registrant/accession pair in master')
                seen.add(key)
                records.append(dict(cik=cik, company_name=name, form=form, filed=filed,
                                    accession=accession, filing_directory_cik=directory_cik,
                                    submission_url='https://www.sec.gov/Archives/' + filename))
        if not header:
            raise ValueError('master header missing')
    return dict(records=records, source_rows=source_rows, retired_selected_rows_excluded=excluded,
                source_sha256=expected_sha256, financial_values_read=False)
