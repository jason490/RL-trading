"""Outcome-free FSDS structural audit, without extracting or trusting ZIP paths."""
import csv
import hashlib
import io
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zipfile import ZipFile

REQUIRED = {
    'sub': {'adsh', 'cik', 'form', 'period', 'filed', 'accepted'},
    'num': {'adsh', 'tag', 'version', 'ddate', 'qtrs', 'uom', 'coreg', 'value'},
    'tag': {'tag', 'version', 'custom'},
    'pre': {'adsh', 'report', 'line', 'tag', 'version'},
}
CUTOFF = datetime(2024, 1, 1)


def _date(value):
    if not re.fullmatch(r'\d{8}', value):
        raise ValueError('invalid SEC date')
    result = datetime.strptime(value, '%Y%m%d')
    if result >= CUTOFF:
        raise ValueError('retired-period record rejected')
    return result


def audit_archive(path: Path, quarter: str, *, max_uncompressed_bytes=2_000_000_000):
    """Read local FSDS ZIP; counts are structural evidence, not causal readiness.

    Acceptance timestamps stay naive source strings: their timezone/publication
    semantics are deliberately not inferred. No current amendment flag is used.
    """
    if not re.fullmatch(r'\d{4}q[1-4]', quarter) or not '2009q2' <= quarter <= '2023q4':
        raise ValueError('audit quarter outside fixed pre2024 scope')
    path = Path(path)
    with path.open('rb') as source:
        before = hashlib.file_digest(source, 'sha256').hexdigest()
    counts, forms, issues = Counter(), Counter(), Counter()
    submissions = {}
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate ZIP member')
        if any(n not in {f'{t}.txt' for t in REQUIRED} for n in names):
            raise ValueError('unexpected ZIP member; no extraction performed')
        if set(names) != {f'{t}.txt' for t in REQUIRED}:
            raise ValueError('missing SEC table')
        if sum(m.file_size for m in archive.infolist()) > max_uncompressed_bytes:
            raise ValueError('SEC archive exceeds uncompressed byte limit')

        def rows(table):
            with archive.open(f'{table}.txt') as raw:
                with io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream, delimiter='\t')
                    fields = reader.fieldnames or []
                    if len(fields) != len(set(fields)) or not REQUIRED[table] <= set(fields):
                        raise ValueError(f'invalid {table} schema')
                    for row in reader:
                        if None in row or any(v is None for v in row.values()):
                            raise ValueError(f'malformed {table} row')
                        counts[table] += 1
                        yield row

        for row in rows('sub'):
            accession = row['adsh']
            if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
                raise ValueError('invalid accession')
            if accession in submissions:
                raise ValueError('duplicate submission accession')
            if not re.fullmatch(r'[1-9]\d{0,9}', row['cik']):
                raise ValueError('invalid CIK')
            period, filed = _date(row['period']), _date(row['filed'])
            accepted = datetime.strptime(row['accepted'], '%Y-%m-%d %H:%M:%S')
            if accepted >= CUTOFF:
                raise ValueError('retired acceptance rejected')
            if accepted.strftime('%Y-%m-%d %H:%M:%S') != row['accepted']:
                raise ValueError('invalid acceptance format')
            if period > filed:
                issues['period_after_filed'] += 1
            if accepted.date() > filed.date():
                issues['acceptance_date_after_filed'] += 1
            if row['form'].endswith('/A'):
                counts['amendments'] += 1
            forms[row['form']] += 1
            submissions[accession] = {k: row[k] for k in REQUIRED['sub']}
        for row in rows('num'):
            if row['adsh'] not in submissions:
                raise ValueError('orphan NUM accession')
            _date(row['ddate'])
            if not re.fullmatch(r'\d+', row['qtrs']):
                raise ValueError('invalid NUM duration')
            if row['coreg']:
                counts['coregistrant_rows'] += 1
            if not row['value']:
                counts['missing_numeric_values'] += 1
            else:
                try:
                    if not Decimal(row['value']).is_finite():
                        raise ValueError('nonfinite NUM value')
                except InvalidOperation as error:
                    raise ValueError('invalid NUM value') from error
        for row in rows('tag'):
            if row['custom'] not in {'0', '1'}:
                raise ValueError('invalid TAG custom flag')
            counts['custom_tags' if row['custom'] == '1' else 'standard_tags'] += 1
        for row in rows('pre'):
            if row['adsh'] not in submissions:
                raise ValueError('orphan PRE accession')
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != before:
            raise ValueError('archive changed during inspection')
    return {
        'schema': 'sec-archive-structural-audit-v1', 'quarter': quarter,
        'archive_sha256': before, 'counts': dict(counts), 'forms': dict(forms),
        'issues': dict(issues), 'submissions': sorted(submissions.values(), key=lambda x: x['adsh']),
        'causal_ready': False, 'timezone_verified': False, 'publication_verified': False,
        'identity_mapping_verified': False, 'financial_features_enabled': False,
        'training_performed': False, 'performance_scored': False,
    }
