"""Human-readable coverage failures and machine-readable ingestion requests."""
import csv
import html
import json
from pathlib import Path


def report(value, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'coverage.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    with (output/'eligible-sessions.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['security_id','session','decision_time','ticker','price_covered'])
        writer.writeheader()
        writer.writerows(value['eligible_panel'])
    with (output/'security-coverage.csv').open('w', newline='') as file:
        fields = ['security_id','eligible_sessions','covered_sessions','missing_sessions','terminal_required','terminal_covered']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(value['securities'])
    summary = {k:v for k,v in value.items() if k not in ('eligible_panel','problems','securities')}
    text = '<!doctype html><meta charset="utf-8"><title>Historical universe coverage</title>'
    text += '<h1>Historical universe coverage</h1><p>No securities with missing data are silently dropped. '
    text += 'Coverage approval is conditional on the supplied source evidence.</p><pre>'
    text += html.escape(json.dumps(summary, indent=2))+'</pre><h2>Problems</h2><pre>'
    text += html.escape(json.dumps(value['problems'], indent=2))+'</pre>'
    (output/'coverage.html').write_text(text)
    return output/'coverage.html'
