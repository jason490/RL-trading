"""Prepare public model assets and derived equity reports; no raw bars or replay."""
import argparse
import html
import json
from pathlib import Path
import tarfile
from urllib.parse import quote

from rl_stocks.reporting.offline import chart


def equity_sections(results):
    sections=[]
    for key,result in results.items():
        rows=result['rows'];base=result['baselines']['buy_hold']['rows']
        sections.append(chart(key,[r['session'] for r in rows],[
            ('Policy equity',[r['equity'] for r in rows]),
            ('Buy and hold',[r['equity'] for r in base])]))
        sections.append('<pre>'+html.escape(json.dumps(result['metrics'],indent=2))+'</pre>')
    return sections


def package_audit(audit_root, output):
    audit_root=Path(audit_root);output=Path(output)
    audit=json.loads((audit_root/'audit.json').read_text())
    if audit['status']!='complete':raise ValueError('held-out audit is incomplete')
    sections=['<!doctype html><meta charset="utf-8"><title>Frozen-family held-out audit</title>',
              '<h1>Frozen-family held-out audit</h1><p>Family and candidate were selected using '
              'validation and committed before this audit. Separate symbol episodes, modeled costs. '
              'No test-driven checkpoint selection or retuning.</p>']
    for entry in audit['seeds']:
        seed=entry['seed']
        results=json.loads((audit_root/f'test-seed-{seed}.json').read_text())
        sections.append(f'<h2>Seed {seed}</h2>')
        sections.extend(equity_sections(results))
    output.mkdir(parents=True,exist_ok=True)
    (output/'test.html').write_text('\n'.join(sections))
    (output/'audit-summary.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')


def package(roots, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report_directory=output/'validation';report_directory.mkdir(exist_ok=True)
    models=output/'research-models.tar.gz'
    sections=['<!doctype html><meta charset="utf-8"><title>Research validation reports</title>',
              '<h1>Research validation reports</h1><p>Selected-model simulations after modeled costs. '
              'Separate symbol episodes; validation selected the models. Not held-out performance.</p>',
              '<p>Extract validation-reports.tar.gz beside this index to open the individual reports.</p><ul>']
    catalog=[]
    with tarfile.open(models,'w:gz') as archive:
        for root in map(Path,roots):
            for run in sorted((root/'runs').glob('*')):
                metadata=json.loads((run/'metadata.json').read_text())
                if metadata.get('status')!='complete':
                    continue
                catalog.append({**metadata,'run_name':run.name})
                bundle=root/'frozen'/metadata['bundle_id']
                # Preserve the exact complete frozen directory/checksum file set.
                allowed={'model.zip','manifest.json','checksums.json','requirements.lock',
                         'requirements.txt','dependencies.json'}
                if any(p.name not in allowed or not p.is_file() for p in bundle.iterdir()):
                    raise ValueError('unexpected frozen asset; review before publication')
                archive.add(bundle,arcname='frozen/'+bundle.name)
                for name in ('metadata.json','best.zip','latest.zip','checkpoints.json','checkpoint-state.json','checkpoint-validation.json','training-sampling.json','training-execution.json','guard-initialization.npz','session-initialization.npz','capacity-initialization.npz','recency-initialization.npz','l2-initialization.npz','clipping-initialization.npz','squashed-initialization.npz'):
                    if (run/name).exists():archive.add(run/name,arcname='runs/'+run.name+'/'+name)
                selected=run/'selected-validation.json'
                if not selected.exists():selected=run/'selected-validation-cpu.json'
                if not selected.exists():continue
                results=json.loads(selected.read_text())
                variant=metadata.get('research_variant',metadata.get('initialization',{}).get('arm','standard'))
                label=(f"{metadata.get('algorithm','unknown')} / {variant} / dataset {metadata.get('dataset','unknown')[:12]} / "
                       f"seed {metadata.get('seed','unknown')} / {metadata.get('actual_steps','unknown')} steps / "
                       +('smoke' if metadata.get('smoke') else 'research'))
                title=html.escape(label)
                page=['<!doctype html><meta charset="utf-8"><title>'+title+'</title>',
                      '<p><a href="../validation.html">All validation runs</a></p>',
                      '<h1>'+title+'</h1><p>Validation-selected model; separate symbol episodes.</p>']
                page.extend(equity_sections(results))
                (report_directory/(run.name+'.html')).write_text('\n'.join(page))
                score=metadata.get('selected_score')
                score_text=f'{score:.2%}' if score is not None else 'not recorded'
                sections.append('<li><a href="validation/'+quote(run.name)+'.html">'+title+'</a> — '
                    +html.escape(metadata.get('device','unknown'))+'; mean selected return '+score_text+'</li>')
    sections.append('</ul>')
    (output/'validation.html').write_text('\n'.join(sections))
    with tarfile.open(output/'validation-reports.tar.gz','w:gz') as archive:
        archive.add(output/'validation.html',arcname='validation.html')
        archive.add(report_directory,arcname='validation')
    (output/'model-catalog.json').write_text(json.dumps(catalog,indent=2,allow_nan=False))
    return models


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('roots',nargs='+',help='Retrieved artifacts roots containing runs/ and frozen/')
    parser.add_argument('--output',required=True)
    parser.add_argument('--audit',help='Completed held-out audit directory; exports derived results only')
    args=parser.parse_args()
    print(package(args.roots,args.output))
    if args.audit:package_audit(args.audit,args.output)
