"""Prepare public model assets and derived equity reports; no raw bars or replay."""
import argparse
import html
import json
from pathlib import Path
import tarfile

from rl_stocks.reporting.offline import chart


def package(roots, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    models=output/'research-models.tar.gz'
    sections=['<!doctype html><meta charset="utf-8"><title>Research validation reports</title>',
              '<h1>Research validation reports</h1><p>Selected-model simulations after modeled costs. '
              'Separate symbol episodes; validation selected the models. Not held-out performance.</p>']
    catalog=[]
    with tarfile.open(models,'w:gz') as archive:
        for root in map(Path,roots):
            for run in sorted((root/'runs').glob('*')):
                metadata=json.loads((run/'metadata.json').read_text())
                if metadata.get('status')!='complete':
                    continue
                catalog.append(metadata)
                bundle=root/'frozen'/metadata['bundle_id']
                # Preserve the exact complete frozen directory/checksum file set.
                allowed={'model.zip','manifest.json','checksums.json','requirements.lock',
                         'requirements.txt','dependencies.json'}
                if any(p.name not in allowed or not p.is_file() for p in bundle.iterdir()):
                    raise ValueError('unexpected frozen asset; review before publication')
                archive.add(bundle,arcname='frozen/'+bundle.name)
                for name in ('metadata.json','best.zip','latest.zip','checkpoints.json'):
                    if (run/name).exists():archive.add(run/name,arcname='runs/'+run.name+'/'+name)
                selected=run/'selected-validation.json'
                if not selected.exists():selected=run/'selected-validation-cpu.json'
                if not selected.exists():continue
                results=json.loads(selected.read_text())
                sections.append('<h2>'+html.escape(run.name)+'</h2>')
                for key,result in results.items():
                    rows=result['rows'];base=result['baselines']['buy_hold']['rows']
                    sections.append(chart(key,[r['session'] for r in rows],[
                        ('Policy equity',[r['equity'] for r in rows]),
                        ('Buy and hold',[r['equity'] for r in base])]))
                    sections.append('<pre>'+html.escape(json.dumps(result['metrics'],indent=2))+'</pre>')
    (output/'validation.html').write_text('\n'.join(sections))
    (output/'model-catalog.json').write_text(json.dumps(catalog,indent=2,allow_nan=False))
    return models


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('roots',nargs='+',help='Retrieved artifacts roots containing runs/ and frozen/')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(package(args.roots,args.output))
