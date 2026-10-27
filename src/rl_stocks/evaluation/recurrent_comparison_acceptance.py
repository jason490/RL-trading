"""Offline physical acceptance and fixed gates, never checkpoint selection or training."""
from pathlib import Path
import re

from rl_stocks.core.contracts import digest
from rl_stocks.reporting.recurrent_sac_comparison import screen_gate,confirmation_gate
from rl_stocks.training.recurrent_comparison import protocol,PROTOCOL_SHA256
from rl_stocks.training.recurrent_comparison_verify import verify_cell

SCHEMA='recurrent-comparison-acceptance-v1'


def receipt_hash(receipt):
    return digest({key:value for key,value in receipt.items() if key!='receipt_sha256'})


def validate_manifest(manifest):
    spec=protocol()
    if (type(manifest) is not dict or set(manifest)!={'schema','protocol_sha256','stage','cells'}
            or manifest['schema']!='recurrent-comparison-matrix-v1'
            or manifest['protocol_sha256']!=PROTOCOL_SHA256
            or manifest['stage'] not in ('screen','confirmation') or type(manifest['cells']) is not list):
        raise ValueError('fixed comparison matrix manifest required')
    seeds=[11] if manifest['stage']=='screen' else spec['confirmation_seeds']
    expected={(arm,seed) for arm in spec['arms'] for seed in seeds}
    actual=set()
    paths=set()
    for entry in manifest['cells']:
        if (type(entry) is not dict or set(entry)!={'arm','seed','folder','source_hash','source_revision'}
                or type(entry['arm']) is not str or entry['arm'] not in spec['arms']
                or type(entry['seed']) is not int or type(entry['folder']) is not str or not entry['folder']
                or type(entry['source_hash']) is not str or not re.fullmatch('[0-9a-f]{64}',entry['source_hash'])
                or type(entry['source_revision']) is not str or not re.fullmatch('[0-9a-f]{40}',entry['source_revision'])
                or (entry['arm'],entry['seed']) in actual or entry['folder'] in paths):
            raise ValueError('unique exact role/seed cells and artifact paths required')
        actual.add((entry['arm'],entry['seed']))
        paths.add(entry['folder'])
    if actual!=expected:
        raise ValueError('complete fixed three-cell screen or nine-cell confirmation required')
    return seeds


def accept_matrix(manifest,dataset,starts,*,root,screen_receipt=None):
    seeds=validate_manifest(manifest)
    spec=protocol()
    if manifest['stage']=='confirmation':
        if (not isinstance(screen_receipt,dict) or screen_receipt.get('schema')!=SCHEMA
                or screen_receipt.get('protocol_sha256')!=PROTOCOL_SHA256
                or screen_receipt.get('stage')!='screen' or screen_receipt.get('dataset')!=spec['dataset']
                or set(screen_receipt.get('cells',{}))!={'11'}
                or screen_receipt.get('retired_test_access') is not False
                or screen_receipt.get('checkpoint_reselection') is not False
                or screen_receipt.get('gate',{}).get('passed') is not True
                or screen_receipt.get('receipt_sha256')!=receipt_hash(screen_receipt)):
            raise ValueError('unchanged passing original seed11 acceptance receipt required')
    elif screen_receipt is not None:
        raise ValueError('screen cannot import a confirmation or prior acceptance receipt')
    by_key={(entry['arm'],entry['seed']):entry for entry in manifest['cells']}
    cells={}
    sources={}
    resolved=set()
    # Always verify seed11 first; a failed screen never consumes remaining seeds.
    for seed in seeds:
        cells[seed]={}
        for arm in spec['arms']:
            entry=by_key[(arm,seed)]
            folder=(Path(root)/entry['folder']).resolve()
            if folder in resolved:
                raise ValueError('distinct physical cell directories required')
            resolved.add(folder)
            proof=verify_cell(folder,dataset,starts,expected_arm=arm,expected_seed=seed,
                expected_source_hash=entry['source_hash'],expected_git_revision=entry['source_revision'])
            if proof.get('saved_artifacts_verified') is not True or proof.get('arm')!=arm or proof.get('seed')!=seed:
                raise ValueError('physically verified matching cell required before gates')
            cells[seed][arm]=proof
            sources[f'{arm}-{seed}']=dict(folder=str(folder),source_hash=entry['source_hash'],
                                       source_revision=entry['source_revision'])
        if seed==11:
            first_gate=screen_gate(cells[11])
            if manifest['stage']=='confirmation':
                previous=screen_receipt['cells']['11']
                if any(cells[11][arm]['artifact_inventory_sha256']!=previous[arm]['artifact_inventory_sha256']
                       for arm in spec['arms']):
                    raise ValueError('original seed11 artifacts must be reused unchanged, never retrained')
                if not first_gate['passed']:
                    raise ValueError('failed seed11 closes the family before confirmation reads')
    gate=first_gate if manifest['stage']=='screen' else confirmation_gate(cells)
    result=dict(schema=SCHEMA,protocol_sha256=PROTOCOL_SHA256,dataset=spec['dataset'],
        stage=manifest['stage'],matrix_manifest_sha256=digest(manifest),
        cells={str(seed):rows for seed,rows in cells.items()},source_cells=sources,gate=gate,
        checkpoint_reselection=False,retired_test_access=False,objective_achieved=False,
        prospective_confirmation_required=True)
    result['receipt_sha256']=receipt_hash(result)
    return result
