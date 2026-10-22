"""Matched session information datasets retaining original seven-field preprocessing."""
from copy import deepcopy
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.features.price.sessions import NAMES


def preserve_preprocessing(source, generated, mode):
    if mode not in ('masked','full'):raise ValueError('augmented session mode required')
    if (len(source['names'])!=7 or generated['names']!=source['names']+list(NAMES)
            or generated['manifest'].get('session_context',{}).get('mode')!=mode
            or set(source['episodes'])!=set(generated['episodes'])
            or source['manifest']['included_splits']!=['train','validation']
            or generated['manifest']['included_splits']!=['train','validation']):
        raise ValueError('same declared development feature/panel contract required')
    extra=[]
    for key,e in generated['episodes'].items():
        old=source['episodes'][key]
        if (e['split'] not in ('train','validation') or e['symbol']!=old['symbol'] or e['split']!=old['split']
                or e['bars']!=old['bars'] or e['atr']!=old['atr']
                or len(e['features'])!=len(old['features'])
                or any(r[:7]!=v for r,v in zip(e['features'],old['features'],strict=True))
                or any(b['session']>('2020-12-31' if e['split']=='train' else '2023-12-31') for b in e['bars'])):
            raise ValueError('original bars/ATR/features and chronological splits must remain exact')
        if mode=='masked' and any(r[-4:]!=[0.]*4 for r in e['features']):raise ValueError('exact zero mask required')
        if e['split']=='train':extra.extend(r[-4:] for r in e['features'])
    fitted=Normalizer.fit(NAMES,extra)
    if (generated['normalizer']['means'][-4:]!=list(fitted.means)
            or generated['normalizer']['scales'][-4:]!=list(fitted.scales)):
        raise ValueError('added preprocessing must fit training rows only')
    result=deepcopy(generated)
    for key in ('means','scales'):result['normalizer'][key]=source['normalizer'][key]+list(getattr(fitted,key))
    result['manifest']['session_context_lineage']=dict(version=1,source_dataset=digest(source),
        generated_dataset=digest(generated),original_seven_normalizer_sha256=digest(source['normalizer']),
        preprocessing='Original seven means/scales frozen; four added columns fit expanded training rows only',
        panels='All source bars/ATR/original features exact; no test reconstructed')
    return result
