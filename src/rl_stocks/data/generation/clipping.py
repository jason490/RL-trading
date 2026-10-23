"""Opt-in frozen market-z clipping, preserving default preprocessing byte contracts."""
from copy import deepcopy
from dataclasses import dataclass
import math
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.features.registry.assemble import NAMES


@dataclass(frozen=True)
class ClippedNormalizer(Normalizer):
    clip_abs: float

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.clip_abs,bool) or not isinstance(self.clip_abs,(int,float)) or not math.isfinite(self.clip_abs) or self.clip_abs<=0:
            raise ValueError('positive finite explicit clipping bound required')

    def transform(self,values,names):
        return [max(-self.clip_abs,min(self.clip_abs,z)) for z in super().transform(values,names)]

    @classmethod
    def fit(cls,*args,**kwargs):
        raise ValueError('wrap explicitly frozen base preprocessing; clipping does not fit')


def clipped_dataset(source):
    if (source['schema']!='daily-indicators-v1' or source['names']!=list(NAMES)
        or set(source['normalizer'])!={'names','means','scales'}
        or source['manifest']['included_splits']!=['train','validation']
        or 'market_input_clipping' in source['manifest']):
        raise ValueError('original baseline train/development preprocessing required')
    frozen=source['normalizer'];ClippedNormalizer(tuple(frozen['names']),tuple(frozen['means']),tuple(frozen['scales']),5.)
    for e in source['episodes'].values():
        if e['split'] not in ('train','validation') or len(e['bars'])!=len(e['features']) or len(e['bars'])!=len(e['atr']):
            raise ValueError('aligned train/development episodes required')
        if any(b['session']>('2020-12-31' if e['split']=='train' else '2023-12-31') for b in e['bars']):
            raise ValueError('source split cutoff violated')
        if any(len(row)!=len(NAMES) or not all(math.isfinite(v) for v in row) for row in e['features']):
            raise ValueError('finite source market features required')
    result=deepcopy(source);result['schema']='daily-indicators-v1-zclip5-v1';result['normalizer']['clip_abs']=5.
    result['manifest']['market_input_clipping']=dict(version=1,clip_abs=5.,source_dataset=digest(source),
        source_normalizer_sha256=digest(frozen),raw_episodes_sha256=digest(source['episodes']),
        transform='Clip only frozen standardized market features before float32 observation cast; portfolio state untouched',
        preserved='All raw bars/features/ATR/availability/splits/rewards/costs/simulation and original means/scales exact; no fitting or test reconstructed')
    return result


def validate_clipped_dataset(source,candidate):
    if digest(candidate)!=digest(clipped_dataset(source)):raise ValueError('exact clipping-only payload required')
    return candidate
