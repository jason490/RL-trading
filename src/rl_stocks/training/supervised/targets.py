"""Completed training-era labels; never future outcomes in model observations."""
import math
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.data.generation.datasets import Normalizer

HORIZON=20
FIT_END='2017-12-31'
PROBE_START='2018-01-01'
PROBE_END='2020-12-31'
FIT_CUTOFF=timestamp('2018-01-01T00:00:00+00:00')
PROBE_CUTOFF=timestamp('2021-01-01T00:00:00+00:00')


def completed_targets(dataset):
    if dataset['manifest'].get('included_splits')!=['train','validation']:
        raise ValueError('development-only source required')
    if dataset['manifest'].get('unverified_override') is not False:
        raise ValueError('verified source required')
    if any(e['split'] not in ('train','validation') for e in dataset['episodes'].values()):
        raise ValueError('test episodes forbidden')
    names=dataset['names'];records=[]
    for key,episode in sorted(dataset['episodes'].items()):
        if episode['split']!='train':continue # Never consume project validation features/labels.
        bars=episode['bars'];features=episode['features']
        if len(bars)!=len(features) or any(b['session']>PROBE_END for b in bars):
            raise ValueError('training episode bounds mismatch')
        sessions=[b['session'] for b in bars]
        if sessions!=sorted(set(sessions)):raise ValueError('ordered unique training sessions required')
        for i in range(len(bars)-HORIZON):
            decision=bars[i];entry=bars[i+1];exit=bars[i+HORIZON]
            if timestamp(decision['available_time'])>=timestamp(entry['open_time']):
                raise ValueError('feature not available before next open')
            completed=timestamp(exit['available_time'])
            if completed<FIT_CUTOFF and exit['session']<=FIT_END:fold='fit'
            elif decision['session']>=PROBE_START and completed<PROBE_CUTOFF and exit['session']<=PROBE_END:fold='probe'
            else:continue # Purge targets straddling fit/probe or training boundaries.
            x=features[i]
            if len(x)!=len(names) or not all(math.isfinite(v) for v in x):raise ValueError('invalid causal feature row')
            price=entry['open']
            if not math.isfinite(price) or price<=0:raise ValueError('positive entry price required')
            shares=1.;cash=0.
            # Corporate actions at the entry open precede entry, so entry-day
            # dividends/splits do not apply to newly opened unit inventory.
            for future in bars[i+2:i+HORIZON+1]:
                split=future.get('split',1.);dividend=future.get('dividend',0.)
                if not math.isfinite(split) or split<=0 or not math.isfinite(dividend) or dividend<0:
                    raise ValueError('invalid target corporate action')
                shares*=split;cash+=shares*dividend
            value=shares*exit['close']+cash
            if not math.isfinite(value) or value<=0:raise ValueError('invalid target terminal value')
            records.append({'episode':key,'symbol':episode['symbol'],'fold':fold,
                'decision_session':decision['session'],'feature_available_time':decision['available_time'],
                'entry_session':entry['session'],'exit_session':exit['session'],
                'label_available_time':exit['available_time'],'features':list(x),
                'target_log_total_return':math.log(value/price)})
    if {r['fold'] for r in records}!={'fit','probe'}:raise ValueError('both completed target folds required')
    return records


def fit_preprocessing(records,names):
    """Fit on pre-2018 inputs only; ignore pooled source normalization."""
    fit=[r for r in records if r['fold']=='fit']
    if not fit or any(timestamp(r['label_available_time'])>=FIT_CUTOFF for r in fit):
        raise ValueError('completed fit labels required')
    return Normalizer.fit(names,[r['features'] for r in fit])
