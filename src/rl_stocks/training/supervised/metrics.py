"""Predictive metrics/gates only; these are never trading returns."""
from statistics import mean
import numpy as np
import pandas as pd


def score(y,pred,sessions):
    y=np.asarray(y,dtype=np.float64);pred=np.asarray(pred,dtype=np.float64);sessions=np.asarray(sessions)
    if y.ndim!=1 or pred.shape!=y.shape or len(sessions)!=len(y) or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError('finite matched forecast vectors required')
    if any(str(d)[:4] not in ('2018','2019','2020') for d in sessions):raise ValueError('internal probe years only')
    error=(y-pred)**2;ics=[];undefined=0
    for day in sorted(set(sessions)):
        take=sessions==day
        if take.sum()<10:continue
        a=pd.Series(y[take]).rank().to_numpy();b=pd.Series(pred[take]).rank().to_numpy()
        if a.std()==0 or b.std()==0:undefined+=1;continue
        ics.append(float(np.corrcoef(a,b)[0,1]))
    return {'mse':float(error.mean()),'mae':float(np.abs(y-pred).mean()),
        'year_mse':{year:float(error[np.array([str(d).startswith(year) for d in sessions])].mean()) for year in ('2018','2019','2020') if any(str(d).startswith(year) for d in sessions)},
        'mean_daily_spearman_ic':float(mean(ics)) if ics else 0.,'ic_days':len(ics),'undefined_ic_days':undefined,'rows':len(y)}


def gate(metrics):
    baseline=metrics['constant'];comparisons={};passing=[]
    if baseline['mse']<=0 or any(metrics['ridge'][a]['mse']<=0 for a in ('masked','full')):raise ValueError('positive baseline MSE required')
    for arm in ('masked','full'):
        runs=metrics['mlp'][arm]
        if set(runs)!={'11','23','37'} or set(baseline['year_mse'])!={'2018','2019','2020'}:raise ValueError('all seeds and probe years required')
        ridge=metrics['ridge'][arm];mse=mean(r['mse'] for r in runs.values())
        reductions={name:1-mse/ref['mse'] for name,ref in [('constant',baseline),('ridge',ridge)]}
        seeds=sum(all(r['mse']<=.98*ref['mse'] for ref in [baseline,ridge]) for r in runs.values())
        ic=mean(r['mean_daily_spearman_ic'] for r in runs.values())
        yearly={year:{name:mean(r['year_mse'][year] for r in runs.values())<=ref['year_mse'][year] for name,ref in [('constant',baseline),('ridge',ridge)]} for year in ('2018','2019','2020')}
        passed=min(reductions.values())>=.02 and seeds>=2 and ic>=.02 and all(all(v.values()) for v in yearly.values())
        comparisons[arm]={'mean_mse':mse,'mse_reductions':reductions,'passing_seeds':seeds,'mean_daily_spearman_ic':ic,'yearly_noninferiority':yearly,'passed':passed}
        if passed:passing.append(arm)
    chosen=min(passing,key=lambda a:comparisons[a]['mean_mse']) if passing else None
    return {'passed':bool(passing),'candidate':chosen,'comparisons':comparisons,
        'paired_full_minus_masked_mse':{seed:metrics['mlp']['full'][seed]['mse']-metrics['mlp']['masked'][seed]['mse'] for seed in ('11','23','37')},'profitability_established':False,'ppo_training_performed':False}
