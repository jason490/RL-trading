"""Small CPU signal models; constructing these never fits data."""
import numpy as np
import torch
from torch import nn


def network():
    return nn.Sequential(nn.Linear(10,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))


def design(raw,normalizer,arm):
    x=np.array(raw,dtype=np.float64,copy=True)
    if x.ndim!=2 or x.shape[1]!=10 or not np.isfinite(x).all():raise ValueError('finite10-field feature matrix required')
    if arm not in ('masked','full'):raise ValueError('unknown signal arm')
    if arm=='masked':x[:,7:]=0.
    mean=np.asarray(normalizer['means']);scale=np.asarray(normalizer['scales'])
    if mean.shape!=(10,) or scale.shape!=(10,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('invalid frozen signal normalization')
    return (x-mean)/scale


def predict(model,x):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):result.append(model(torch.as_tensor(x[start:start+1024],dtype=torch.float32)).squeeze(-1).numpy())
    return np.concatenate(result) if result else np.empty(0,dtype=np.float32)


def ridge_fit(x,y):
    z=np.column_stack((np.ones(len(x)),x));penalty=np.eye(11)*.01;penalty[0,0]=0.
    return np.linalg.solve(z.T@z/len(z)+penalty,z.T@y/len(z))


def ridge_predict(coefficients,x):
    return np.column_stack((np.ones(len(x)),x))@coefficients
