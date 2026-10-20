"""Portable finite-input regression forests, independent of sklearn/pickle."""
import numpy as np

FIELDS=('children_left','children_right','feature','threshold','value')


def validate_tree(tree):
    if set(tree)!=set(FIELDS):raise ValueError('tree field set mismatch')
    arrays={k:np.asarray(v) for k,v in tree.items()};n=len(arrays['feature'])
    if not 1<=n<=511 or any(v.shape!=(n,) for v in arrays.values()):raise ValueError('bounded flat tree arrays required')
    for name in ['children_left','children_right','feature']:
        if arrays[name].dtype.kind not in 'iu':raise ValueError('integer tree indices required')
    if not np.isfinite(arrays['threshold']).all() or not np.isfinite(arrays['value']).all():raise ValueError('finite tree parameters required')
    seen=set();stack=[(0,0)]
    while stack:
        node,depth=stack.pop()
        if node in seen or not 0<=node<n or depth>8:raise ValueError('cyclic, invalid or overdeep tree')
        seen.add(node);left=int(arrays['children_left'][node]);right=int(arrays['children_right'][node])
        if left==-1:
            if right!=-1:raise ValueError('inconsistent leaf')
        else:
            if not 0<=int(arrays['feature'][node])<10 or right<0:raise ValueError('invalid split')
            stack.extend([(left,depth+1),(right,depth+1)])
    if len(seen)!=n:raise ValueError('unreachable tree node')
    return {k:np.array(v,copy=True) for k,v in arrays.items()}


def export_forest(model,path):
    if model.n_features_in_!=10 or model.n_outputs_!=1:raise ValueError('ten-feature scalar forest required')
    arrays={}
    for i,estimator in enumerate(model.estimators_):
        t=estimator.tree_
        tree=validate_tree({name:np.asarray(getattr(t,name)).reshape(-1) for name in FIELDS})
        for name,value in tree.items():arrays[f'{i:03d}_{name}']=value
    np.savez_compressed(path,**arrays)


def load_forest(path,expected_trees):
    if not 1<=expected_trees<=128:raise ValueError('bounded tree count required')
    expected={f'{i:03d}_{name}' for i in range(expected_trees) for name in FIELDS}
    with np.load(path,allow_pickle=False) as source:
        if set(source.files)!=expected:raise ValueError('forest array file set mismatch')
        return [validate_tree({name:source[f'{i:03d}_{name}'] for name in FIELDS}) for i in range(expected_trees)]


def predict_forest(trees,features):
    raw=np.asarray(features,dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]!=10 or not np.isfinite(raw).all() or not trees:raise ValueError('finite ten-field forest inputs required')
    with np.errstate(over='ignore'):x=raw.astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('forest input exceeds float32 range')
    total=np.zeros(len(x),dtype=np.float64)
    for tree in trees:
        nodes=np.zeros(len(x),dtype=np.int64)
        for _ in range(9):
            active=np.flatnonzero(tree['children_left'][nodes]!=-1)
            if not len(active):break
            current=nodes[active];left=x[active,tree['feature'][current]]<=tree['threshold'][current]
            nodes[active]=np.where(left,tree['children_left'][current],tree['children_right'][current])
        if np.any(tree['children_left'][nodes]!=-1):raise ValueError('forest traversal exceeded depth bound')
        total+=tree['value'][nodes]
    return total/len(trees)
