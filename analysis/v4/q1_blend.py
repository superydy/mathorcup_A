"""L1/L2 comparison, inner OOF simplex MAE weights. AI assistance: OpenAI Codex, 2026-10-26."""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,hstack,vstack,eye
from scipy.stats import t,spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Parallel,delayed
from common import ROOT,OUT,load,splits
sys.stdout.reconfigure(encoding='utf-8')
X,Y,style=load();A=X.to_numpy();yy=Y.to_numpy();Q=Y.shape[1]
exp=ROOT/'analysis/exploration_20261026'
orig=np.load(ROOT/'analysis/A_feasibility/floor_ablation/baseline_oof.npz',allow_pickle=True)['prediction']
ind=np.load(exp/'q1_independent_predictions.npz',allow_pickle=True)['ET_independent']
prev=np.load(exp/'q1_nested_checkpoint.npz',allow_pickle=True)
names=[a[0] for a in json.loads((exp/'q1_candidates.json').read_text())]
pls=prev['all_candidate'][names.index('PLS8')]
meta=pd.read_csv(ROOT/'analysis/A_feasibility/day1/sensory_metadata.csv',index_col=0).loc[Y.columns]
types=meta.DataType.to_numpy()

def experts(a,y,b,seed):
    imp=SimpleImputer(strategy='median');at=imp.fit_transform(a);bt=imp.transform(b)
    shared=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.7,random_state=seed,n_jobs=2).fit(at,y).predict(bt)
    def one(j):return ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.7,random_state=seed,n_jobs=1).fit(at,y[:,j]).predict(bt)
    independent=np.stack(Parallel(n_jobs=4,prefer='threads')(delayed(one)(j) for j in range(y.shape[1])),axis=1)
    qt=QuantileTransformer(n_quantiles=min(100,len(a)),output_distribution='normal',random_state=seed)
    linear=PLSRegression(n_components=8,scale=False,max_iter=1000).fit(qt.fit_transform(at),y).predict(qt.transform(bt))
    return np.stack([shared,independent,linear],axis=-1)

def weights(P,y):
    # P: observations x experts. min sum(t), subject to +/- residual <=t, w>=0, sum(w)=1.
    n,k=P.shape;sp=csr_matrix(P);ident=eye(n,format='csr')
    C=vstack([hstack([sp,-ident]),hstack([-sp,-ident])],format='csr')
    eq=csr_matrix(np.r_[np.ones(k),np.zeros(n)][None,:])
    fit=linprog(np.r_[np.zeros(k),np.ones(n)/n],A_ub=C,b_ub=np.r_[y,-y],A_eq=eq,b_eq=[1],bounds=[(0,None)]*(k+n),method='highs')
    assert fit.success,fit.message
    w=np.maximum(fit.x[:k],0);return w/w.sum()

pred={n:np.full_like(orig,np.nan) for n in ['three_equal','ET_pair','group_LP','attribute_LP']}
rows=[];wrows=[];start=time.perf_counter()
for outer,sp in enumerate(splits()):
    rep,f=sp['repeat'],sp['fold'];tr=X.index.get_indexer(sp['train_beer']);te=X.index.get_indexer(sp['test_beer'])
    inner=np.full((len(tr),Q,3),np.nan)
    inn=StratifiedKFold(2,shuffle=True,random_state=202610260+outer)
    for k,(ia,ib) in enumerate(inn.split(A[tr],style.iloc[tr])):
        inner[ib]=experts(A[tr[ia]],yy[tr[ia]],A[tr[ib]],20261025+outer*10+k)
    cp=np.stack([orig[rep,te],ind[rep,te],pls[rep,te]],axis=-1)
    pred['three_equal'][rep,te]=cp.mean(-1);pred['ET_pair'][rep,te]=cp[:,:,:2].mean(-1)
    wg=np.empty((Q,3));wa=np.empty((Q,3))
    for typ in ['numeric','categorical']:
        mask=types==typ;w=weights(inner[:,mask,:].reshape(-1,3),yy[tr][:,mask].ravel());wg[mask]=w
    for j in range(Q):wa[j]=weights(inner[:,j,:],yy[tr,j])
    pred['group_LP'][rep,te]=(cp*wg[None,:,:]).sum(-1)
    pred['attribute_LP'][rep,te]=(cp*wa[None,:,:]).sum(-1)
    for method,ww in [('group_LP',wg),('attribute_LP',wa)]:
        for j,name in enumerate(Y.columns):wrows.append(dict(outer=outer,method=method,field=name,type=types[j],shared_ET=ww[j,0],independent_ET=ww[j,1],PLS=ww[j,2]))
    row=dict(outer=outer,repeat=rep,fold=f,n_train=len(tr),n_test=len(te))
    for n,p in pred.items():row[n]=float(abs(p[rep,te]-yy[te]).mean())
    rows.append(row);pd.DataFrame(rows).to_csv(OUT/'q1_blend_fold_metrics.csv',index=False)
    pd.DataFrame(wrows).to_csv(OUT/'q1_blend_inner_weights.csv',index=False)
    np.savez_compressed(OUT/'q1_blend_oof.npz',**pred,fields=Y.columns.to_numpy(),beer=X.index.to_numpy())
    np.savez_compressed(OUT/f'q1_inner_experts_{outer:02d}.npz',prediction=inner,truth=yy[tr],train_beer=X.index[tr].to_numpy())
    print(f'Q1 blend {outer+1}/15 '+', '.join(f'{n}={row[n]:.4f}' for n in pred)+f'; {time.perf_counter()-start:.0f}s',flush=True)
df=pd.DataFrame(rows);ratio=(df.n_test/df.n_train).mean();summary={};dims=[]
for n,p in pred.items():
    assert np.isfinite(p).all()
    r=np.mean([__import__('sklearn.metrics',fromlist=['r2_score']).r2_score(yy,v,multioutput='raw_values') for v in p],axis=0)
    rho=np.mean([[spearmanr(yy[:,j],v[:,j]).statistic for j in range(Q)] for v in p],axis=0)
    mae=abs(p-yy[None,:,:]).mean(axis=(0,1));summary[n]=dict(MAE=float(mae.mean()),R2=float(r.mean()),Spearman=float(rho.mean()))
    dims.extend(dict(method=n,field=f,MAE=float(a),R2=float(b),Spearman=float(c)) for f,a,b,c in zip(Y.columns,mae,r,rho))
def comparison(a,b):
    d=(df[b]-df[a])/df[b];m=float(d.mean());se=float(np.sqrt((1/15+ratio)*d.var(ddof=1)));ci=[m-t.ppf(.975,14)*se,m+t.ppf(.975,14)*se]
    return dict(relative_MAE_gain=m,approx_95pct_CI=ci,R2_difference=summary[a]['R2']-summary[b]['R2'],
        passes=bool(m>=.01 and ci[0]>0 and summary[a]['R2']>=summary[b]['R2']))
comp={f'{a}_vs_{b}':comparison(a,b) for a,b in [('group_LP','three_equal'),('group_LP','ET_pair'),('attribute_LP','three_equal'),('attribute_LP','ET_pair'),('attribute_LP','group_LP')]}
selected='ET_pair'
if comp['group_LP_vs_three_equal']['passes'] and comp['group_LP_vs_ET_pair']['passes']:selected='group_LP'
if comp['attribute_LP_vs_three_equal']['passes'] and comp['attribute_LP_vs_ET_pair']['passes'] and comp['attribute_LP_vs_group_LP']['passes']:selected='attribute_LP'
pd.DataFrame(dims).to_csv(OUT/'q1_blend_dimension_metrics.csv',index=False)
(OUT/'q1_blend_summary.json').write_text(json.dumps(dict(metrics=summary,comparisons=comp,selected=selected,elapsed_seconds=time.perf_counter()-start,
    status='Reused 15 outer splits; true inner OOF fits; internal exploratory decision, no independent confirmation.'),indent=2),encoding='utf-8')
print('Selected Q1: '+selected,flush=True)
