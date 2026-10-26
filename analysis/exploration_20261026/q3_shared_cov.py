"""Post-result targeted ablation: preserve marginal Gaussian mean/variance.
AI assistance: OpenAI Codex, 2026-10-26. No frozen result changes.
"""
from pathlib import Path
import os,ast,json,sys,time
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import pandas as pd
from scipy.stats import norm,wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
threadpool_limits(2);sys.stdout.reconfigure(encoding='utf-8')
W=Path(__file__).resolve().parent;ROOT=W.parents[1];DATA=ROOT if (ROOT/'1.csv').exists() else ROOT/'mathorcup_A'
src=(ROOT/'src/q3_style_generation/latent_gaussian_ot.py').read_text(encoding='utf-8');tree=ast.parse(src)
for name in ['EmpiricalMarginals','LatentStyles']:
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),__file__,'exec'),globals())
x=pd.read_csv(DATA/'1.csv').set_index('beer');X=x.drop(columns=['beer_id','tasting_category_fine']);labs=x.tasting_category_fine.to_numpy();p=X.shape[1]
splits=json.loads((ROOT/'analysis/q3/generator_holdout_splits.json').read_text(encoding='utf-8'))
styles=['Tripel','Kriek','Lambic'];methods=['common_original','common_shared_mix','common_shared_only']
def correlation(S):
    sd=np.sqrt(np.maximum(np.diag(S),1e-12));c=S/sd[:,None]/sd[None,:];c=(c+c.T)/2;np.fill_diagonal(c,1);return c
def root(S):
    v,q=np.linalg.eigh(S);return q*np.sqrt(np.maximum(v,0))[None,:]
def corr(A):
    d=A-A.mean(0);sd=np.sqrt((d*d).sum(0));return (d.T@d)/np.maximum(sd[:,None]*sd[None,:],1e-15),sd>1e-12
rows=[];cr=[];start=time.perf_counter()
for fold,sp in enumerate(splits):
    tr=X.index.get_indexer(sp['train_beer']);te=X.index.get_indexer(sp['test_beer'])
    fit=LatentStyles().fit(X.iloc[tr].to_numpy(),labs[tr]);B=fit.imp.transform(X.iloc[tr].to_numpy());Bt=fit.imp.transform(X.iloc[te].to_numpy());scaler=StandardScaler().fit(B)
    rs=[]
    for s in np.unique(labs[tr]):
        z=fit.Z[labs[tr]==s];sd=z.std(0);rs.append((z-z.mean(0))/np.where(sd>1e-12,sd,1))
    shared=correlation(LedoitWolf().fit(np.vstack(rs)).covariance_)
    roots={}
    for s in styles:
        m=fit.models[s];cov=m.covariance_;sd=np.sqrt(np.diag(cov));c=correlation(cov);n=int((labs[tr]==s).sum());w=n/(n+20)
        for method,C in [('common_original',c),('common_shared_mix',w*c+(1-w)*shared),('common_shared_only',shared)]:
            S=sd[:,None]*C*sd[None,:]
            assert np.max(abs(np.diag(S)-np.diag(cov)))<1e-10
            roots[s,method]=root(S)
    floor=B.min(0);fc=np.where((B==floor).mean(0)>=.1)[0]
    for seed in range(5):
        rng=np.random.default_rng(20261026+fold+1000*seed);centers={m:[[],[]] for m in methods}
        for s in styles:
            at=np.where(labs[te]==s)[0];tz=scaler.transform(Bt[at]);trainz=scaler.transform(B[labs[tr]==s]);noise=rng.normal(size=(300,p))
            denominator=cdist(tz,trainz).min(1).mean()/np.sqrt(p)
            for method in methods:
                A=fit.G.inverse(noise@roots[s,method].T+fit.models[s].location_);z=scaler.transform(A)
                rows.append(dict(fold=fold,seed=seed,style=s,method=method,MAE=float(abs(z.mean(0)-tz.mean(0)).mean()),
                    Wasserstein=float(np.mean([wasserstein_distance(z[:,j],tz[:,j]) for j in range(p)])),
                    floor_error=float(abs((A[:,fc]==floor[fc]).mean(0)-(Bt[at][:,fc]==floor[fc]).mean(0)).mean()),
                    NN_ratio=float(cdist(z,trainz).min(1).mean()/np.sqrt(p)/denominator)))
                centers[method][0].append(z-z.mean(0));centers[method][1].append(tz-tz.mean(0))
        for method,(a,b) in centers.items():
            cg,vg=corr(np.vstack(a));ct,vt=corr(np.vstack(b));ids=np.where(vg&vt)[0];ii,jj=np.triu_indices(len(ids),1)
            cr.append(dict(fold=fold,seed=seed,method=method,correlation_RMSE=float(np.sqrt(np.mean((cg[np.ix_(ids,ids)][ii,jj]-ct[np.ix_(ids,ids)][ii,jj])**2)))))
        print(f'Shared covariance fold {fold+1}/3 seed {seed+1}/5',flush=True)
df=pd.DataFrame(rows);cf=pd.DataFrame(cr)
df.to_csv(W/'q3_shared_cov_fold_seed.csv',index=False);cf.to_csv(W/'q3_shared_cov_correlation.csv',index=False)
ag=df.groupby('method')[['MAE','Wasserstein','floor_error','NN_ratio']].mean();ag['correlation_RMSE']=cf.groupby('method').correlation_RMSE.mean()
ag.to_csv(W/'q3_shared_cov_summary.csv')
summary=dict(results=ag.to_dict('index'),elapsed_seconds=time.perf_counter()-start,sampling_seeds=5,holdout_folds=3,
    interpretation='All latent marginal means and variances are identical across methods; differences in finite sampled marginals may result from Monte Carlo coupling. Reused 3 folds, exploratory, no independent confirmation or OT replacement.')
(W/'q3_shared_cov_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(ag.to_string(),flush=True)
