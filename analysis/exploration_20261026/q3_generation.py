"""Generator-only ablations; no changes to frozen OT or official results.
AI assistance: OpenAI Codex, 2026-10-26; exact model identifier unavailable.
"""
from pathlib import Path
import ast,json,sys,os,time
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
src=(ROOT/'src/q3_style_generation/latent_gaussian_ot.py').read_text(encoding='utf-8')
tree=ast.parse(src)
for name in ['EmpiricalMarginals','LatentStyles']:
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(W/'q3_generation.py'),'exec'),globals())
x=pd.read_csv(DATA/'1.csv').set_index('beer');X=x.drop(columns=['beer_id','tasting_category_fine'])
labels=x.tasting_category_fine.to_numpy();fields=list(X);p=len(fields)
splits=json.loads((ROOT/'analysis/q3/generator_holdout_splits.json').read_text(encoding='utf-8'))
styles=['Tripel','Kriek','Lambic'];methods=['common_latent_original','conditional_local_corr','conditional_pooled_corr','conditional_mixed_corr']

def as_corr(S):
    sd=np.sqrt(np.maximum(np.diag(S),1e-12));C=S/sd[:,None]/sd[None,:]
    C=(C+C.T)*.5;np.fill_diagonal(C,1.0)
    return C
def factor(S):
    v,Q=np.linalg.eigh(S);return Q*np.sqrt(np.maximum(v,0))[None,:]
def corr(A):
    D=A-A.mean(0);sd=np.sqrt((D*D).sum(0));return (D.T@D)/np.maximum(sd[:,None]*sd[None,:],1e-15),sd>1e-12

rows=[];corrrows=[];start=time.perf_counter()
for fold,sp in enumerate(splits):
    tr=X.index.get_indexer(sp['train_beer']);te=X.index.get_indexer(sp['test_beer'])
    fit=LatentStyles().fit(X.iloc[tr],labels[tr]);B=fit.imp.transform(X.iloc[tr]);test=fit.imp.transform(X.iloc[te]);scaler=StandardScaler().fit(B)
    local={};rankrows=[]
    for s in np.unique(labels[tr]):
        bs=B[labels[tr]==s];g=EmpiricalMarginals().fit(bs);z=g.transform(bs)
        local[s]=(g,as_corr(LedoitWolf().fit(z).covariance_),len(bs))
        sd=z.std(0);rankrows.append((z-z.mean(0))/np.where(sd>1e-12,sd,1))
    pooled=as_corr(LedoitWolf().fit(np.vstack(rankrows)).covariance_)
    centers={method:[[],[]] for method in methods};rng=np.random.default_rng(20261026+fold)
    floor=B.min(0);floor_cols=np.where((B==floor).mean(0)>=.10)[0]
    for s in styles:
        at=np.where(labels[te]==s)[0];testz=scaler.transform(test[at]);trainz=scaler.transform(B[labels[tr]==s])
        noises=rng.normal(size=(300,p));g,C,n=local[s];weight=n/(n+20)
        generated={'common_latent_original':fit.sample(s,noises)}
        for method,R in [('conditional_local_corr',C),('conditional_pooled_corr',pooled),('conditional_mixed_corr',weight*C+(1-weight)*pooled)]:
            generated[method]=g.inverse(noises@factor(R).T)
        denom=cdist(testz,trainz).min(1).mean()/np.sqrt(p)
        for method,A in generated.items():
            z=scaler.transform(A)
            rows.append(dict(fold=fold,style=s,method=method,n_train=n,n_test=len(at),
                mean_coordinate_MAE_standardized=float(abs(z.mean(0)-testz.mean(0)).mean()),
                mean_coordinate_Wasserstein_standardized=float(np.mean([wasserstein_distance(z[:,j],testz[:,j]) for j in range(p)])),
                floor_fraction_MAE_floor_columns=float(abs((A[:,floor_cols]==floor[floor_cols]).mean(0)-(test[at][:,floor_cols]==floor[floor_cols]).mean(0)).mean()),
                synthetic_to_real_NN_ratio=float(cdist(z,trainz).min(1).mean()/np.sqrt(p)/denom),
                exact_training_copy_count=int((cdist(z,trainz).min(1)<1e-12).sum())))
            centers[method][0].append(z-z.mean(0));centers[method][1].append(testz-testz.mean(0))
    for method,(a,b) in centers.items():
        cg,vg=corr(np.vstack(a));cr,vr=corr(np.vstack(b));ids=np.where(vg&vr)[0];ii,jj=np.triu_indices(len(ids),1)
        corrrows.append(dict(fold=fold,method=method,pooled_within_style_correlation_RMSE=float(np.sqrt(np.mean((cg[np.ix_(ids,ids)][ii,jj]-cr[np.ix_(ids,ids)][ii,jj])**2)))))
    print(f'Generator fold {fold+1}/3 done',flush=True)
df=pd.DataFrame(rows);cf=pd.DataFrame(corrrows)
df.to_csv(W/'q3_generator_fold_metrics.csv',index=False);cf.to_csv(W/'q3_correlation_fold_metrics.csv',index=False)
agg=df.groupby('method').mean(numeric_only=True)
agg['pooled_within_style_correlation_RMSE']=cf.groupby('method').pooled_within_style_correlation_RMSE.mean()
agg.to_csv(W/'q3_generator_summary.csv')
(W/'q3_summary.json').write_text(json.dumps(dict(results=agg.to_dict('index'),elapsed_seconds=time.perf_counter()-start,
    note='Exploratory new generator variants; same holdout folds and Gaussian noises, no OT change; eigenfactor implementation changes may affect exact paired noise coupling.'),indent=2),encoding='utf-8')
print(agg.to_string(),flush=True)
