"""Joint module permutation, globally and within recorded style.
AI assistance: OpenAI Codex, 2026-10-26. Repeated folds are dependent.
"""
from pathlib import Path
import os,sys,json,time
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
threadpool_limits(2);sys.stdout.reconfigure(encoding='utf-8')
W=Path(__file__).resolve().parent;ROOT=W.parents[1];DATA=ROOT if (ROOT/'1.csv').exists() else ROOT/'mathorcup_A'
x=pd.read_csv(DATA/'1.csv').set_index('beer');y=pd.read_csv(DATA/'4.csv').set_index('beer').loc[x.index]
X=x.drop(columns=['beer_id','tasting_category_fine']);Y=y.drop(columns=['beer_id','tasting_category_fine']);styles=x.tasting_category_fine.to_numpy()
mapping=pd.read_csv(ROOT/'analysis/q2/chemical_mapping_and_modules.csv').set_index('field').loc[X.columns]
groups={m:np.where(mapping.module.to_numpy()==m)[0] for m in mapping.module.unique() if (mapping.module==m).sum()>=2}
assert len(groups)==19
splits=json.loads((ROOT/'analysis/A_feasibility/floor_ablation/split_and_floor_log.json').read_text(encoding='utf-8'))
rows=[];runtimes=[];start=time.perf_counter()
for outer,sp in enumerate(splits):
    tr=x.index.get_indexer(sp['train_beer']);te=x.index.get_indexer(sp['test_beer'])
    imp=SimpleImputer(strategy='median');a=imp.fit_transform(X.iloc[tr]);b=imp.transform(X.iloc[te])
    fit=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.7,random_state=20261025+outer,n_jobs=2).fit(a,Y.iloc[tr])
    labs=styles[te];valid=np.array([np.sum(labs==s)>=2 for s in labs]);b=b[valid];labs=labs[valid];truth=Y.iloc[te].to_numpy()[valid]
    original=fit.predict(b);base=abs(original-truth).mean(0);rng=np.random.default_rng(20261026+outer)
    for m,cols in groups.items():
        scores={kind:[] for kind in ['global','within_style']}
        for rep in range(3):
            global_idx=rng.permutation(len(b));within=np.arange(len(b))
            for s in np.unique(labs):
                ix=np.where(labs==s)[0];within[ix]=rng.permutation(ix)
            for kind,ix in [('global',global_idx),('within_style',within)]:
                bb=b.copy();bb[:,cols]=b[ix][:,cols]
                scores[kind].append(abs(fit.predict(bb)-truth).mean(0)-base)
        for kind,arr in scores.items():
            vals=np.mean(arr,axis=0)
            for j,f in enumerate(Y.columns):rows.append(dict(outer=outer,module=m,field=f,kind=kind,importance=float(vals[j]),relative_MAE_increase=float(vals[j]/base[j]),module_size=len(cols),n_test=len(b)))
    print(f'Joint module permutation {outer+1}/15',flush=True)
df=pd.DataFrame(rows);df.to_csv(W/'q2_module_fold_importance.csv',index=False)
agg=df.groupby(['field','module','kind'],as_index=False).agg(mean_importance=('importance','mean'),relative_MAE_increase=('relative_MAE_increase','mean'),positive_fold_share=('importance',lambda s:float((s>0).mean())),module_size=('module_size','first'))
agg['rank']=agg.groupby(['field','kind']).mean_importance.rank(ascending=False,method='min')
agg.to_csv(W/'q2_module_summary.csv',index=False)
top=agg[agg['rank']<=3].copy();top.to_csv(W/'q2_module_top3.csv',index=False)
(W/'q2_summary.json').write_text(json.dumps(dict(modules=len(groups),outer_splits=15,repetitions=3,elapsed_seconds=time.perf_counter()-start,
    note='Module-level predictive dependence only. Joint shuffling preserves within-module combinations. Positive-fold shares are dependent descriptive counts, not bootstrap confidence.'),indent=2),encoding='utf-8')
print('Grouped permutation complete',flush=True)
