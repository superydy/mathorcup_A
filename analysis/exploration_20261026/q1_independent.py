"""Fixed structural comparison: shared vs independent outputs; L1/L2 loss.
AI assistance: OpenAI Codex, 2026-10-26. Exploratory, no original overwrite.
"""
from pathlib import Path
import os,sys,json,time,warnings
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr,t
from threadpoolctl import threadpool_limits
threadpool_limits(2);sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore',message='X does not have valid feature names')
W=Path(__file__).resolve().parent;ROOT=W.parents[1];DATA=ROOT if (ROOT/'1.csv').exists() else ROOT/'mathorcup_A';base=ROOT/'analysis/A_feasibility/floor_ablation'
x=pd.read_csv(DATA/'1.csv').set_index('beer');y=pd.read_csv(DATA/'4.csv').set_index('beer').loc[x.index]
X=x.drop(columns=['beer_id','tasting_category_fine']).to_numpy(float);Y=y.drop(columns=['beer_id','tasting_category_fine']).to_numpy(float);fields=y.drop(columns=['beer_id','tasting_category_fine']).columns.to_numpy()
splits=json.loads((base/'split_and_floor_log.json').read_text(encoding='utf-8'))
original=np.load(base/'baseline_oof.npz',allow_pickle=True)['prediction']
pred={n:np.full_like(original,np.nan) for n in ['ET_independent','LGBM_L1','LGBM_L2','ET_half_independent']}
start=time.perf_counter();rows=[]
for outer,sp in enumerate(splits):
    r,f=sp['repeat'],sp['fold'];tr=x.index.get_indexer(sp['train_beer']);te=x.index.get_indexer(sp['test_beer'])
    imp=SimpleImputer(strategy='median');a=imp.fit_transform(X[tr]);b=imp.transform(X[te])
    for j in range(Y.shape[1]):
        et=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.7,random_state=20261025+outer,n_jobs=2)
        pred['ET_independent'][r,te,j]=et.fit(a,Y[tr,j]).predict(b)
        for name,obj in [('LGBM_L1','regression_l1'),('LGBM_L2','regression')]:
            fit=LGBMRegressor(n_estimators=200,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=10,
                reg_lambda=10,reg_alpha=.05,colsample_bytree=.7,objective=obj,random_state=20261025+outer,n_jobs=2,verbosity=-1)
            pred[name][r,te,j]=fit.fit(a,Y[tr,j]).predict(b)
    pred['ET_half_independent'][r,te]=.5*(original[r,te]+pred['ET_independent'][r,te])
    row=dict(outer=outer,n_train=len(tr),n_test=len(te),ET_original=float(abs(Y[te]-original[r,te]).mean()))
    for n,p in pred.items():row[n]=float(abs(Y[te]-p[r,te]).mean())
    rows.append(row);pd.DataFrame(rows).to_csv(W/'q1_independent_fold_metrics.csv',index=False)
    np.savez_compressed(W/'q1_independent_predictions.npz',**pred,fields=fields,beer=x.index.to_numpy())
    print(f'Independent-target fold {outer+1}/15 '+', '.join(f'{n}={row[n]:.4f}' for n in pred)+f'; elapsed {time.perf_counter()-start:.0f}s',flush=True)
df=pd.DataFrame(rows);ratio=(df.n_test/df.n_train).mean();summary={};dim=[]
for n,pp in pred.items():
    assert np.isfinite(pp).all()
    mae=abs(Y[None,:,:]-pp).mean(axis=(0,1));r2=np.mean([r2_score(Y,p,multioutput='raw_values') for p in pp],axis=0)
    rho=np.mean([[spearmanr(Y[:,j],p[:,j]).statistic for j in range(Y.shape[1])] for p in pp],axis=0)
    dd=(df.ET_original-df[n])/df.ET_original;mu=float(dd.mean());se=float(np.sqrt((1/15+ratio)*dd.var(ddof=1)));ci=[mu-t.ppf(.975,14)*se,mu+t.ppf(.975,14)*se]
    summary[n]=dict(MAE=float(mae.mean()),R2=float(r2.mean()),Spearman=float(rho.mean()),relative_MAE_gain_vs_ET=mu,corrected_approx_95pct_CI=ci,
        meets_predeclared_adoption=bool(mu>=.01 and ci[0]>0))
    dim.extend(dict(method=n,field=str(f),MAE=float(a),R2=float(b),Spearman=float(c)) for f,a,b,c in zip(fields,mae,r2,rho))
pd.DataFrame(dim).to_csv(W/'q1_independent_dimension_metrics.csv',index=False)
summary['metadata']=dict(elapsed_seconds=time.perf_counter()-start,n_outer_splits=15,hyperparameter_search=False,
    note='New fixed structural comparisons after reading source methods. Reused outer folds, exploratory; no target-wise selection based on outer outcomes.')
(W/'q1_independent_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
