"""Bounded nested comparison; exploratory reuse of existing outer splits.
AI assistance: OpenAI Codex, 2026-10-26; model identifier unavailable.
Never overwrites frozen results or the paper.
"""
from pathlib import Path
import os,json,time,warnings,hashlib,sys
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score,mean_absolute_error
from scipy.stats import spearmanr,t
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
sys.stdout.reconfigure(encoding='utf-8')
threadpool_limits(2)
warnings.filterwarnings('ignore',message='The least populated class')
W=Path(__file__).resolve().parent;ROOT=W.parents[1];DATA=ROOT if (ROOT/'1.csv').exists() else ROOT/'mathorcup_A'
base=ROOT/'analysis/A_feasibility/floor_ablation'
raw=pd.read_csv(DATA/'1.csv').set_index('beer'); yy=pd.read_csv(DATA/'4.csv').set_index('beer').loc[raw.index]
X=raw.drop(columns=['beer_id','tasting_category_fine']).to_numpy(float)
Y=yy.drop(columns=['beer_id','tasting_category_fine']).to_numpy(float)
fields=yy.drop(columns=['beer_id','tasting_category_fine']).columns.to_numpy()
styles=raw.tasting_category_fine.to_numpy()
splits=json.loads((base/'split_and_floor_log.json').read_text(encoding='utf-8'))
CANDIDATES=[
 ('ET_original',dict(kind='ET',trees=160,leaf=3,mf=.7)),
 ('ET_leaf1',dict(kind='ET',trees=320,leaf=1,mf=.7)),
 ('ET_mf03',dict(kind='ET',trees=320,leaf=3,mf=.3)),
 ('ET_leaf6',dict(kind='ET',trees=320,leaf=6,mf=.7)),
 ('ET_unit_y',dict(kind='ET',trees=320,leaf=3,mf=.7,yscale=True)),
 ('RF_regularized',dict(kind='RF',trees=320,leaf=3,mf=.7)),
 ('Ridge10',dict(kind='Ridge',alpha=10)),
 ('Ridge100',dict(kind='Ridge',alpha=100)),
 ('PLS8',dict(kind='PLS',components=8)),
 ('SVR_rbf',dict(kind='SVR',C=10,epsilon=.05)),
 ('XGB_depth2',dict(kind='XGB',depth=2,trees=240,lr=.03,reg=10)),
 ('XGB_depth3',dict(kind='XGB',depth=3,trees=180,lr=.03,reg=20)),
]
(W/'q1_candidates.json').write_text(json.dumps(CANDIDATES,indent=2),encoding='utf-8')

def predict(params,a,y,b,seed):
    imp=SimpleImputer(strategy='median');a=imp.fit_transform(a);b=imp.transform(b)
    kind=params['kind'];scaler=None
    if kind in ['Ridge','PLS','SVR']:
        qt=QuantileTransformer(n_quantiles=min(100,len(a)),output_distribution='normal',random_state=seed)
        a=qt.fit_transform(a);b=qt.transform(b)
    if params.get('yscale') or kind=='SVR':
        scaler=StandardScaler();y=scaler.fit_transform(y)
    if kind in ['ET','RF']:
        cls=ExtraTreesRegressor if kind=='ET' else RandomForestRegressor
        fit=cls(n_estimators=params['trees'],min_samples_leaf=params['leaf'],max_features=params['mf'],random_state=seed,n_jobs=2)
    elif kind=='Ridge':fit=Ridge(alpha=params['alpha'])
    elif kind=='PLS':fit=PLSRegression(n_components=params['components'],scale=False,max_iter=1000)
    elif kind=='SVR':fit=MultiOutputRegressor(SVR(C=params['C'],epsilon=params['epsilon'],gamma='scale'),n_jobs=None)
    else:fit=XGBRegressor(n_estimators=params['trees'],max_depth=params['depth'],learning_rate=params['lr'],
        reg_lambda=params['reg'],reg_alpha=.05,min_child_weight=5,colsample_bytree=.7,subsample=.85,
        objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=seed,verbosity=0)
    fit.fit(a,y);p=fit.predict(b)
    return scaler.inverse_transform(p) if scaler else p

N=len(Y);Q=Y.shape[1];K=len(CANDIDATES)
oof={name:np.full((5,N,Q),np.nan) for name in ['ET_original','inner_global','inner_per_target','inner_half_blend']}
all_candidate=np.full((K,5,N,Q),np.nan)
foldrows=[];choices=[];timings=[];start=time.perf_counter()
for outer,sp in enumerate(splits):
    rep,fold=sp['repeat'],sp['fold']
    tr=raw.index.get_indexer(sp['train_beer']);te=raw.index.get_indexer(sp['test_beer'])
    count=pd.Series(styles[tr]).value_counts().min();ninner=min(3,int(count))
    assert ninner>=2
    inner=list(StratifiedKFold(ninner,shuffle=True,random_state=202610260+outer).split(X[tr],styles[tr]))
    inner_pred=np.full((K,len(tr),Q),np.nan)
    candidates=[]
    for k,(name,params) in enumerate(CANDIDATES):
        tic=time.perf_counter()
        for inn,(ia,ib) in enumerate(inner):
            inner_pred[k,ib]=predict(params,X[tr[ia]],Y[tr[ia]],X[tr[ib]],20261025+outer*10+inn)
        pred=predict(params,X[tr],Y[tr],X[te],20261025+outer)
        candidates.append(pred);all_candidate[k,rep,te]=pred
        timings.append(dict(outer=outer,candidate=name,seconds=time.perf_counter()-tic))
    cp=np.stack(candidates)
    score=np.abs(inner_pred-Y[tr][None,:,:]).mean(axis=1)
    best_global=int(score.mean(axis=1).argmin());best=score.argmin(axis=0)
    sel=cp[best,:,np.arange(Q)].T
    oof['ET_original'][rep,te]=cp[0]
    oof['inner_global'][rep,te]=cp[best_global]
    oof['inner_per_target'][rep,te]=sel
    oof['inner_half_blend'][rep,te]=.5*cp[0]+.5*sel
    choices.extend(dict(outer=outer,field=str(f),selected=CANDIDATES[int(k)][0],global_choice=CANDIDATES[best_global][0],n_inner=ninner)
                   for f,k in zip(fields,best))
    row=dict(outer=outer,repeat=rep,fold=fold,n_train=len(tr),n_test=len(te))
    for name,pred in oof.items():row[name]=float(np.abs(Y[te]-pred[rep,te]).mean())
    foldrows.append(row)
    print(f'Outer {outer+1}/15: '+', '.join(f'{n}={row[n]:.4f}' for n in oof)+f'; elapsed {time.perf_counter()-start:.0f}s',flush=True)
    np.savez_compressed(W/'q1_nested_checkpoint.npz',**oof,all_candidate=all_candidate,fields=fields,beer=raw.index.to_numpy())
    pd.DataFrame(foldrows).to_csv(W/'q1_fold_metrics.csv',index=False)
    pd.DataFrame(choices).to_csv(W/'q1_inner_choices.csv',index=False)
    pd.DataFrame(timings).to_csv(W/'q1_runtime.csv',index=False)

saved=np.load(base/'baseline_oof.npz',allow_pickle=True)['prediction']
assert np.max(np.abs(saved-oof['ET_original']))<1e-12
df=pd.DataFrame(foldrows);ratio=(df.n_test/df.n_train).mean();summ={};dims=[]
for name,pred in oof.items():
    assert np.isfinite(pred).all()
    rs=[];rho=[]
    for p in pred:
        rs.append(r2_score(Y,p,multioutput='raw_values'))
        rho.append([float(spearmanr(Y[:,j],p[:,j]).statistic) for j in range(Q)])
    r2=np.mean(rs,axis=0);rh=np.mean(rho,axis=0);mae=np.abs(Y[None,:,:]-pred).mean(axis=(0,1))
    diff=(df.ET_original-df[name])/df.ET_original
    se=float(np.sqrt((1/len(df)+ratio)*diff.var(ddof=1)));mean=float(diff.mean());ci=[mean-float(t.ppf(.975,14))*se,mean+float(t.ppf(.975,14))*se]
    summ[name]=dict(MAE=float(mae.mean()),R2=float(r2.mean()),Spearman=float(rh.mean()),relative_MAE_gain_vs_ET=mean,corrected_approx_95pct_CI=ci,
        meets_predeclared_adoption=bool(mean>=.01 and ci[0]>0),n_positive_R2=int((r2>0).sum()))
    dims.extend(dict(method=name,field=str(f),MAE=float(ma),R2=float(r),Spearman=float(s)) for f,ma,r,s in zip(fields,mae,r2,rh))
pd.DataFrame(dims).to_csv(W/'q1_dimension_metrics.csv',index=False)
summ['metadata']=dict(candidate_count=K,outer_splits=15,inner_splits=sorted({c['n_inner'] for c in choices}),baseline_max_abs_reproduction_error=float(np.max(abs(saved-oof['ET_original']))),
    elapsed_seconds=time.perf_counter()-start,protocol_sha256=hashlib.sha256((W/'实验协议.md').read_bytes()).hexdigest(),
    note='Post-result exploratory study on reused outer folds; inner model selection avoids direct test-label tuning, not an independent confirmation.')
(W/'q1_summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summ,ensure_ascii=False,indent=2),flush=True)
