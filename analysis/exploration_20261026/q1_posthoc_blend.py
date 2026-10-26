"""Explicitly post-hoc fixed half blend after observing L1/MSE tradeoff.
AI assistance: OpenAI Codex, 2026-10-26. Not an independent confirmation.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import t,spearmanr
from sklearn.metrics import r2_score
sys.stdout.reconfigure(encoding='utf-8')
W=Path(__file__).resolve().parent;ROOT=W.parents[1];data=ROOT if (ROOT/'1.csv').exists() else ROOT/'mathorcup_A'
x=pd.read_csv(data/'1.csv').set_index('beer');y=pd.read_csv(data/'4.csv').set_index('beer').loc[x.index].drop(columns=['beer_id','tasting_category_fine']);Y=y.to_numpy(float)
baseline=np.load(ROOT/'analysis/A_feasibility/floor_ablation/baseline_oof.npz',allow_pickle=True)['prediction']
p=np.load(W/'q1_independent_predictions.npz',allow_pickle=True)['LGBM_L1']
blend=.5*(baseline+p)
splits=json.loads((ROOT/'analysis/A_feasibility/floor_ablation/split_and_floor_log.json').read_text(encoding='utf-8'))
rows=[]
for sp in splits:
    r=sp['repeat'];te=y.index.get_indexer(sp['test_beer']);a=float(abs(Y[te]-baseline[r,te]).mean());b=float(abs(Y[te]-blend[r,te]).mean())
    rows.append(dict(repeat=r,fold=sp['fold'],baseline_MAE=a,blend_MAE=b,relative_gain=(a-b)/a,n_train=len(sp['train_beer']),n_test=len(te)))
df=pd.DataFrame(rows);diff=df.relative_gain;mu=float(diff.mean());se=float(np.sqrt((1/15+(df.n_test/df.n_train).mean())*diff.var(ddof=1)))
mae=abs(Y[None,:,:]-blend).mean(axis=(0,1));r2=np.mean([r2_score(Y,b,multioutput='raw_values') for b in blend],axis=0);rho=np.mean([[spearmanr(Y[:,j],b[:,j]).statistic for j in range(Y.shape[1])] for b in blend],axis=0)
summary=dict(MAE=float(mae.mean()),R2=float(r2.mean()),Spearman=float(rho.mean()),relative_MAE_gain_vs_ET=mu,
    corrected_approx_95pct_CI=[mu-t.ppf(.975,14)*se,mu+t.ppf(.975,14)*se],
    label='Post-hoc fixed 0.5 blend after observing opposing L1/MSE results. No weight scan. Reused folds; CI does not correct method-selection bias.')
(W/'q1_posthoc_blend_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
df.to_csv(W/'q1_posthoc_blend_folds.csv',index=False)
pd.DataFrame(dict(field=y.columns,MAE=mae,R2=r2,Spearman=rho)).to_csv(W/'q1_posthoc_blend_dimensions.csv',index=False)
np.savez_compressed(W/'q1_posthoc_blend_predictions.npz',prediction=blend,fields=y.columns.to_numpy(),beer=y.index.to_numpy())
print(json.dumps(summary,indent=2))
