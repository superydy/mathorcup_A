"""Point-conditioned latent Gaussian OT; preserve observed defect coordinates.
AI assistance: OpenAI Codex, 2026-10-26. Missing sources remain uncertifiable.
"""
import json,sys,warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
from common import ROOT,OUT,load,LatentStyles,ot_matrix,select_classifier
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore',message='X does not have valid feature names')
X,Y,style=load();A=X.to_numpy(float);labs=style.to_numpy();fields=list(X);p=len(fields)
fit=LatentStyles().fit(A,labs);B=fit.imp.transform(A);Z=fit.Z
config=json.loads((ROOT/'analysis/q3/defect_thresholds.json').read_text())
defs={k:v['field'] for k,v in config['measured_defects'].items()};D=np.array([fields.index(f) for f in defs.values()]);R=np.array([j for j in range(p) if j not in D])
sm=fit.models['Pils/Lager'];tm=fit.models['Tripel'];T=ot_matrix(sm.covariance_,tm.covariance_)
trip=Z[labs=='Tripel'];ids=np.where(labs=='Pils/Lager')[0]
th=pd.read_excel(ROOT/'8.xlsx',sheet_name='Sensory thresholds')
jp=[fields.index(f) for f in th.loc[th.aging=='increase','field_name_R'] if f in fields]
jm=[fields.index(f) for f in th.loc[th.aging=='decrease','field_name_R'] if f in fields]
enabled=json.loads((OUT/'classifier_summary.json').read_text())['enabled_as_v4_style_criterion']
def aging(a):
    if not np.isfinite(a[jp+jm]).all():return np.nan
    u=fit.G.cdf(a[None,:])[0];return float(.5*(u[jp].mean()+(1-u[jm]).mean()))
def cond(model,c):
    S=model.covariance_;dd=S[np.ix_(D,D)];rd=S[np.ix_(R,D)]
    shift=np.linalg.solve(dd,c-model.location_[D]);mu=model.location_[R]+rd@shift
    cov=S[np.ix_(R,R)]-rd@np.linalg.solve(dd,S[np.ix_(D,R)])
    return mu,(cov+cov.T)/2
rows=[];coords=[];checks=[];source_scores=[]
for ii,i in enumerate(ids):
    beer=int(X.index[i]);source=A[i];z0=Z[i];observed=np.isfinite(source[D]);complete=bool(observed.all());c=z0[D]
    ms,ss=cond(sm,c);mt,st=cond(tm,c);Tc=ot_matrix(ss,st);targetR=mt+Tc@(z0[R]-ms)
    err=float(np.max(abs(Tc@ss@Tc-st))/np.max(abs(st)));assert err<1e-7
    train=np.arange(len(X))!=i
    clf,clf_name,_=select_classifier(X.iloc[train],(labs[train]=='Tripel').astype(int),style.iloc[train],20261026+beer)
    original_score=float(clf.predict_proba(source[None,:])[0,1]);source_scores.append(dict(beer=beer,selected_classifier=clf_name,source_score=original_score))
    d0c=float(np.linalg.norm(z0-tm.location_)/np.sqrt(p));d0n=float(cdist(z0[None,:],trip).min()/np.sqrt(p));a0=aging(source)
    for method in ['ordinary_projected','conditional_fixed']:
        for alpha in [0,.25,.5,.75,1.]:
            za=z0+alpha*(tm.location_+T@(z0-sm.location_)-z0)
            if method=='conditional_fixed':
                za=z0.copy();za[R]=z0[R]+alpha*(targetR-z0[R]);za[D]=c
            cand=fit.G.inverse(za[None,:])[0];before=cand.copy()
            if method=='ordinary_projected':cand[D[observed]]=np.minimum(cand[D[observed]],source[D[observed]])
            else:cand[D[observed]]=source[D[observed]]
            actual_clip=int(np.sum(before[D[observed]]>source[D[observed]]+1e-12)) if method=='ordinary_projected' else 0
            zc=fit.G.transform(cand[None,:])[0];score=float(clf.predict_proba(cand[None,:])[0,1])
            a1=aging(cand);aging_eval=bool(np.isfinite(a0) and np.isfinite(a1));d1c=float(np.linalg.norm(zc-tm.location_)/np.sqrt(p));d1n=float(cdist(zc[None,:],trip).min()/np.sqrt(p))
            floors=source[D[observed]]==B[:,D[observed]].min(0)
            defects=complete and bool((cand[D]<=source[D]+1e-12).all()) and bool((cand[D[observed]][floors]<=source[D[observed]][floors]+1e-12).all())
            aging_ok=aging_eval and a1<=a0+1e-12;nn_ok=1-d1n/d0n>=.10
            old=defects and aging_ok and nn_ok and 1-d1c/d0c>=.20
            new=enabled and defects and aging_ok and nn_ok and score>=.5
            rows.append(dict(beer=beer,method=method,alpha=alpha,complete_evaluation=complete and aging_eval,
                defects_nonincrease=defects,aging_nonincrease=aging_ok,center_reduction=1-d1c/d0c,nn_reduction=1-d1n/d0n,
                classifier_score=score,classified_Tripel=score>=.5,v31_pass=old,v4_pass=new,
                source_change_RMS=float(np.sqrt(np.mean((zc-z0)**2))),actual_projection_triggers=actual_clip,
                observed_defect_count=int(observed.sum()),unchanged_observed_defects=int(np.isclose(cand[D[observed]],source[D[observed]],rtol=0,atol=1e-12).sum()),
                aging_source=a0,aging_candidate=a1,conditional_covariance_error=err))
            coords.append(dict(beer=beer,method=method,alpha=alpha,**dict(zip(fields,cand))))
    print(f'Conditional OT {ii+1}/11: beer {beer}',flush=True)
df=pd.DataFrame(rows);coord=pd.DataFrame(coords);df.to_csv(OUT/'migration_all.csv',index=False);coord.to_csv(OUT/'migration_coordinates.csv',index=False)
pd.DataFrame(source_scores).to_csv(OUT/'migration_classifier_choices.csv',index=False)
criterion='v4_pass' if enabled else 'v31_pass';selected=[]
for (method,beer),g in df[df.alpha>0].groupby(['method','beer']):
    ok=g[g[criterion]];status='incomplete' if not g.complete_evaluation.all() else ('pass' if len(ok) else 'fail')
    row=dict(method=method,beer=int(beer),status=status,alpha=float(ok.alpha.iloc[0]) if status=='pass' else None,
        source_change_RMS=float(ok.source_change_RMS.iloc[0]) if status=='pass' else None)
    selected.append(row)
sel=pd.DataFrame(selected);sel.to_csv(OUT/'migration_selected.csv',index=False)
disp=[]
for (method,alpha),g in coord.groupby(['method','alpha']):
    z=fit.G.transform(g[fields].to_numpy());disp.append(dict(method=method,alpha=float(alpha),mean_pairwise_distance=float(pdist(z).mean()/np.sqrt(p))))
dis=pd.DataFrame(disp);dis.to_csv(OUT/'migration_dispersion.csv',index=False)
counts=sel.groupby(['method','status']).size().to_dict();paired=sel.pivot(index='beer',columns='method',values='source_change_RMS').dropna()
cmp=paired.mean().to_dict();ratio=dis.pivot(index='alpha',columns='method',values='mean_pairwise_distance')
min_ratio=float((ratio.loc[ratio.index>0,'conditional_fixed']/ratio.loc[ratio.index>0,'ordinary_projected']).min())
npas=lambda m:int(((sel.method==m)&(sel.status=='pass')).sum())
use=bool(npas('conditional_fixed')>=npas('ordinary_projected') and len(paired)>0 and cmp['conditional_fixed']<cmp['ordinary_projected'] and min_ratio>=.85)
summary=dict(classifier_enabled=enabled,active_acceptance=criterion,status_counts={f'{a}|{b}':int(v) for (a,b),v in counts.items()},
    paired_complete_success_count=len(paired),paired_mean_change=cmp,minimum_same_alpha_dispersion_ratio=min_ratio,
    selected_migration='conditional_fixed' if use else 'ordinary_projected',
    note='Point conditioning in fitted latent Gaussian; fixed observed native defect values; missing values not certified; no physical or causal guarantee.')
(OUT/'migration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
