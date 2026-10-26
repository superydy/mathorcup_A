"""Fixed real-data OOF discriminator validation. AI assistance: OpenAI Codex, 2026-10-26."""
import json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,recall_score,confusion_matrix
from common import ROOT,OUT,load,splits,classifier
sys.stdout.reconfigure(encoding='utf-8')
X,Y,style=load();target=(style=='Tripel').to_numpy(int)
pred=np.full((5,len(X)),np.nan);seen=np.zeros((5,len(X)),int)
for sp in splits():
    tr=X.index.get_indexer(sp['train_beer']);te=X.index.get_indexer(sp['test_beer']);r=sp['repeat']
    f=classifier().fit(X.iloc[tr],target[tr]);pred[r,te]=f.predict_proba(X.iloc[te])[:,1];seen[r,te]+=1
assert (seen==1).all() and np.isfinite(pred).all()
rows=[]
for r,p in enumerate(pred):
    labels=p>=.5;tn,fp,fn,tp=confusion_matrix(target,labels).ravel()
    rows.append(dict(repeat=r,AUROC=roc_auc_score(target,p),balanced_accuracy=balanced_accuracy_score(target,labels),
        Tripel_recall=recall_score(target,labels),Pils_false_positive_rate=float(labels[style.to_numpy()=='Pils/Lager'].mean()),
        specificity=tn/(tn+fp),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)))
df=pd.DataFrame(rows);means=df[['AUROC','balanced_accuracy','Tripel_recall','Pils_false_positive_rate','specificity']].mean()
enable=bool(means.AUROC>=.8 and means.balanced_accuracy>=.75 and means.Tripel_recall>=.7 and means.Pils_false_positive_rate<=.2)
df.to_csv(OUT/'classifier_repeat_metrics.csv',index=False)
pd.DataFrame({'beer':X.index,'style':style.to_numpy(),'is_Tripel':target,'OOF_mean_score':pred.mean(0)}).to_csv(OUT/'classifier_oof.csv',index=False)
np.savez_compressed(OUT/'classifier_oof.npz',prediction=pred,beer=X.index.to_numpy())
result=dict(activation_rule='AUROC>=.80, BA>=.75, recall>=.70, Pils FPR<=.20; repeat means; threshold .5 fixed',
    enabled_as_v4_style_criterion=enable,mean_metrics=means.to_dict(),n_Tripel=int(target.sum()),n_Pils=int((style=='Pils/Lager').sum()),
    note='Internal repeated OOF validation of fixed classifier. Scores are uncalibrated. No external confirmation.')
(OUT/'classifier_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
