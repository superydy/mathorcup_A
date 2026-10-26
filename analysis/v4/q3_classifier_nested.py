"""One bounded nested classifier check after fixed candidate failed.
AI assistance: OpenAI Codex, 2026-10-26. No test threshold tuning.
"""
import json,sys,shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,recall_score,confusion_matrix
from common import ROOT,OUT,load,splits,select_classifier
sys.stdout.reconfigure(encoding='utf-8')
for name in ['classifier_summary.json','classifier_repeat_metrics.csv','classifier_oof.csv','classifier_oof.npz']:
    shutil.copy2(OUT/name,OUT/name.replace('classifier_','classifier_fixed_'))
X,Y,style=load();target=(style=='Tripel').to_numpy(int);p=np.full((5,len(X)),np.nan);choices=[]
for outer,sp in enumerate(splits()):
    tr=X.index.get_indexer(sp['train_beer']);te=X.index.get_indexer(sp['test_beer'])
    f,name,scores=select_classifier(X.iloc[tr],target[tr],style.iloc[tr],202610260+outer)
    p[sp['repeat'],te]=f.predict_proba(X.iloc[te])[:,1]
    choices.append(dict(outer=outer,selected=name,**scores));print(f'Classifier nested {outer+1}/15: {name}',flush=True)
rows=[]
for r,v in enumerate(p):
    labs=v>=.5;tn,fp,fn,tp=confusion_matrix(target,labs).ravel()
    rows.append(dict(repeat=r,AUROC=roc_auc_score(target,v),balanced_accuracy=balanced_accuracy_score(target,labs),Tripel_recall=recall_score(target,labs),
        Pils_false_positive_rate=float(labs[style.to_numpy()=='Pils/Lager'].mean()),specificity=tn/(tn+fp),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)))
df=pd.DataFrame(rows);means=df[['AUROC','balanced_accuracy','Tripel_recall','Pils_false_positive_rate','specificity']].mean()
enable=bool(means.AUROC>=.8 and means.balanced_accuracy>=.75 and means.Tripel_recall>=.7 and means.Pils_false_positive_rate<=.2)
df.to_csv(OUT/'classifier_repeat_metrics.csv',index=False);pd.DataFrame(choices).to_csv(OUT/'classifier_inner_choices.csv',index=False)
pd.DataFrame(dict(beer=X.index,style=style.to_numpy(),is_Tripel=target,OOF_mean_score=p.mean(0))).to_csv(OUT/'classifier_oof.csv',index=False)
np.savez_compressed(OUT/'classifier_oof.npz',prediction=p,beer=X.index.to_numpy())
summary=dict(enabled_as_v4_style_criterion=enable,mean_metrics=means.to_dict(),n_Tripel=int(target.sum()),n_Pils=int((style=='Pils/Lager').sum()),
    selection='Four fixed candidates, selected only in each outer training set using 2-fold inner balanced accuracy; threshold .5 fixed.',
    note='Exploratory internal nested validation, reused subjects; no external confirmation; if not enabled no further classifier search.')
(OUT/'classifier_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
