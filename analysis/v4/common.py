"""Shared v4 primitives. AI assistance: OpenAI Codex, 2026-10-26.
No experiment side effects on import.
"""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
threadpool_limits(2)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/v4';OUT.mkdir(parents=True,exist_ok=True)

def load():
    x=pd.read_csv(ROOT/'1.csv').set_index('beer')
    y=pd.read_csv(ROOT/'4.csv').set_index('beer').loc[x.index].drop(columns=['beer_id','tasting_category_fine'])
    return x.drop(columns=['beer_id','tasting_category_fine']),y,x.tasting_category_fine

def splits():return json.loads((ROOT/'analysis/A_feasibility/floor_ablation/split_and_floor_log.json').read_text(encoding='utf-8'))

src=(ROOT/'src/q3_style_generation/latent_gaussian_ot.py').read_text(encoding='utf-8')
tree=ast.parse(src)
for name in ['EmpiricalMarginals','LatentStyles']:
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),__file__,'exec'),globals())

def classifier():
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PCA(n_components=10,svd_solver='full'),
        LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=20261026))

CLASSIFIER_NAMES=['PCA10_logit','ridge_logit','shrink_LDA','ExtraTrees']
def classifier_candidate(name):
    if name=='PCA10_logit':return classifier()
    if name=='ridge_logit':return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=20261026))
    if name=='shrink_LDA':return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]))
    return make_pipeline(SimpleImputer(strategy='median'),ExtraTreesClassifier(n_estimators=320,min_samples_leaf=3,max_features=.7,class_weight='balanced',random_state=20261026,n_jobs=2))

def select_classifier(X,y,style,seed):
    scores=[]
    for name in CLASSIFIER_NAMES:
        p=np.empty(len(X))
        for tr,te in StratifiedKFold(2,shuffle=True,random_state=seed).split(X,style):
            f=classifier_candidate(name).fit(X.iloc[tr],y[tr]);p[te]=f.predict_proba(X.iloc[te])[:,1]
        scores.append(balanced_accuracy_score(y,p>=.5))
    name=CLASSIFIER_NAMES[int(np.argmax(scores))]
    return classifier_candidate(name).fit(X,y),name,dict(zip(CLASSIFIER_NAMES,map(float,scores)))

def sqrtm(S):
    v,Q=np.linalg.eigh((S+S.T)/2)
    assert v.min()>-1e-8
    return (Q*np.sqrt(np.maximum(v,1e-12))[None,:])@Q.T

def ot_matrix(S,T):
    r=sqrtm(S);ir=np.linalg.inv(r)
    A=ir@sqrtm(r@T@r)@ir
    return (A+A.T)/2
