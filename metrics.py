
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score, log_loss

def ece(prob,y,n_bins=12):
    conf=prob.max(axis=1); pred=prob.argmax(axis=1)
    edges=np.linspace(0,1,n_bins+1)
    val=0.0
    for b in range(n_bins):
        lo,hi=edges[b],edges[b+1]
        m=(conf>=lo)&((conf<hi) if b<n_bins-1 else (conf<=hi))
        if m.any():
            val+=m.mean()*abs((pred[m]==y[m]).mean()-conf[m].mean())
    return float(val)

def multiclass_brier(prob,y):
    Y=np.eye(prob.shape[1])[y]
    return float(np.mean(np.sum((prob-Y)**2,axis=1)))

def aurc(prob,y):
    conf=prob.max(axis=1)
    err=(prob.argmax(axis=1)!=y).astype(float)
    order=np.argsort(-conf)
    err=err[order]
    cum=np.cumsum(err)/(np.arange(len(err))+1)
    coverage=(np.arange(len(err))+1)/len(err)
    return float(np.trapz(cum,coverage))

def selective_error(prob,y,threshold):
    conf=prob.max(axis=1)
    pred=prob.argmax(axis=1)
    m=conf>=threshold
    return float(np.mean(pred[m]!=y[m])) if m.any() else np.nan, float(m.mean())

def classification_metrics(prob,y,threshold=None,n_bins=12):
    pred=prob.argmax(axis=1)
    out={
        "macro_f1":float(f1_score(y,pred,average="macro")),
        "mcc":float(matthews_corrcoef(y,pred)),
        "cohen_kappa":float(cohen_kappa_score(y,pred)),
        "ece":ece(prob,y,n_bins),
        "brier":multiclass_brier(prob,y),
        "nll":float(log_loss(y,prob,labels=np.arange(prob.shape[1]))),
        "aurc":aurc(prob,y),
    }
    try:
        Y=np.eye(prob.shape[1])[y]
        out["macro_auroc"]=float(roc_auc_score(Y,prob,average="macro",multi_class="ovr"))
    except Exception:
        out["macro_auroc"]=float("nan")
    if threshold is not None:
        se,cov=selective_error(prob,y,threshold)
        out["selective_error"]=se
        out["coverage"]=cov
        out["abstention"]=1-cov
    return out

def nrmse_on_missing(recovered,truth,feature_missing_mask):
    e=((recovered-truth)**2)[feature_missing_mask]
    if e.size==0: return 0.0
    rmse=np.sqrt(e.mean())
    denom=np.sqrt((truth[feature_missing_mask]**2).mean())+1e-12
    return float(rmse/denom)
