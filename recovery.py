
from __future__ import annotations
import itertools
import numpy as np
import torch
from scipy.optimize import minimize_scalar

def class_modality_prototypes(X, y, modality_slices, n_classes):
    prot={}
    for c in range(n_classes):
        prot[c]={}
        m=y==c
        for name,sl in modality_slices.items():
            prot[c][name]=X[m,sl].mean(axis=0) if m.any() else np.zeros(sl.stop-sl.start)
    return prot

def prototype_fill(X_masked, observed, pred_class, modality_slices, prototypes):
    out=X_masked.copy()
    names=list(modality_slices)
    for i in range(len(out)):
        c=int(pred_class[i])
        for j,name in enumerate(names):
            if observed[i,j]==0:
                out[i,modality_slices[name]]=prototypes[c][name]
    return out

def simplex_grid(step=0.25, n=3):
    units=int(round(1/step))
    out=[]
    for a in range(units+1):
        for b in range(units+1-a):
            c=units-a-b
            out.append(np.array([a,b,c],dtype=float)/units)
    return out

def nll_from_prob(prob,y,eps=1e-12):
    return -np.log(np.clip(prob[np.arange(len(y)),y],eps,1.0)).mean()

def fit_pattern_router(candidate_probs, y, patterns, step=0.25, min_examples=8):
    """
    candidate_probs shape: [N, 3, C] for direct/prototype/deep candidates.
    Returns pattern-specific convex weights and a global fallback.
    """
    grids=simplex_grid(step,3)
    def best(ids):
        best_w=None; best_loss=float("inf")
        for w in grids:
            p=(candidate_probs[ids]*w[None,:,None]).sum(axis=1)
            loss=nll_from_prob(p,y[ids])
            if loss<best_loss:
                best_loss,best_w=loss,w.copy()
        return best_w
    all_ids=np.arange(len(y))
    global_w=best(all_ids)
    mapping={}
    keys=[tuple(row.tolist()) for row in patterns]
    for k in sorted(set(keys)):
        ids=np.array([i for i,z in enumerate(keys) if z==k],dtype=int)
        mapping[k]=best(ids) if len(ids)>=min_examples else global_w
    return mapping,global_w

def apply_router(candidate_probs,patterns,mapping,global_w):
    out=[]
    for i,pat in enumerate(patterns):
        w=mapping.get(tuple(pat.tolist()),global_w)
        out.append((candidate_probs[i]*w[:,None]).sum(axis=0))
    return np.vstack(out)

def fit_temperature_from_prob(prob,y):
    logits=np.log(np.clip(prob,1e-12,1.0))
    def objective(logT):
        T=np.exp(logT)
        z=logits/T
        z=z-z.max(axis=1,keepdims=True)
        p=np.exp(z); p/=p.sum(axis=1,keepdims=True)
        return nll_from_prob(p,y)
    res=minimize_scalar(objective,bounds=(-4,4),method="bounded")
    return float(np.exp(res.x))

def apply_temperature(prob,T):
    z=np.log(np.clip(prob,1e-12,1.0))/T
    z=z-z.max(axis=1,keepdims=True)
    p=np.exp(z); p/=p.sum(axis=1,keepdims=True)
    return p

def acceptance_threshold(prob, target_coverage=0.80):
    conf=prob.max(axis=1)
    # 20th percentile -> approx. 80% acceptance, matching the manuscript.
    return float(np.quantile(conf,1-target_coverage))

def selective_predict(prob,threshold):
    conf=prob.max(axis=1)
    pred=prob.argmax(axis=1)
    accept=conf>=threshold
    return pred,accept,conf
