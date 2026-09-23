
from __future__ import annotations
import numpy as np

def predictive_entropy(prob: np.ndarray, eps: float=1e-12) -> np.ndarray:
    p=np.clip(prob,eps,1.0)
    h=-(p*np.log(p)).sum(axis=1)
    return h/np.log(p.shape[1])

def normalized_novelty(client_repr: np.ndarray, global_repr: np.ndarray, eps:float=1e-12) -> np.ndarray:
    d=np.linalg.norm(client_repr-global_repr[None,:],axis=1)
    return (d-d.min())/(d.max()-d.min()+eps)

def agent_score(u,v,q,r,weights):
    return (weights["uncertainty"]*u
            +weights["novelty"]*v
            +weights["staleness"]*q
            +weights["reliability"]*r)

def quota_select(scores:np.ndarray, cumulative_counts:np.ndarray, budget:int):
    """
    Lexicographic count balancing:
      1) lower cumulative selection count first
      2) score breaks ties
    This enforces the manuscript's participation-count bound.
    """
    idx=list(range(len(scores)))
    idx.sort(key=lambda k:(cumulative_counts[k],-scores[k]))
    return idx[:budget]
