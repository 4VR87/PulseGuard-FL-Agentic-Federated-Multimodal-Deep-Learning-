
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def jain_index(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if np.all(counts == 0):
        return 0.0
    return float(counts.sum()**2 / (len(counts) * np.square(counts).sum()))

def macro_f1_from_cm(cm: np.ndarray) -> float:
    vals=[]
    for c in range(cm.shape[0]):
        tp=cm[c,c]
        fp=cm[:,c].sum()-tp
        fn=cm[c,:].sum()-tp
        p=tp/(tp+fp+1e-12)
        r=tp/(tp+fn+1e-12)
        vals.append(2*p*r/(p+r+1e-12))
    return float(np.mean(vals))

def normalize_minmax(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    x=np.asarray(x,dtype=float)
    return (x-x.min())/(x.max()-x.min()+eps)

def max_silence(selection_history: List[List[int]], n_clients: int) -> int:
    last=[-1]*n_clients
    longest=[0]*n_clients
    for t,selected in enumerate(selection_history):
        s=set(selected)
        for k in range(n_clients):
            if k in s:
                gap=t-last[k]-1
                longest[k]=max(longest[k],gap)
                last[k]=t
    T=len(selection_history)
    for k in range(n_clients):
        longest[k]=max(longest[k], T-last[k]-1)
    return int(max(longest))
