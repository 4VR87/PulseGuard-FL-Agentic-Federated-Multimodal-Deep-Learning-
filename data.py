
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class SubjectDataset:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    modality_slices: Dict[str, slice]

def subject_disjoint_folds(subject_ids: np.ndarray, n_folds: int=5, seed: int=17):
    """
    Five deterministic subject folds.
    For fold f:
      test = fold f
      calibration = fold (f+1) mod 5
      training = remaining 3 folds
    This matches the role logic stated in the manuscript.
    """
    subjects=np.array(sorted(np.unique(subject_ids)))
    rng=np.random.default_rng(seed)
    rng.shuffle(subjects)
    folds=np.array_split(subjects,n_folds)
    out=[]
    for f in range(n_folds):
        test=np.asarray(folds[f])
        cal=np.asarray(folds[(f+1)%n_folds])
        train=np.concatenate([np.asarray(folds[j]) for j in range(n_folds)
                              if j not in (f,(f+1)%n_folds)])
        out.append((train,cal,test))
    return out

def fit_train_scaler(X: np.ndarray, subject: np.ndarray, train_subjects: np.ndarray):
    m=np.isin(subject,train_subjects)
    scaler=StandardScaler().fit(X[m])
    return scaler

def apply_modality_missingness(
    X: np.ndarray,
    modality_slices: Dict[str,slice],
    p_drop: float,
    rng: np.random.Generator,
):
    """
    Independent Bernoulli modality drop per example; if all modalities drop,
    one modality is uniformly retained, exactly as described in the manuscript.
    """
    X=np.asarray(X)
    out=X.copy()
    names=list(modality_slices)
    observed=np.ones((len(X),len(names)),dtype=np.int64)
    for i in range(len(X)):
        keep = rng.random(len(names)) >= p_drop
        if not keep.any():
            keep[rng.integers(0,len(names))]=True
        observed[i]=keep.astype(np.int64)
        for j,name in enumerate(names):
            if not keep[j]:
                out[i,modality_slices[name]]=0.0
    return out,observed

def infer_subject_ids_from_uci_har_original_split(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    subject_train_path: str, subject_test_path: str
):
    """
    Helper when working from the original UCI HAR archive.
    The paper does NOT use the original 70/30 train/test partition; it rejoins
    subjects and performs a five-fold subject-disjoint evaluation.
    """
    s_tr=np.loadtxt(subject_train_path,dtype=int)
    s_te=np.loadtxt(subject_test_path,dtype=int)
    X=np.vstack([X_train.values,X_test.values])
    subject=np.concatenate([s_tr,s_te])
    return X,subject
