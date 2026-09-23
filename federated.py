
from __future__ import annotations
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

def weighted_average_state_dict(states,weights):
    total=float(sum(weights))
    out=OrderedDict()
    for k in states[0]:
        out[k]=sum(s[k].detach().cpu()*float(w) for s,w in zip(states,weights))/total
    return out

def split_modalities_tensor(X, modality_slices, device):
    return {name:torch.tensor(X[:,sl],dtype=torch.float32,device=device)
            for name,sl in modality_slices.items()}

def local_train_classifier(
    model, X, y, observed, modality_slices, local_steps=4,
    lr=1e-3, prox_mu=0.0, global_state=None, device="cpu"
):
    local=deepcopy(model).to(device)
    local.train()
    opt=torch.optim.AdamW(local.parameters(),lr=lr)
    Xt=split_modalities_tensor(X,modality_slices,device)
    yt=torch.tensor(y,dtype=torch.long,device=device)
    ot=torch.tensor(observed,dtype=torch.long,device=device)
    for _ in range(local_steps):
        opt.zero_grad()
        logits,_,_=local(Xt,ot)
        loss=F.cross_entropy(logits,yt)
        if prox_mu>0 and global_state is not None:
            prox=0.0
            for name,p in local.named_parameters():
                prox=prox+torch.sum((p-global_state[name].to(device))**2)
            loss=loss+0.5*prox_mu*prox
        loss.backward(); opt.step()
    return local.cpu().state_dict()

def infer(model,X,observed,modality_slices,device="cpu"):
    model=model.to(device).eval()
    Xt=split_modalities_tensor(X,modality_slices,device)
    ot=torch.tensor(observed,dtype=torch.long,device=device)
    with torch.no_grad():
        logits,h,a=model(Xt,ot)
        prob=torch.softmax(logits,dim=1).cpu().numpy()
    return prob,h.cpu().numpy(),a.cpu().numpy()
