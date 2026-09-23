
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim:int, hidden:int=64, latent:int=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden),nn.ReLU(),
            nn.Linear(hidden,latent),nn.ReLU(),
        )
    def forward(self,x): return self.net(x)

class ModalityGatedClassifier(nn.Module):
    """
    Two-layer modality encoders + availability-masked soft attention + classifier.
    This follows the textual description of Eqs. (4)-(6).
    """
    def __init__(self, modality_dims:Dict[str,int], n_classes:int, hidden:int=64, latent:int=32):
        super().__init__()
        self.names=list(modality_dims)
        self.encoders=nn.ModuleDict({k:ModalityEncoder(v,hidden,latent) for k,v in modality_dims.items()})
        self.gate=nn.Sequential(nn.Linear(latent,16),nn.Tanh(),nn.Linear(16,1))
        self.classifier=nn.Linear(latent,n_classes)

    def forward(self, x_by_modality:Dict[str,torch.Tensor], observed:torch.Tensor):
        zs=[]
        logits_gate=[]
        for j,k in enumerate(self.names):
            z=self.encoders[k](x_by_modality[k])
            z=z*observed[:,j:j+1].float()
            zs.append(z)
            logits_gate.append(self.gate(z))
        G=torch.cat(logits_gate,dim=1)
        G=G.masked_fill(observed==0,-1e9)
        a=torch.softmax(G,dim=1)
        Z=torch.stack(zs,dim=1)
        h=(a.unsqueeze(-1)*Z).sum(dim=1)
        return self.classifier(h),h,a

class DenoisingRecovery(nn.Module):
    def __init__(self, input_dim:int, n_modalities:int, hidden:int=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim+n_modalities,hidden),nn.ReLU(),
            nn.Linear(hidden,hidden),nn.ReLU(),
            nn.Linear(hidden,input_dim),
        )
    def forward(self,x_masked,observed):
        return self.net(torch.cat([x_masked,observed.float()],dim=1))

def hidden_coordinate_mse(reconstructed, target, feature_missing_mask):
    denom=feature_missing_mask.float().sum().clamp_min(1.0)
    return (((reconstructed-target)**2)*feature_missing_mask.float()).sum()/denom
