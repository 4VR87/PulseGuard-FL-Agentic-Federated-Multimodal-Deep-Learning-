
"""
PulseGuard-FL experiment scaffold reconstructed from the supplied manuscript.

IMPORTANT:
The paper provides the experimental protocol and many hyperparameters, but it
does not include the authors' original source code, processed feature artifacts,
exact feature-engineering procedure, exact client-score weights, client budget,
classifier hidden sizes, learning rates, or every reliability-generation rule.
Therefore this script is a faithful framework, not proof of bit-exact reproduction.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from pulseguard.utils import seed_everything, jain_index
from pulseguard.data import subject_disjoint_folds

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"config/paper_config.json").read_text())

def print_protocol():
    print("PulseGuard-FL paper protocol")
    print("Seeds:",CFG["seeds"])
    print("Folds:",CFG["n_subject_folds"])
    print("Drop probabilities:",CFG["drop_probabilities"])
    print("Classifier global rounds:",CFG["global_rounds_classifier"])
    print("Classifier local steps:",CFG["local_steps"])
    print("Denoiser rounds:",CFG["global_rounds_denoiser"])
    print("Methods:",", ".join(CFG["methods"]))

if __name__=="__main__":
    print_protocol()
    print("\nThis package intentionally stops short of manufacturing the paper's")
    print("reported numerical results because the supplied manuscript does not")
    print("contain the original processed feature tables or all training hyperparameters.")
    print("Use scripts/download_public_datasets.py to fetch the two public UCI datasets.")
