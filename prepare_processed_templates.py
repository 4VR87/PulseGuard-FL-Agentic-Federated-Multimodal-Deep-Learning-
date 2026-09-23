
"""
Feature-preprocessing helper.

The supplied manuscript states:
  MHEALTH: 2,670 non-overlapping windows, 276 standardized statistical features
           = 60 chest/ECG + 108 ankle + 108 wrist.
  UCI-HAR: 10,299 windows, 72 engineered features
           = 36 accelerometer + 36 gyroscope.

The exact feature list/windowing/statistics used to create those two processed
artifacts are NOT specified in the supplied paper. This module therefore
documents the required output schema without pretending to reconstruct an
unknown feature pipeline.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMAS={
 "MHEALTH":{
   "n_rows_expected":2670,
   "modalities":{"chest_ecg":60,"ankle":108,"wrist":108},
   "n_features":276,
 },
 "UCI_HAR":{
   "n_rows_expected":10299,
   "modalities":{"accelerometer":36,"gyroscope":36},
   "n_features":72,
 }
}

def validate_processed_artifact(path, dataset):
    df=pd.read_csv(path)
    spec=SCHEMAS[dataset]
    feature_cols=[c for c in df.columns if c.startswith("f_")]
    assert len(feature_cols)==spec["n_features"], (len(feature_cols),spec["n_features"])
    assert "subject" in df and "label" in df
    return df

def create_empty_template(dataset, path):
    spec=SCHEMAS[dataset]
    cols=["subject","label"]+[f"f_{i:03d}" for i in range(spec["n_features"])]
    pd.DataFrame(columns=cols).to_csv(path,index=False)

if __name__=="__main__":
    out=Path(__file__).resolve().parents[1]/"data/processed"
    out.mkdir(exist_ok=True,parents=True)
    for name in SCHEMAS:
        create_empty_template(name,out/f"{name}_processed_template.csv")
        print("Wrote template",name)
