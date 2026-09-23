
from pathlib import Path
import json
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def fetch_and_save(uci_id: int, name: str):
    ds = fetch_ucirepo(id=uci_id)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    X.to_csv(RAW / f"{name}_uci_features.csv", index=False)
    y.to_csv(RAW / f"{name}_uci_targets.csv", index=False)
    meta = {
        "name": name,
        "uci_id": uci_id,
        "metadata": ds.metadata if isinstance(ds.metadata, dict) else str(ds.metadata),
    }
    (RAW / f"{name}_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved {name}: X={X.shape}, y={y.shape}")

if __name__ == "__main__":
    fetch_and_save(319, "MHEALTH")
    fetch_and_save(240, "UCI_HAR")
    print("Raw public datasets downloaded from UCI.")
