#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

IN_PATH  = "/projects/bezp/yfeng7/data/predicted_conformations.parquet"
OUT_PATH = "/projects/bezp/yfeng7/data/predict.pkl"

# Atomic number -> symbol
ZMAP = {
    1:"H", 6:"C", 7:"N", 8:"O", 9:"F",
    15:"P", 16:"S", 17:"Cl", 35:"Br", 53:"I"
}

def to_symbols(z_array):
    z = np.asarray(z_array).astype(int).tolist()
    return [ZMAP.get(int(v), "X") for v in z]

def to_coords(c_array):
    # c_array is an array/list of length N where each element is a 3-vector
    arr = np.asarray(c_array, dtype=object)
    # If already (N,3) numeric, just cast
    if arr.dtype != object and arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float64, copy=False)
    # Common case in your file: (N,) of 3-vectors -> vstack
    stacked = np.vstack([np.asarray(v, dtype=float).reshape(3,) for v in arr])
    if stacked.ndim != 2 or stacked.shape[1] != 3:
        raise ValueError(f"coords stacked to unexpected shape {stacked.shape}")
    return stacked

def main():
    df = pd.read_parquet(IN_PATH)
    preds = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            symbols = to_symbols(row["types"])
            coords  = to_coords(row["coords"])
            if len(symbols) != coords.shape[0]:
                raise ValueError(f"len(symbols)={len(symbols)} != coords.shape[0]={coords.shape[0]}")
            preds.append({"symbols": symbols, "confs": [coords]})
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"[WARN] skip row {idx}: {e}")
            continue

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(preds, f)
    print(f"wrote {len(preds)} molecules to {OUT_PATH} (skipped={skipped})")

if __name__ == "__main__":
    main()
