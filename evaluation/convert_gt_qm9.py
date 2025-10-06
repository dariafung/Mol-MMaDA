#!/usr/bin/env python3
"""
Convert QM9 gdb9.sdf into gt.pkl for 3D conformer evaluation.
Format: [{'symbols': [...], 'confs': [np.array([[x,y,z], ...])]}, ...]
"""

from rdkit import Chem, RDLogger
import numpy as np
import pickle
from pathlib import Path

IN_SDF  = "/projects/bezp/yfeng7/data/QM92014/raw/gdb9.sdf"
OUT_PKL = "/projects/bezp/yfeng7/data/gt.pkl"

def mol_to_entry(mol):
    """Convert RDKit Mol -> dict(symbols, confs)."""
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    return {"symbols": symbols, "confs": [coords]}

def main():
    RDLogger.DisableLog('rdApp.*')

    print(f"Reading SDF from: {IN_SDF}")
    suppl = Chem.SDMolSupplier(IN_SDF, removeHs=False, sanitize=False)
    entries = []
    skipped = 0

    for i, mol in enumerate(suppl):
        if mol is None:
            skipped += 1
            continue
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
            entries.append(mol_to_entry(mol))
        except Exception:
            skipped += 1
            continue

    Path(OUT_PKL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(entries, f)

    print(f"wrote {len(entries)} molecules to {OUT_PKL} (skipped={skipped})")

if __name__ == "__main__":
    main()
