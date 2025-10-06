#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from rdkit import Chem, RDLogger
from selfies import encoder as smiles_to_selfies

RDLogger.DisableLog('rdApp.*')  # suppress RDKit console warnings


def _find_smiles_col(df: pd.DataFrame):
    for c in df.columns:
        if c.lower() in ("smiles", "smile", "smiles_string", "smiles_str"):
            return c
    return None


def _mol_to_canonical_smiles_from_sdf_mol(mol: Chem.Mol) -> str | None:
    """
    Sanitize an SDF mol and produce a canonical SMILES safely.
    Returns None on failure.
    """
    if mol is None:
        return None
    # Work on a copy to avoid side-effects
    m2 = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(
            m2,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            | Chem.SanitizeFlags.SANITIZE_KEKULIZE
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_ADJUSTHS
            | Chem.SanitizeFlags.SANITIZE_CLEANUP,
        )
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(m2), canonical=True)
    except Exception:
        return None


def convert_qm9_to_selfies_txt(
    csv_path: str,
    sdf_path: str,
    output_txt_path: str,
    num_rows_to_process: int | None = None,
) -> None:
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    smiles_col = _find_smiles_col(df)

    # Limit rows if requested (applies to both CSV and SDF-backed paths)
    if num_rows_to_process is not None:
        df = df.head(num_rows_to_process)
        print(f"Processing only first {len(df)} rows")

    selfies_list: list[str] = []
    kept = 0
    skipped = 0

    if smiles_col is not None:
        print(f"Using SMILES column from CSV: '{smiles_col}'")
        series = df[smiles_col].astype(str)

        for i, s in enumerate(series, start=1):
            try:
                m = Chem.MolFromSmiles(s)
                if m is None:
                    skipped += 1
                    continue
                selfies = smiles_to_selfies(s)
                if not selfies:
                    skipped += 1
                    continue
                selfies_list.append(selfies)
                kept += 1
            except Exception:
                skipped += 1

            if i % 2000 == 0:
                print(f"Processed {i} | kept {kept} | skipped {skipped}")

    else:
        print("No SMILES column found in CSV. Falling back to SDF -> SMILES.")
        # Load SDF in order (do not sanitize here; we sanitize per-molecule)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        sdf_mols = [m for m in suppl]

        n_csv = len(df)
        n_sdf = len(sdf_mols)
        if n_sdf < n_csv:
            print(f"Warning: SDF has fewer molecules ({n_sdf}) than CSV rows ({n_csv}); truncating to {n_sdf}.")
            df = df.iloc[:n_sdf].copy()
        n = min(len(df), len(sdf_mols))

        for i in range(n):
            mol = sdf_mols[i]
            # require at least one conformer (QM9 SDF should have it)
            if mol is None or mol.GetNumConformers() == 0:
                skipped += 1
                continue

            # robust SDF->SMILES
            smiles = _mol_to_canonical_smiles_from_sdf_mol(mol)
            if not smiles:
                skipped += 1
                continue

            try:
                selfies = smiles_to_selfies(smiles)
                if not selfies:
                    skipped += 1
                    continue
                selfies_list.append(selfies)
                kept += 1
            except Exception:
                skipped += 1

            if (i + 1) % 2000 == 0:
                print(f"Processed {i+1} | kept {kept} | skipped {skipped}")

    # Write out
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for s in selfies_list:
            f.write(s + "\n")

    print(f"Finished: kept={kept}, skipped={skipped}")
    print(f"Written SELFIES to: {output_txt_path}")


if __name__ == "__main__":
    # Hardcoded defaults; adjust as needed
    CSV = "/projects/bezp/yfeng7/data/QM92014/raw/gdb9.sdf.csv"
    SDF = "/projects/bezp/yfeng7/data/QM92014/raw/gdb9.sdf"
    OUT = "/projects/bezp/yfeng7/data/molecules.txt"

    # Example: set num_rows_to_process=None for full dataset, or a small number for a quick test
    convert_qm9_to_selfies_txt(CSV, SDF, OUT)
