import pandas as pd
import json
from selfies import encoder
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import Optional, Tuple, List


def atom_to_id(symbol: str) -> int:
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    except Exception:
        return 0


def smiles_to_selfies(smiles_string: Optional[str]) -> Optional[str]:
    try:
        if smiles_string is None or pd.isna(smiles_string):
            return None
        return encoder(str(smiles_string))
    except Exception:
        return None


def get_3d_from_mol(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    conf = mol.GetConformer(0)
    num_atoms = mol.GetNumAtoms()

    atom_vec = np.array([atom_to_id(a.GetSymbol()) for a in mol.GetAtoms()], dtype=np.int64)
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                       for i in range(num_atoms)], dtype=np.float32)

    bond_type_mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    edge_type = np.zeros((num_atoms, num_atoms), dtype=np.int64)
    bond_type = np.zeros((num_atoms, num_atoms), dtype=np.int64)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edge_type[i, j] = edge_type[j, i] = 1
        bond_type[i, j] = bond_type[j, i] = bond_type_mapping.get(b.GetBondType(), 0)

    dist_matrix = np.array(Chem.Get3DDistanceMatrix(mol, confId=conf.GetId()), dtype=np.float32)
    return atom_vec, coords, edge_type, bond_type, dist_matrix


def find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_sdf_in_order(sdf_path: str) -> List[Chem.Mol]:
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    return [m for m in suppl]  # keep order


def convert_qm9_to_parquet(
    csv_path: str,
    sdf_path: str,
    output_parquet_path: str,
    num_rows_to_process: Optional[int] = None,
) -> None:
    print(f"Loading QM9 CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if num_rows_to_process is not None:
        df = df.head(num_rows_to_process)
        print(f"Processing only first {len(df)} rows")

    # property columns (case-insensitive)
    wanted = ["mu", "alpha", "homo", "lumo", "gap", "cv"]  # note: cv is lowercase in your file
    prop_cols = []
    for name in wanted:
        col = find_col(df, [name, name.lower(), name.upper()])
        if col is not None:
            prop_cols.append(col)
    if not prop_cols:
        print("Warning: no property columns found among", wanted)

    col_mol_id = find_col(df, ["mol_id"])

    print(f"Reading SDF in order: {sdf_path}")
    sdf_mols = load_sdf_in_order(sdf_path)

    n_csv = len(df)
    n_sdf = len(sdf_mols)
    if n_sdf < n_csv:
        print(f"Warning: SDF has fewer molecules ({n_sdf}) than CSV rows ({n_csv}); truncating to {n_sdf}.")
        df = df.iloc[:n_sdf].copy()
        n = n_sdf
    else:
        n = n_csv

    records = []
    skipped = 0
    for i in range(n):
        mol = sdf_mols[i]
        if mol is None or mol.GetNumConformers() == 0:
            skipped += 1
            continue

        # try to get a canonical smiles
        try:
            m2 = Chem.Mol(mol)
            Chem.SanitizeMol(m2, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        except Exception:
            m2 = Chem.Mol(mol)
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(m2), canonical=True)
        except Exception:
            try:
                smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
            except Exception:
                skipped += 1
                continue

        selfies = smiles_to_selfies(smiles)
        if selfies is None:
            skipped += 1
            continue

        atom_vec, coords, edge_type, bond_type, dist_matrix = get_3d_from_mol(mol)

        row = df.iloc[i]
        rec = {
            "id": int(i),
            "selfies_string": selfies,
            "text_description": "",
            "atom_vec_str": json.dumps(atom_vec.tolist()),
            "coordinates_str": json.dumps(coords.tolist()),
            "edge_type_str": json.dumps(edge_type.tolist()),
            "bond_type_str": json.dumps(bond_type.tolist()),
            "dist_str": json.dumps(dist_matrix.tolist()),
            "rdmol2selfies_str": json.dumps([]),
        }
        if col_mol_id is not None:
            rec["mol_id"] = str(row[col_mol_id])

        for p in prop_cols:
            rec[p] = float(row[p])

        records.append(rec)

        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1} | kept {len(records)} | skipped {skipped}")

    out_df = pd.DataFrame(records)
    print(f"Finished: kept {len(out_df)} | skipped {skipped}")
    print(f"Writing parquet: {output_parquet_path}")
    out_df.to_parquet(output_parquet_path, index=False)
    print("Done.")


if __name__ == "__main__":
    CSV = "/projects/bezp/yfeng7/data/QM92014/raw/gdb9.sdf.csv"
    SDF = "/projects/bezp/yfeng7/data/QM92014/raw/gdb9.sdf"
    OUT = "/projects/bezp/yfeng7/data/qm9_molecular_data.parquet"
    convert_qm9_to_parquet(CSV, SDF, OUT, num_rows_to_process=None)
