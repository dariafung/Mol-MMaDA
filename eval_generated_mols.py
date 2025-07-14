#!/usr/bin/env python
"""Evaluate the quality of the 3 D molecules you just generated.

Usage (inside your mmada_env):
  python eval_generated_mols.py /path/to/generated_3d_molecules_for_evaluation.parquet

Optional flags:
  --coord_col   name of the coordinates column   (default: generated_coords)
  --type_col    name of the atom‑types column    (default: generated_types)
  --dataset     dataset rules for bond checking  (QM9 | Geom | …)
  --mmff        run an MMFF minimisation before evaluation

The script converts every row into an RDKit Mol, then calls
`evaluation.eval_functions.get_3D_edm_metric` to obtain the EDM
(3‑D stability + RDKit validity/uniqueness) scores, printing them as JSON.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import pyarrow.parquet as pq           # <‑‑ NEW: skip pandas ragged‑list bug
from rdkit import Chem
from rdkit.Geometry import Point3D

try:
    from evaluation.eval_functions import get_3D_edm_metric
except ImportError as e:
    sys.exit(f"❌  无法导入 evaluation.eval_functions —— 确认 PYTHONPATH 设置正确\n{e}")

_PERIODIC = Chem.GetPeriodicTable()

def _type_to_symbol(t: int) -> str:
    """Convert integer atomic number → element symbol (‘C’, ‘O’, …)."""
    return _PERIODIC.GetElementSymbol(int(t))


def _row_to_mol(types, coords):
    """Make an RDKit Mol from a single dataframe row.

    * skips padding atoms where `typ==0` **or** `coords is None`.
    * coordinates are centred exactly as stored (no normalisation here).
    """
    mol = Chem.RWMol()
    idx_map: dict[int, int] = {}

    for i, (typ, xyz) in enumerate(zip(types, coords)):
        if typ == 0 or xyz is None:
            continue  # padding
        idx_map[i] = mol.AddAtom(Chem.Atom(_type_to_symbol(typ)))

    conf = Chem.Conformer(mol.GetNumAtoms())
    for old_i, new_i in idx_map.items():
        x, y, z = map(float, coords[old_i])
        conf.SetAtomPosition(new_i, Point3D(x, y, z))
    mol.AddConformer(conf)

    return mol.GetMol()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("Evaluate generated molecules (3D‑EDM)")
    parser.add_argument("parquet", help="Parquet file produced by generate.py")
    parser.add_argument("--coord_col", default="generated_coords",
                        help="column holding coordinates list [default: %(default)s]")
    parser.add_argument("--type_col",  default="generated_types",
                        help="column holding atom‑type list [default: %(default)s]")
    parser.add_argument("--dataset",   default="QM9",
                        help="QM9 / Geom – decides bond rules [default: %(default)s]")
    parser.add_argument("--mmff", action="store_true",
                        help="MMFF‑optimise each molecule before metrics")
    args = parser.parse_args(argv)

    path = Path(args.parquet).expanduser().resolve()
    if not path.exists():
        sys.exit(f"❌  找不到文件: {path}")

    print(f"📂 Reading {path} …")
    try:
        table = pq.read_table(path)
        rows  = table.to_pylist()      # list[dict]
    except Exception as e:
        sys.exit(f"❌  Parquet 读取失败: {e}")

    print("   rows:", len(rows))

    rd_mols = [_row_to_mol(r[args.type_col], r[args.coord_col]) for r in rows]

    print("⚙️  Running get_3D_edm_metric … this may take a while")
    scores, _ = get_3D_edm_metric(
        rd_mols,
        train_mols=None,
        dataset_name=args.dataset,
        use_mmff=args.mmff,
    )

    print("\n🎯 3D‑EDM result:")
    print(json.dumps(scores, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
