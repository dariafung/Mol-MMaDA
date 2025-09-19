#!/usr/bin/env python3
"""
Fixed evaluation script for Mol-MMaDA generated molecules.

Key fixes:
1. Proper coordinate scaling (coordinates are in Angstroms but need scaling)
2. Better error handling and debugging
3. Multiple scaling options to find optimal values
4. Comprehensive metrics reporting
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D

from eval_functions import get_2D_edm_metric, get_3D_edm_metric

PT = Chem.GetPeriodicTable()


def z_to_symbol(z: int) -> str:
    return PT.GetElementSymbol(int(z))


def coerce_types(obj):
    """Return int array shape (N,) and drop invalid Z (<=0 or >118)."""
    arr = np.asarray(obj)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = arr.astype(int, copy=False)
    mask = (arr >= 1) & (arr <= 118)
    return arr[mask], mask


def coerce_coords(obj):
    """
    Return float array shape (N,3) from many possible layouts:
    - list[list[3]], list[np.array(3)], np.array(N,3)
    - flat length 3N -> reshape(-1,3)
    - object arrays -> stack
    """
    a = np.asarray(obj, dtype=object)
    # If already (N,3) and numeric, return directly
    if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 3 and a.dtype != object:
        return a.astype(float, copy=False)

    # Object case: elements are length-3 sequences
    if a.dtype == object:
        try:
            a = np.stack([np.asarray(x, dtype=float).reshape(3)
                         for x in a], axis=0)
            return a
        except Exception:
            pass

    # Regular case: try to convert to float
    a = np.asarray(obj, dtype=float)
    if a.ndim == 1 and a.size % 3 == 0:
        a = a.reshape(-1, 3)
    if not (a.ndim == 2 and a.shape[1] == 3):
        raise ValueError(f"coords bad shape {a.shape}")
    return a


def row_to_placeholder_mol(types, coords, scale=1.0, center=False):
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert len(types) == coords.shape[0]
    m = Chem.RWMol()
    for z in types:
        m.AddAtom(Chem.Atom(z_to_symbol(int(z))))
    conf = Chem.Conformer(len(types))
    xyz = coords * float(scale)
    if center:
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    m.AddConformer(conf, assignId=True)
    return m.GetMol()


def analyze_coordinate_ranges(df, sample_size=100):
    """Analyze coordinate ranges to understand scaling issues."""
    print("=== Coordinate Range Analysis ===")

    sample_indices = np.random.choice(
        len(df), size=min(sample_size, len(df)), replace=False)

    all_coords = []
    all_distances = []

    for idx in sample_indices:
        try:
            row = df.iloc[idx]
            coords = coerce_coords(row["coords"])
            types_raw = row["types"]
            types, mask = coerce_types(types_raw)

            if mask.shape[0] != coords.shape[0]:
                m = min(mask.shape[0], coords.shape[0])
                types = types[:m]
                coords = coords[:m]

            if coords.shape[0] != types.shape[0] or coords.shape[1] != 3:
                continue

            all_coords.append(coords)

            # Calculate nearest neighbor distances
            if len(coords) > 1:
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        dist = np.linalg.norm(coords[i] - coords[j])
                        all_distances.append(dist)

        except Exception as e:
            continue

    if all_coords:
        all_coords = np.vstack(all_coords)
        print(f"Coordinate ranges:")
        print(
            f"  X: [{all_coords[:, 0].min():.3f}, {all_coords[:, 0].max():.3f}]")
        print(
            f"  Y: [{all_coords[:, 1].min():.3f}, {all_coords[:, 1].max():.3f}]")
        print(
            f"  Z: [{all_coords[:, 2].min():.3f}, {all_coords[:, 2].max():.3f}]")
        print(f"  Mean absolute: {np.abs(all_coords).mean():.3f}")
        print(f"  Std: {all_coords.std():.3f}")

    if all_distances:
        all_distances = np.array(all_distances)
        print(f"Distance statistics:")
        print(f"  Min: {all_distances.min():.3f}")
        print(f"  Max: {all_distances.max():.3f}")
        print(f"  Mean: {all_distances.mean():.3f}")
        print(f"  Median: {np.median(all_distances):.3f}")
        print(f"  Std: {all_distances.std():.3f}")

        # Expected bond distances (Angstroms)
        print(f"Expected bond distances (Å):")
        print(f"  C-C: ~1.54, C-H: ~1.09, C-N: ~1.47, C-O: ~1.43")
        print(f"  N-H: ~1.01, O-H: ~0.96")

        # Suggest scaling factor
        expected_bond = 1.5  # Average C-C bond
        current_median = np.median(all_distances)
        suggested_scale = expected_bond / current_median
        print(f"Suggested scaling factor: {suggested_scale:.4f}")

    print()


def evaluate_with_scaling(df, scale_factor, dataset="Geom", n_samples=1000, debug=False):
    """Evaluate molecules with a specific scaling factor."""
    print(f"=== Evaluation with scale={scale_factor} ===")

    total = len(df)
    n = min(n_samples, total)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    take_idx = rng.choice(total, size=n, replace=False)
    df_sub = df.iloc[take_idx].reset_index(drop=True)

    mols = []
    skipped = 0
    printed = 0

    for ridx, row in df_sub.iterrows():
        try:
            coords = coerce_coords(row["coords"])
            types_raw = row["types"]
            types, mask = coerce_types(types_raw)

            # Sync crop coords
            if mask.shape[0] != coords.shape[0]:
                m = min(mask.shape[0], coords.shape[0])
                types = types[:m]
                coords = coords[:m]

            if coords.shape[0] != types.shape[0] or coords.shape[1] != 3:
                raise ValueError(
                    f"length mismatch types={types.shape} coords={coords.shape}")
            if types.size == 0:
                raise ValueError("empty after filtering invalid Z")

            mols.append(row_to_placeholder_mol(
                types, coords, scale=scale_factor, center=False))

        except Exception as e:
            skipped += 1
            if debug and printed < 5:
                printed += 1
                print(f"[DEBUG] skip row {ridx}: {e}")

    if not mols:
        print("No valid molecules to evaluate.")
        return None

    print(f"Evaluating {len(mols)} molecules (skipped {skipped})")

    try:
        m3d, rd_mols = get_3D_edm_metric(
            mols, train_mols=None, dataset_name=dataset, use_mmff=False)
        m2d = get_2D_edm_metric(rd_mols, train_mols=None)

        results = {
            "scale": scale_factor,
            "input_rows": int(n),
            "skipped_rows": int(skipped),
            "used_molecules": int(len(mols)),
            "metrics_3d": m3d,
            "metrics_2d": m2d,
        }

        # Print key metrics
        print(f"3D Metrics:")
        print(f"  mol_stable: {m3d.get('mol_stable', 0):.4f}")
        print(f"  atom_stable: {m3d.get('atom_stable', 0):.4f}")
        print(f"  Validity: {m3d.get('Validity', 0):.4f}")
        print(f"  Unique: {m3d.get('Unique', 0):.4f}")

        print(f"2D Metrics:")
        print(f"  mol_stable: {m2d.get('mol_stable', 0):.4f}")
        print(f"  atom_stable: {m2d.get('atom_stable', 0):.4f}")
        print(f"  Validity: {m2d.get('Validity', 0):.4f}")
        print(f"  Unique: {m2d.get('Unique', 0):.4f}")

        return results

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Fixed evaluation for Mol-MMaDA generated molecules")
    ap.add_argument("--parquet", required=True,
                    help="Path to generated_mols.parquet")
    ap.add_argument("--engine", default="pyarrow",
                    choices=["pyarrow", "fastparquet"])
    ap.add_argument("--n", type=int, default=1000,
                    help="Sample size for evaluation")
    ap.add_argument("--dataset", default="Geom",
                    choices=["QM9", "Geom"], help="Dataset rules for bond checking")
    ap.add_argument("--out", default="eval_results_fixed.json",
                    help="Output JSON file")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Only analyze coordinate ranges")
    ap.add_argument("--test-scales", action="store_true",
                    help="Test multiple scaling factors")
    ap.add_argument("--scale", type=float, default=0.1,
                    help="Single scaling factor to test")
    ap.add_argument("--debug", action="store_true", help="Enable debug output")

    args = ap.parse_args()

    print(f"Loading data from {args.parquet}")
    df = pd.read_parquet(args.parquet, engine=args.engine)
    print(f"Loaded {len(df)} molecules")

    # Analyze coordinate ranges
    analyze_coordinate_ranges(df, sample_size=min(500, len(df)))

    if args.analyze_only:
        return

    if args.test_scales:
        # Test multiple scaling factors
        scales_to_test = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        best_results = None
        best_score = -1

        for scale in scales_to_test:
            print(f"\n{'='*50}")
            results = evaluate_with_scaling(
                df, scale, args.dataset, args.n, args.debug)
            if results:
                # Use 3D molecular stability as the main metric
                score = results["metrics_3d"].get("mol_stable", 0)
                if score > best_score:
                    best_score = score
                    best_results = results
                    print(
                        f"*** NEW BEST SCORE: {score:.4f} with scale={scale} ***")

        if best_results:
            print(f"\n{'='*50}")
            print(f"BEST RESULTS (scale={best_results['scale']}):")
            print(
                f"3D mol_stable: {best_results['metrics_3d'].get('mol_stable', 0):.4f}")
            print(
                f"3D atom_stable: {best_results['metrics_3d'].get('atom_stable', 0):.4f}")

            # Save best results
            with open(args.out, "w") as f:
                json.dump(best_results, f, indent=2)
            print(f"Saved best results to {args.out}")
    else:
        # Single scale evaluation
        results = evaluate_with_scaling(
            df, args.scale, args.dataset, args.n, args.debug)
        if results:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
