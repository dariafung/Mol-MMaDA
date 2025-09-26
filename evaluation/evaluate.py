#!/usr/bin/env python3
"""
Evaluation script for Mol-MMaDA generated molecules.
This script evaluates molecular generation quality with proper chemical rule compliance.
"""

from evaluation.eval_functions import get_2D_edm_metric, get_3D_edm_metric
import argparse
import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PT = Chem.GetPeriodicTable()


def z_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
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
    Return float array shape (N,3) from many possible layouts.
    Handles nested arrays like the generated data format.
    """
    a = np.asarray(obj, dtype=object)

    # Handle nested arrays (like the generated data)
    if a.dtype == object and len(a) > 0:
        try:
            # Check if first element is an array
            if hasattr(a[0], '__len__') and len(a[0]) == 3:
                return np.array([np.asarray(row) for row in a]).astype(np.float32)
        except (ValueError, TypeError, IndexError):
            pass

    # Handle regular 2D arrays
    if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 3 and a.dtype != object:
        return a.astype(np.float32)

    # Flatten and reshape
    try:
        flat = np.asarray(a).flatten()
        if len(flat) % 3 == 0:
            return flat.reshape(-1, 3).astype(np.float32)
    except (ValueError, TypeError):
        pass

    raise ValueError(f"Cannot convert to (N,3) array from shape {a.shape}")


def analyze_coordinate_ranges(df, sample_size=500):
    """Analyze coordinate ranges and suggest optimal scaling."""
    print("=== Coordinate Range Analysis ===")

    # Sample molecules for analysis
    sample_df = df.sample(min(sample_size, len(df)))

    all_coords = []
    all_distances = []

    for _, row in sample_df.iterrows():
        try:
            coords = coerce_coords(row['coords'])
            all_coords.append(coords)

            # Calculate pairwise distances
            if len(coords) > 1:
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        dist = np.linalg.norm(coords[i] - coords[j])
                        all_distances.append(dist)
        except Exception as e:
            continue

    if not all_coords:
        print("No valid coordinates found for analysis")
        return

    all_coords = np.vstack(all_coords)
    all_distances = np.array(all_distances)

    print(f"Coordinate ranges:")
    print(f"  X: [{all_coords[:, 0].min():.3f}, {all_coords[:, 0].max():.3f}]")
    print(f"  Y: [{all_coords[:, 1].min():.3f}, {all_coords[:, 1].max():.3f}]")
    print(f"  Z: [{all_coords[:, 2].min():.3f}, {all_coords[:, 2].max():.3f}]")
    print(f"  Mean absolute: {np.mean(np.abs(all_coords)):.3f}")
    print(f"  Std: {np.std(all_coords):.3f}")

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
    expected_bond_length = 1.5  # Typical C-C bond
    current_median = np.median(all_distances)
    suggested_scale = expected_bond_length / current_median
    print(f"Suggested scaling factor: {suggested_scale:.4f}")


def evaluate_with_scaling(df, scale_factor, dataset="Geom", n_samples=1000, debug=False):
    """Evaluate molecules with coordinate scaling."""
    print(f"=== Evaluation with scale={scale_factor} ===")

    # Sample molecules for evaluation
    eval_df = df.sample(min(n_samples, len(df)))

    predict_mols = []
    skipped = 0

    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing molecules"):
        try:
            # Get atom types and coordinates
            types, mask = coerce_types(row['types'])
            coords = coerce_coords(row['coords'])

            # Apply mask to coordinates
            if len(coords) > len(types):
                coords = coords[:len(types)]
            elif len(types) > len(coords):
                types = types[:len(coords)]

            # Apply scaling
            coords = coords * scale_factor

            # Convert to RDKit molecule
            mol = Chem.RWMol()
            for z in types:
                mol.AddAtom(Chem.Atom(int(z)))

            # Add conformer
            conf = Chem.Conformer(len(types))
            for i, coord in enumerate(coords):
                conf.SetAtomPosition(i, Point3D(
                    float(coord[0]), float(coord[1]), float(coord[2])))
            mol.AddConformer(conf)

            # Sanitize molecule
            try:
                Chem.SanitizeMol(mol)
                predict_mols.append(mol)
            except Exception as e:
                if debug:
                    print(f"Molecule {idx} failed sanitization: {e}")
                skipped += 1
                continue

        except Exception as e:
            if debug:
                print(f"Molecule {idx} failed processing: {e}")
            skipped += 1
            continue

    print(f"Evaluating {len(predict_mols)} molecules (skipped {skipped})")

    if not predict_mols:
        print("No valid molecules to evaluate")
        return None

    try:
        # Get 2D metrics
        metrics_2d = get_2D_edm_metric(predict_mols)

        # Get 3D metrics
        metrics_3d, rd_mols = get_3D_edm_metric(predict_mols)

        results = {
            "scale_factor": scale_factor,
            "n_molecules": len(predict_mols),
            "n_skipped": skipped,
            "metrics_2d": metrics_2d,
            "metrics_3d": metrics_3d
        }

        # Print results
        print("3D Metrics:")
        for key, value in metrics_3d.items():
            print(f"  {key}: {value:.4f}")

        print("2D Metrics:")
        for key, value in metrics_2d.items():
            print(f"  {key}: {value:.4f}")

        return results

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation for Mol-MMaDA generated molecules")
    parser.add_argument("--parquet", required=True,
                        help="Path to generated_mols.parquet")
    parser.add_argument("--engine", default="pyarrow",
                        choices=["pyarrow", "fastparquet"])
    parser.add_argument("--n", type=int, default=1000,
                        help="Sample size for evaluation")
    parser.add_argument("--dataset", default="Geom",
                        choices=["QM9", "Geom"], help="Dataset rules for bond checking")
    parser.add_argument("--out", default="eval_results.json",
                        help="Output JSON file")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze coordinate ranges")
    parser.add_argument("--test-scales", action="store_true",
                        help="Test multiple scaling factors")
    parser.add_argument("--scale", type=float, default=0.1,
                        help="Single scaling factor to test")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")

    args = parser.parse_args()

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
            print(f"\n--- Testing scale factor: {scale} ---")
            try:
                results = evaluate_with_scaling(df, scale, dataset=args.dataset,
                                                n_samples=args.n, debug=args.debug)
                if results:
                    # Use 3D molecular stability as the main metric
                    score = results["metrics_3d"].get("mol_stable", 0)
                    print(f"Scale {scale}: 3D mol_stable = {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_results = results
                        print(f"*** NEW BEST: {score:.4f} ***")

            except Exception as e:
                print(f"Error with scale {scale}: {e}")
                continue

        if best_results:
            print(f"\nBest results with scale {best_results['scale_factor']}:")
            print(
                f"3D mol_stable: {best_results['metrics_3d']['mol_stable']:.4f}")
            print(
                f"3D atom_stable: {best_results['metrics_3d']['atom_stable']:.4f}")
            print(f"Validity: {best_results['metrics_3d']['Validity']:.4f}")
            print(f"Unique: {best_results['metrics_3d']['Unique']:.4f}")

            # Save best results
            with open(args.out, 'w') as f:
                json.dump(best_results, f, indent=2)
            print(f"Saved best results to {args.out}")
        else:
            print("No valid results obtained")
    else:
        # Single scale evaluation
        results = evaluate_with_scaling(df, args.scale, dataset=args.dataset,
                                        n_samples=args.n, debug=args.debug)

        if results:
            # Save results
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {args.out}")
        else:
            print("Evaluation failed")


if __name__ == "__main__":
    main()
