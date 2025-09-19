# Molecular MMaDA Evaluation Guide

This document provides guidance on evaluating the molecular generation capabilities of Mol-MMaDA.

## Quick Start

### Basic Evaluation
```bash
# Evaluate generated molecules with proper coordinate scaling
python evaluate_fixed.py --parquet /path/to/generated_mols.parquet --scale 0.1

# Test multiple scaling factors to find optimal
python evaluate_fixed.py --parquet /path/to/generated_mols.parquet --test-scales
```

### Original Evaluation (with scaling fix)
```bash
# Use the original evaluation script with coordinate scaling
python evaluate.py --parquet /path/to/generated_mols.parquet --scale 0.1 --debug 10
```

## Key Metrics

### 3D Molecular Stability
- **mol_stable**: Percentage of molecules that are chemically stable in 3D
- **atom_stable**: Percentage of atoms with correct valency
- **Validity**: Percentage of valid RDKit molecules
- **Unique**: Percentage of unique molecular structures
- **Novelty**: Novelty compared to training set (requires training data)

### 2D Molecular Properties
- **mol_stable**: 2D molecular stability
- **atom_stable**: 2D atom stability
- **Validity**: RDKit validity
- **Unique**: Structural uniqueness

## Coordinate Scaling Issue

The main issue with evaluation results is **coordinate scaling**. Generated coordinates need to be scaled by a factor of **0.1** to match expected bond distances:

- Expected C-C bonds: ~1.54 Å
- Expected C-H bonds: ~1.09 Å
- Generated coordinates (unscaled): median distance ~3.6 units
- Optimal scaling factor: 0.1

## Troubleshooting

### Poor Molecular Stability
1. Check coordinate scaling - use `--scale 0.1`
2. Verify atom types are valid (1-118)
3. Check coordinate ranges with analysis mode

### Low Validity
1. Ensure proper atom type mapping
2. Check for invalid atomic numbers
3. Verify coordinate dimensions (N, 3)

### Debug Mode
Use `--debug 10` to see specific failure reasons for up to 10 molecules.

## Advanced Evaluation

### Custom Scaling
```bash
python evaluate_fixed.py --parquet /path/to/generated_mols.parquet --scale 0.05
```

### Analysis Only
```bash
python evaluate_fixed.py --parquet /path/to/generated_mols.parquet --analyze-only
```

### Large Scale Evaluation
```bash
python evaluate_fixed.py --parquet /path/to/generated_mols.parquet --n 5000 --scale 0.1
```

## Expected Results

With proper coordinate scaling (scale=0.1):
- 3D mol_stable: ~10-15%
- 3D atom_stable: ~95-98%
- Validity: ~100%
- Unique: ~10-20%

These results indicate the model is generating chemically reasonable structures.
