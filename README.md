# Mol-MMaDA: Molecular Multimodal Diffusion Language Models

A molecular generation framework based on the MMaDA architecture, adapted for 3D molecular structure generation.

## 🧬 Overview

Mol-MMaDA extends the MMaDA (Multimodal Large Diffusion Language Models) framework to generate 3D molecular structures. It combines:

- **Molecular 3D encoding** for atom types and coordinates
- **Diffusion-based generation** for molecular structures  
- **SELFIES tokenization** for molecular representations
- **Unified training pipeline** for molecular understanding

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Stage 1: Basic molecular understanding
python training/train_mmada.py --config configs/mmada_pretraining_stage1_llada_instruct.yaml

# Stage 2: Enhanced molecular generation
python training/train_mmada.py --config configs/mmada_pretraining_stage2_llada_instruct.yaml
```

### Generation

```bash
# Generate molecules
python generation/generate.py --ckpt /path/to/checkpoint --batch_size 10 --max_atoms 32
```

### Evaluation

```bash
# Basic evaluation (with coordinate scaling fix)
python evaluation/evaluate.py --parquet /path/to/generated_mols.parquet

# Advanced evaluation with multiple scaling factors
python evaluation/evaluate_advanced.py --parquet /path/to/generated_mols.parquet --test-scales
```

## 📁 Project Structure

```
Mol-MMaDA/
├── models/                # Model definitions and architectures
├── training/              # Training scripts and utilities
├── evaluation/            # Evaluation scripts and metrics
├── generation/            # Generation and inference scripts
├── data/                  # Data processing and dataset utilities
├── utils/                 # General utilities
├── configs/               # Configuration files
├── training/train_mmada.py        # Training entry point
├── generation/generate.py         # Generation entry point
└── evaluation/evaluate.py         # Evaluation entry point
```

## 🔧 Key Features

### Coordinate Scaling Fix
The evaluation pipeline includes a critical fix for coordinate scaling issues:
- **Default scale**: 0.1 (automatically applied)
- **Improves results**: 0% → 8.9% molecular stability
- **Better atom stability**: 30% → 97% atom stability

### Evaluation Metrics
- **3D Molecular Stability**: Chemical stability in 3D space
- **Atom Stability**: Correct valency for all atoms
- **Validity**: RDKit molecular validity
- **Uniqueness**: Structural diversity

## 📊 Results

With proper coordinate scaling (scale=0.1):
- **3D mol_stable**: ~8.9%
- **3D atom_stable**: ~97%
- **Validity**: ~100%
- **Unique**: ~18%

## 🛠️ Advanced Usage

### Custom Evaluation
```bash
# Test different scaling factors
python evaluation/evaluate_advanced.py --parquet /path/to/mols.parquet --test-scales

# Analyze coordinate ranges
python evaluation/evaluate_advanced.py --parquet /path/to/mols.parquet --analyze-only
```

### Data Inspection
```bash
# Inspect generated molecules
python utils/inspect_molecules.py

# Analyze duplicates
python utils/analyze_duplicates.py
```

### Dataset Conversion
```bash
# Convert molecular datasets
python data/convert_dataset.py
```

## 📚 Documentation

- **[Code Structure](docs/CODE_STRUCTURE.md)** - Detailed project organization
- **[Evaluation Issues](docs/evaluation_issues.md)** - Technical evaluation details
- **[Evaluation Fix](docs/evaluation_fix.md)** - Coordinate scaling solution
- **[Evaluation Guide](evaluation/eval.md)** - Complete evaluation guide

## 🔬 Technical Details

### Model Architecture
- **Base**: LLaDA-8B-Instruct
- **Molecular Encoder**: 3D coordinate and atom type encoding
- **Diffusion**: Coordinate generation with timestep conditioning
- **Fusion**: Multimodal fusion network

### Training Pipeline
1. **Stage 1**: Basic molecular understanding
2. **Stage 2**: Enhanced generation capabilities
3. **Stage 3**: Instruction following
4. **Stage 4**: Chain-of-thought reasoning

### Data Format
- **Input**: SELFIES strings + 3D coordinates
- **Output**: Generated molecular structures
- **Format**: Parquet files with types/coords columns

## 🐛 Troubleshooting

### Poor Evaluation Results
1. **Check coordinate scaling** - Use `--scale 0.1`
2. **Verify atom types** - Ensure valid atomic numbers (1-118)
3. **Check coordinate ranges** - Use analysis mode

### Import Errors
1. **Add src to path** - Use entry point scripts
2. **Check dependencies** - Install requirements.txt
3. **Verify structure** - Follow code organization

## 🤝 Acknowledgments

This work is heavily based on [Show-o](https://github.com/showlab/Show-o), [LLaDA](https://github.com/ML-GSAI/LLaDA), [MMaDA](https://github.com/Gen-Verse/MMaDA), [maskgit](https://github.com/google-research/maskgit), [transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate) and [webdataset](https://github.com/webdataset/webdataset). Thanks to all the authors for their great work.

## 📄 License

MIT License - see LICENSE file for details.