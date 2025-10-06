#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone evaluator for 3D conformer prediction (NExT-Mol style).

- Provides `conformer_evaluation_V2` with the same output keys that their eval script prints:
  ['recall_coverage_mean', 'recall_coverage_median',
   'recall_amr_mean', 'recall_amr_median',
   'precision_coverage_mean', 'precision_coverage_median',
   'precision_amr_mean', 'precision_amr_median']

- CLI mimics their eval_confs.py but adds --gt to supply ground-truth conformers directly.
- Threshold defaults follow the paper: QM9=0.5 Å, GEOM-DRUGS=0.75 Å.
"""

import argparse
import pickle
import numpy as np
from typing import Any, Dict, List, Tuple

try:
    from rdkit import Chem
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

Array = np.ndarray
CoordList = List[Array]

# --------------------------- utilities ---------------------------

def _centered(x: Array) -> Array:
    return x - x.mean(axis=0, keepdims=True)

def _kabsch_rmsd(P: Array, Q: Array) -> float:
    """Kabsch-aligned RMSD for two (N,3) arrays."""
    Pc, Qc = _centered(P), _centered(Q)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    P_rot = Pc @ U
    return float(np.sqrt(np.mean(np.sum((P_rot - Qc) ** 2, axis=1))))

def _to_coord_list_from_rdkit_mol(mol) -> CoordList:
    coords: CoordList = []
    if hasattr(mol, "__iter__") and not isinstance(mol, Chem.Mol):
        # List[Mol]
        for m in mol:
            if m is None:
                continue
            confs = m.GetConformers()
            if len(confs) == 0:
                continue
            conf = confs[0]
            N = m.GetNumAtoms()
            arr = np.array([list(conf.GetAtomPosition(i)) for i in range(N)], dtype=float)
            coords.append(arr)
    else:
        # Single Mol
        m = mol
        if m is None:
            return coords
        confs = m.GetConformers()
        if len(confs) == 0:
            return coords
        conf = confs[0]
        N = m.GetNumAtoms()
        arr = np.array([list(conf.GetAtomPosition(i)) for i in range(N)], dtype=float)
        coords.append(arr)
    return coords

def _to_coord_list(item: Any) -> Tuple[CoordList, List[str]]:
    """
    Normalize different input formats to a list of (N,3) numpy arrays.
    Returns: (confs, symbols)
    """
    symbols: List[str] = []
    # Dict format
    if isinstance(item, dict):
        confs = item.get("confs", [])
        symbols = item.get("symbols", [])
        if len(confs) > 0 and not isinstance(confs[0], np.ndarray):
            if _HAVE_RDKIT:
                merged = []
                for c in confs:
                    merged += _to_coord_list_from_rdkit_mol(c)
                confs = merged
            else:
                raise ValueError("Found RDKit Mol in dict['confs'] but RDKit is not installed.")
        return confs, symbols
    # List of arrays
    if isinstance(item, list) and (len(item) == 0 or isinstance(item[0], np.ndarray)):
        return item, symbols
    # RDKit Mol / List[Mol]
    if _HAVE_RDKIT:
        if isinstance(item, Chem.Mol) or (isinstance(item, list) and (len(item) == 0 or isinstance(item[0], Chem.Mol))):
            return _to_coord_list_from_rdkit_mol(item), symbols
    # Fallback
    return [], symbols

def _pairwise_min_rmsd(pred_confs: CoordList, gt_confs: CoordList) -> Tuple[Array, Array]:
    """
    Returns:
      - min_pred_to_gt[i]: RMSD from i-th predicted conformer to its closest GT conformer
      - min_gt_to_pred[j]: RMSD from j-th GT conformer to its closest predicted conformer
    """
    P, K = len(pred_confs), len(gt_confs)
    if P == 0 or K == 0:
        return np.full(P, np.inf), np.full(K, np.inf)
    dmat = np.zeros((P, K), dtype=float)
    for i, p in enumerate(pred_confs):
        for j, g in enumerate(gt_confs):
            if p.shape[0] != g.shape[0]:
                raise ValueError("Atom count mismatch between a pred and a GT conformer.")
            dmat[i, j] = _kabsch_rmsd(p, g)
    return dmat.min(axis=1), dmat.min(axis=0)

def _cov_amr(min_dists: Array, threshold: float) -> Tuple[float, float]:
    """Coverage (fraction < threshold) and AMR (mean of distances)."""
    finite = np.isfinite(min_dists)
    if not finite.any():
        return 0.0, float("inf")
    d = min_dists[finite]
    cov = float(np.mean((d < threshold).astype(float)))
    amr = float(np.mean(d))
    return cov, amr

# ---------------------- input normalization helpers ----------------------

def _maybe_unwrap_tuple(obj):
    """Unwrap only if root is a tuple and obj[0] is list/tuple (classic (pred_list, aux))."""
    if isinstance(obj, tuple) and len(obj) >= 1 and isinstance(obj[0], (list, tuple)):
        return list(obj[0])
    return obj

def _normalize_pack(obj):
    """Coerce various pickle roots into a list aligned by molecule."""
    obj = _maybe_unwrap_tuple(obj)

    if isinstance(obj, (list, tuple)):
        return list(obj)

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return list(obj.tolist())
        raise TypeError(f"Unsupported numpy dtype at root: {obj.dtype}")

    if isinstance(obj, dict):
        # single molecule dict
        if 'symbols' in obj and 'confs' in obj:
            return [obj]
        # mapping id->entry
        try:
            keys = sorted(obj.keys(), key=lambda x: int(x))
        except Exception:
            keys = list(obj.keys())
        return [obj[k] for k in keys]

    try:
        import pandas as pd
        if 'DataFrame' in str(type(obj)):
            if 'symbols' in obj.columns and 'confs' in obj.columns:
                return obj[['symbols', 'confs']].to_dict('records')
    except Exception:
        pass

    raise TypeError(f"Unsupported pickle root type: {type(obj)}")

# ---------------------- core evaluator (matches keys) ----------------------

def conformer_evaluation_V2(
    predict_pack: Any,
    gt_pack: Any,
    threshold: float,
    num_failures: int = 0,
    logger: Any = None,
    num_process: int = 1,
    dataset_name: str = ""
) -> Dict[str, float]:
    """
    Standalone version mirroring NExT-Mol's evaluation outputs.
    Required outputs (keys):
      recall_coverage_mean/median, recall_amr_mean/median,
      precision_coverage_mean/median, precision_amr_mean/median
    """
    predict_pack = _normalize_pack(predict_pack)
    gt_pack      = _normalize_pack(gt_pack)

    # align lengths to the minimum to avoid assertion when one side is shorter
    n = min(len(predict_pack), len(gt_pack))
    if len(predict_pack) != len(gt_pack):
        print(f"[info] length mismatch: predict={len(predict_pack)}, gt={len(gt_pack)}; "
              f"evaluating on the first {n} pairs.")
    predict_pack = predict_pack[:n]
    gt_pack      = gt_pack[:n]

    rec_cov_list, rec_amr_list, pre_cov_list, pre_amr_list = [], [], [], []
    skipped = 0

    for idx, (pred_item, gt_item) in enumerate(zip(predict_pack, gt_pack)):
        pred_confs, _ = _to_coord_list(pred_item)
        gt_confs, _ = _to_coord_list(gt_item)

        if len(pred_confs) == 0 or len(gt_confs) == 0:
            skipped += 1
            continue

        min_pred_to_gt, min_gt_to_pred = _pairwise_min_rmsd(pred_confs, gt_confs)

        # Precision direction: pred -> gt
        cov_p, amr_p = _cov_amr(min_pred_to_gt, threshold)
        # Recall direction: gt -> pred
        cov_r, amr_r = _cov_amr(min_gt_to_pred, threshold)

        pre_cov_list.append(cov_p); pre_amr_list.append(amr_p)
        rec_cov_list.append(cov_r); rec_amr_list.append(amr_r)

    def _mean_and_median(xs: List[float]) -> Tuple[float, float]:
        if len(xs) == 0:
            return 0.0, 0.0
        arr = np.array(xs, dtype=float)
        return float(arr.mean()), float(np.median(arr))

    rec_cov_mean, rec_cov_median = _mean_and_median(rec_cov_list)
    rec_amr_mean, rec_amr_median = _mean_and_median(rec_amr_list)
    pre_cov_mean, pre_cov_median = _mean_and_median(pre_cov_list)
    pre_amr_mean, pre_amr_median = _mean_and_median(pre_amr_list)

    metrics = {
        "recall_coverage_mean":   rec_cov_mean,
        "recall_coverage_median": rec_cov_median,
        "recall_amr_mean":        rec_amr_mean,
        "recall_amr_median":      rec_amr_median,
        "precision_coverage_mean":   pre_cov_mean,
        "precision_coverage_median": pre_cov_median,
        "precision_amr_mean":        pre_amr_mean,
        "precision_amr_median":      pre_amr_median,
    }
    if skipped > 0:
        metrics["_skipped"] = skipped
    return metrics

# --------------------------- CLI (close to theirs) ---------------------------

def main(args):
    # dataset threshold defaults
    if args.threshold is None:
        if args.dataset == 'QM9-df':
            threshold = 0.5
        elif args.dataset == 'Geom-drugs-df':
            threshold = 0.75
        else:
            threshold = 0.75
    else:
        threshold = float(args.threshold)

    with open(args.input, 'rb') as f:
        predict_pack = pickle.load(f)
    with open(args.gt, 'rb') as f:
        gt_pack = pickle.load(f)

    metrics = conformer_evaluation_V2(
        predict_pack,
        gt_pack,
        threshold=threshold,
        num_failures=0,
        logger=None,
        num_process=args.num_process,
        dataset_name=args.dataset
    )

    print("\n--------------------------")
    for metric in [
        'recall_coverage_mean', 'recall_coverage_median',
        'recall_amr_mean', 'recall_amr_median',
        'precision_coverage_mean', 'precision_coverage_median',
        'precision_amr_mean', 'precision_amr_median'
    ]:
        print(metric, metrics[metric])
    if "_skipped" in metrics:
        print("_skipped", metrics["_skipped"])
    print('--------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to predict.pkl')
    parser.add_argument('--gt', type=str, required=True, help='path to ground-truth conformers pickle')
    parser.add_argument('--dataset', type=str, default='QM9-df', choices=['QM9-df', 'Geom-drugs-df'])
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=None, help='override default threshold')
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)
