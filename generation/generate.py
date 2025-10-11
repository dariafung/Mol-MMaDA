#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 3D molecular conformations from 1D molecular inputs (SELFIES) using diffusion sampling.

Key changes vs. earlier version:
- FIX OOM: do NOT pass the whole input_selfies list into the model at once.
  We slice the input into per-batch chunks and only pass the current chunk.

- Everything else (output format, cleaning, parquet writing) remains the same.
"""

import argparse
import glob
import inspect
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
import yaml

# ensure local 'models' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:
    torch = None  # allow running on login node without torch if you use --dummy

# Optional deps (used only in non-dummy mode)
try:
    from transformers import AutoTokenizer
    from selfies import get_semantic_robust_alphabet
    import selfies as sf
    from rdkit import Chem
    from models.modeling_mmada import MMadaModelLM, MMadaConfig
except Exception:
    AutoTokenizer = None
    get_semantic_robust_alphabet = None
    sf = None
    Chem = None
    MMadaModelLM = None
    MMadaConfig = None

# Default compact idx (0..5) -> atomic number Z  (0=pad/mask)
DEFAULT_IDX2Z = np.array([0, 1, 6, 7, 8, 9], dtype=np.int64)
ALLOWED_QM9 = {1, 6, 7, 8, 9}  # H, C, N, O, F

_STATE = {"model": None, "tokenizer": None, "device": "cpu", "idx2z": DEFAULT_IDX2Z}


def set_global_seed(seed: Optional[int] = None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None and torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_atom_count_from_selfies(selfies_str: str) -> int:
    """Rough atom-count estimate from SELFIES (used for logging/cap)."""
    if sf is None or Chem is None:
        return min(len(selfies_str) + 1, 32)
    try:
        smiles = sf.decoder(selfies_str)
        if not smiles:
            return min(len(selfies_str) + 1, 32)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return min(len(selfies_str) + 1, 32)
        return mol.GetNumAtoms()
    except Exception:
        return min(len(selfies_str) + 1, 32)


def _load_gen_cfg(gen_cfg_path: Optional[str]) -> Dict:
    defaults = {
        "seed": None,
        "t_steps": None,
        "guidance_scale": 1.0,
        "type_conf_frac": 0.15,
        "type_topk": 3,
        "temperature": 1.0,
        "scheduler_name": None,
        "idx2z": None,
        "commit_class0": True,
    }
    if not gen_cfg_path:
        return {"generation": defaults}
    with open(gen_cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    gen = cfg.get("generation", {})
    out = defaults.copy()
    out.update({k: gen.get(k, v) for k, v in defaults.items()})
    return {"generation": out}


def postprocess_atoms(
    atom_types: np.ndarray,
    coords: np.ndarray,
    atoms_mask: Optional[np.ndarray] = None,
    dataset: str = "QM9",
    origin_eps: float = 1e-8,
    dedup_eps: float = 1e-6,
    cap: int = 0,
    drop_nonfinite: bool = True,
    drop_type0: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Clean one molecule and return (types, coords) or (None, None)."""
    atom_types = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != atom_types.shape[0]:
        return None, None

    N = atom_types.shape[0]
    keep = np.ones(N, dtype=bool)

    if atoms_mask is not None:
        atoms_mask = np.asarray(atoms_mask, dtype=bool).reshape(-1)
        if atoms_mask.shape[0] == N:
            keep &= atoms_mask

    if drop_type0:
        keep &= (atom_types > 0)

    if origin_eps is not None and origin_eps > 0:
        keep &= (np.linalg.norm(coords, axis=1) > origin_eps)

    if drop_nonfinite:
        keep &= np.isfinite(coords).all(axis=1)

    if dataset.upper() == "QM9":
        keep &= np.isin(atom_types, list(ALLOWED_QM9))

    atom_types = atom_types[keep]
    coords = coords[keep]
    if atom_types.size == 0:
        return None, None

    if dedup_eps is not None and dedup_eps > 0:
        keys = np.round(coords / dedup_eps).astype(np.int64)
        keys_view = np.ascontiguousarray(keys).view(
            np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
        ).ravel()
        _, idx = np.unique(keys_view, return_index=True)
        idx = np.sort(idx)
        atom_types = atom_types[idx]
        coords = coords[idx]

    if cap and cap > 0 and atom_types.shape[0] > cap:
        atom_types = atom_types[:cap]
        coords = coords[:cap]

    return atom_types, coords


def _load_model_and_tokenizer(ckpt_dir: str, device: str, cfg_path: Optional[str] = None, gen_idx2z: Optional[list] = None):
    """Lazy-load model/tokenizer once."""
    if _STATE["model"] is not None:
        return

    if torch is None:
        raise RuntimeError("PyTorch is required for real sampling. Use --dummy for a smoke test.")
    if cfg_path is None:
        raise ValueError("Please provide --config (the training YAML) to rebuild model/tokenizer.")

    # Read training YAML
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Tokenizer (match training)
    llm_name = cfg["model"]["llm_model_name_or_path"]
    if AutoTokenizer is None:
        raise RuntimeError("transformers not available. Please install transformers.")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if get_semantic_robust_alphabet is not None:
        add_tokens = list(set(get_semantic_robust_alphabet()) - set(tokenizer.get_vocab().keys()))
        if add_tokens:
            tokenizer.add_tokens(add_tokens)

    # Build config (mirror training)
    m = cfg["model"]
    vocab_size = len(tokenizer)
    embedding_size = ((vocab_size + 127) // 128) * 128

    if MMadaConfig is None:
        raise RuntimeError("MMadaConfig is not available. Ensure your project imports are correct.")

    model_cfg = MMadaConfig(
        llm_config_path=m["llm_config_path"],
        llm_model_name_or_path=m["llm_model_name_or_path"],
        mol_atom_embedding_dim=m["mol_atom_embedding_dim"],
        mol_coord_embedding_dim=m["mol_coord_embedding_dim"],
        mol_3d_encoder_output_dim=m["mol_3d_encoder_output_dim"],
        num_atom_types=m["num_atom_types"],
        max_atoms=m["max_atoms"],
        max_selfies_length=m["max_selfies_length"],
        output_atom_coords_dim=m["output_atom_coords_dim"],
        output_atom_type_dim=m["output_atom_type_dim"],
        d_model=m["d_model"],
        fusion_hidden_dim=m["fusion_hidden_dim"],
        final_condition_dim=m["final_condition_dim"],
        diffusion_timesteps=m["diffusion_timesteps"],
        noise_schedule_beta_start=m["noise_schedule_beta_start"],
        noise_schedule_beta_end=m["noise_schedule_beta_end"],
        noise_schedule_name=m["noise_schedule_name"],
        lm_coeff=m.get("lm_coeff", 1.0),
        diff_coeff=m.get("diff_coeff", 0.0),
        mae_coeff=m.get("mae_coeff", 0.0),
        atom_type_coeff=m.get("atom_type_coeff", 1.0),
        num_scalar_props=6,
        mask_token_id=m["mask_token_id"],
        mask_replace_ratio=m["mask_replace_ratio"],
        mask_schedule_name=m["mask_schedule_name"],
        mask_schedule_start=m["mask_schedule_start"],
        mask_schedule_end=m["mask_schedule_end"],
        vocab_size=vocab_size,
        embedding_size=embedding_size,
    )

    device_t = torch.device(device)
    if MMadaModelLM is None:
        raise RuntimeError("MMadaModelLM is not available. Ensure your project imports are correct.")
    model = MMadaModelLM(model_cfg, tokenizer=tokenizer).to(device_t)
    model.eval()

    # Find a weight file inside the checkpoint directory
    candidates = []
    candidates += glob.glob(os.path.join(ckpt_dir, "model.safetensors"))
    candidates += glob.glob(os.path.join(ckpt_dir, "pytorch_model.bin"))
    candidates += glob.glob(os.path.join(ckpt_dir, "pytorch_model_*.bin"))
    if not candidates:
        raise FileNotFoundError(
            f"No model weights found in {ckpt_dir}. Expected one of: model.safetensors, pytorch_model.bin, pytorch_model_*.bin"
        )

    weights_path = sorted(candidates)[0]
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(weights_path, device=str(device_t))
    else:
        state = torch.load(weights_path, map_location=device_t)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing first 20): {missing[:20]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing first 20): {unexpected[:20]}")

    idx2z = DEFAULT_IDX2Z
    if gen_idx2z and isinstance(gen_idx2z, list) and len(gen_idx2z) == len(DEFAULT_IDX2Z):
        idx2z = np.asarray(gen_idx2z, dtype=np.int64)

    _STATE.update({"model": model, "tokenizer": tokenizer, "device": device_t, "idx2z": idx2z})
    print("[sampler-info] has sample_diffusion?:", hasattr(model, "sample_diffusion"))


def _filter_kwargs_for_func(func, kwargs: Dict) -> Dict:
    try:
        sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return {}


def sample_batch(
    batch_size: int,
    max_atoms: int,
    device: str = "cpu",
    gen_params: Optional[Dict] = None,
    **kwargs
) -> Dict[str, "np.ndarray"]:
    """Generate 3D conformations from 1D molecular inputs (SELFIES)."""
    use_dummy = kwargs.pop("dummy", False)
    require_diffusion = kwargs.pop("require_diffusion", False)
    input_selfies = kwargs.pop("input_selfies", None)  # NOTE: now we pass a per-batch chunk

    # Per-batch size
    if input_selfies is not None:
        actual_batch_size = len(input_selfies)
        # (optional) quick estimate of atoms per mol for logging
        atom_counts = [get_atom_count_from_selfies(s) for s in input_selfies]
        actual_max_atoms = min(max(atom_counts) if atom_counts else 5, max_atoms)
        print(f"Batch size: {actual_batch_size}, Max atoms in batch: {actual_max_atoms}")
    else:
        actual_batch_size = batch_size
        actual_max_atoms = max_atoms

    if use_dummy:
        B = actual_batch_size
        K = actual_max_atoms
        t = np.zeros((B, K), dtype=np.int64)
        c = np.zeros((B, K, 3), dtype=np.float32)
        m = np.zeros((B, K), dtype=bool)
        for b in range(B):
            n = min(5, K)
            t[b, :n] = np.array([6, 1, 1, 1, 1], dtype=np.int64)[:n]
            base = np.array([[0.12, 0.80, 0.10],
                             [0.12, 0.92, 0.10],
                             [0.22, 0.80, 0.10],
                             [0.12, 0.80, 0.22],
                             [0.02, 0.80, 0.10]], dtype=np.float32)[:n]
            c[b, :n, :] = base + 1e-4 * np.random.randn(n, 3).astype(np.float32)
            m[b, :n] = True
        return {"types": t, "coords": c, "mask": m}

    ckpt_dir = kwargs.get("ckpt", "")
    cfg_path = kwargs.get("config", None)
    if not ckpt_dir:
        raise ValueError("Please provide --ckpt pointing to a checkpoint directory.")
    _load_model_and_tokenizer(ckpt_dir, device, cfg_path, gen_idx2z=(gen_params or {}).get("idx2z"))

    model: "MMadaModelLM" = _STATE["model"]  # type: ignore
    dev: "torch.device" = _STATE["device"]   # type: ignore
    idx2z: np.ndarray = _STATE["idx2z"]

    if torch is None:
        raise RuntimeError("PyTorch is required for real sampling.")

    with torch.no_grad(), torch.inference_mode():
        use_diff = hasattr(model, "sample_diffusion")
        if require_diffusion and not use_diff:
            raise RuntimeError("require-diffusion set but model has no sample_diffusion().")

        print(f"[sampler-info] using: {'sample_diffusion' if use_diff else 'sample'}")
        if use_diff and gen_params:
            print("[sampler-params]",
                  "t_steps=", gen_params.get("t_steps"),
                  "guidance_scale=", gen_params.get("guidance_scale"),
                  "type_conf_frac=", gen_params.get("type_conf_frac"),
                  "type_topk=", gen_params.get("type_topk"),
                  "temperature=", gen_params.get("temperature"),
                  "scheduler_name=", gen_params.get("scheduler_name"))

        if use_diff:
            call_kwargs = dict(batch_size=actual_batch_size, max_atoms=actual_max_atoms, device=dev)
            if gen_params:
                call_kwargs.update(gen_params)
            if input_selfies is not None:
                call_kwargs["input_selfies"] = input_selfies  # per-batch chunk only
            call_kwargs = _filter_kwargs_for_func(model.sample_diffusion, call_kwargs)
            out = model.sample_diffusion(**call_kwargs)
        else:
            call_kwargs = dict(batch_size=actual_batch_size, max_atoms=actual_max_atoms, device=dev)
            if gen_params:
                call_kwargs.update(gen_params)
            if input_selfies is not None:
                call_kwargs["input_selfies"] = input_selfies
            call_kwargs = _filter_kwargs_for_func(model.sample, call_kwargs)
            out = model.sample(**call_kwargs)

        if "atom_idx" in out:
            atom_idx_t = out["atom_idx"].detach().clamp_(min=0, max=len(idx2z) - 1)
            atom_idx = atom_idx_t.cpu().numpy()
            types_z = idx2z[atom_idx]
        elif "types_z" in out:
            types_z = out["types_z"].detach().cpu().numpy().astype(np.int64)
        else:
            raise KeyError("Sampler output must contain 'atom_idx' or 'types_z'.")

        coords = out["coords"].detach().cpu().numpy().astype(np.float32)
        mask = out.get("mask", None)
        if mask is None:
            mask = (types_z > 0)
        else:
            mask = mask.detach().cpu().numpy().astype(bool)

        return {"types": types_z, "coords": coords, "mask": mask}


def main():
    p = argparse.ArgumentParser(description="Generate 3D conformations from SELFIES.")
    p.add_argument("--ckpt", type=str, default="", help="Path to model checkpoint directory.")
    p.add_argument("--config", type=str, default=None, help="Training YAML to rebuild model/tokenizer.")
    p.add_argument("--gen-config", type=str, default=None, help="Optional generation YAML (diffusion params).")
    p.add_argument("--input-selfies", type=str, default=None, help="Path to SELFIES file (one per line).")
    p.add_argument("--num", type=int, default=10000, help="Total molecules to generate (ignored if --input-selfies).")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for sampling.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-atoms", type=int, default=32, help="Model max atoms.")
    p.add_argument("--dataset", type=str, default="QM9", choices=["QM9", "Geom"])
    p.add_argument("--out", type=str, required=True, help="Output parquet path.")
    p.add_argument("--cap", type=int, default=32, help="Cap kept atoms after cleaning; 0=keep all.")
    p.add_argument("--origin-eps", type=float, default=1e-8, help="Treat near-zero coords as padding.")
    p.add_argument("--dedup-eps", type=float, default=1e-6, help="Merge coincident coords within this tol.")
    p.add_argument("--dummy", action="store_true", help="Use dummy sampler (no model needed).")
    p.add_argument("--print-every", type=int, default=2000, help="Log every N accepted mols.")
    p.add_argument("--require-diffusion", action="store_true", help="Fail if no sample_diffusion().")
    args = p.parse_args()

    if not args.dummy and not args.config:
        raise SystemExit("--config is required in non-dummy mode.")

    gen_cfg = _load_gen_cfg(args.gen_config)
    gen_params = gen_cfg["generation"]
    set_global_seed(gen_params.get("seed", None))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load SELFIES list if provided
    input_selfies: Optional[List[str]] = None
    if args.input_selfies:
        with open(args.input_selfies, 'r') as f:
            input_selfies = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(input_selfies)} SELFIES")
        args.num = len(input_selfies)  # override

    rows = []
    accepted = 0
    generated = 0
    t0 = time.time()

    while accepted < args.num:
        # IMPORTANT: pass ONLY the current batch slice to the sampler
        selfies_chunk = None
        if input_selfies is not None:
            start = generated
            end = min(generated + args.batch_size, args.num)
            selfies_chunk = input_selfies[start:end]

        batch = sample_batch(
            batch_size=args.batch_size,
            max_atoms=args.max_atoms,
            device=args.device,
            gen_params=gen_params,
            dummy=args.dummy,
            ckpt=args.ckpt,
            config=args.config,
            require_diffusion=args.require_diffusion,
            input_selfies=selfies_chunk,  # <-- FIXED: per-batch chunk only
        )

        t = np.asarray(batch["types"], dtype=np.int64)          # (B, K)
        c = np.asarray(batch["coords"], dtype=np.float32)       # (B, K, 3)
        m = np.asarray(batch.get("mask"), dtype=bool)           # (B, K)
        B = t.shape[0]
        generated += B

        for b in range(B):
            # optional dynamic cap from SELFIES-estimated atom count
            dynamic_cap = args.cap
            if selfies_chunk is not None and b < len(selfies_chunk):
                est_atoms = get_atom_count_from_selfies(selfies_chunk[b])
                dynamic_cap = min(est_atoms, args.cap) if args.cap > 0 else est_atoms

            tt, cc = postprocess_atoms(
                t[b], c[b],
                atoms_mask=m[b],
                dataset=args.dataset,
                origin_eps=args.origin_eps,
                dedup_eps=args.dedup_eps,
                cap=dynamic_cap,
            )
            if tt is None:
                continue
            rows.append({"types": tt.tolist(), "coords": cc.tolist()})
            accepted += 1
            if accepted % max(1, args.print_every) == 0:
                dt = time.time() - t0
                print(f"[info] accepted={accepted}/{args.num} (generated so far={generated}) | elapsed={dt:.1f}s")
            if accepted >= args.num:
                break

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, engine="pyarrow")
    print(f"[done] wrote {len(df)} rows to {out_path} (accepted={len(df)} of requested {args.num})")


if __name__ == "__main__":
    main()
