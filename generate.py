# generate.py
import os
import gc
import json
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import safetensors.torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from models.modeling_mmada import MMadaConfig, MMadaModelLM

# -----------------------------------------------------------------------------#
# helpers
# -----------------------------------------------------------------------------#
_pt = Chem.GetPeriodicTable()


def _atomic_number_to_symbol(z: int) -> str:
    return _pt.GetElementSymbol(int(z))


def _coords_types_to_mol(
    types: torch.Tensor, coords: torch.Tensor
) -> Chem.Mol:
    mol = Chem.RWMol()
    conf = Chem.Conformer()

    types_np = types.cpu().numpy()
    coords_np = coords.cpu().numpy()

    for i, z in enumerate(types_np):
        if z < 1 or z > 118:
            continue
        rd_idx = mol.AddAtom(Chem.Atom(_atomic_number_to_symbol(z)))
        conf.SetAtomPosition(rd_idx, Point3D(*map(float, coords_np[i])))

    mol.AddConformer(conf, assignId=True)
    return mol.GetMol()


def _add_gumbel(logits: torch.Tensor, temp: float) -> torch.Tensor:
    if temp == 0:
        return logits
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    return logits + g * temp


# -----------------------------------------------------------------------------#
# diffusion‑based 3‑D generation
# -----------------------------------------------------------------------------#
@torch.no_grad()
def generate_mol_3d(
    model: MMadaModelLM,
    tokenizer: AutoTokenizer,
    selfies: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = model.config

    tok = tokenizer(
        selfies,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg.max_selfies_length,
    ).to(device)

    bsz = 1
    coord = torch.randn(
        bsz,
        cfg.max_atoms,
        cfg.output_atom_coords_dim,
        device=device,
        dtype=torch.float32,
    )
    atype = torch.zeros(bsz, cfg.max_atoms, device=device, dtype=torch.long)
    mask = torch.ones(bsz, cfg.max_atoms, device=device, dtype=torch.bool)

    # noise schedule
    betas = torch.linspace(
        cfg.noise_schedule_beta_start,
        cfg.noise_schedule_beta_end,
        cfg.diffusion_timesteps,
        device=device,
    )
    alphas = 1.0 - betas
    a_bar = torch.cumprod(alphas, 0)
    sqrt_ab = torch.sqrt(a_bar)
    sqrt_one_minus_ab = torch.sqrt(1.0 - a_bar)

    n_steps = getattr(cfg, "generation_timesteps", 30)
    time_grid = torch.linspace(
        cfg.diffusion_timesteps - 1, 0, n_steps, device=device, dtype=torch.long
    )

    for t in time_grid:
        t_int = int(t.item())
        t_tensor = t.expand(bsz)

        # model prediction
        pred_coord_0, atype_logits, *_ = model(
            selfies_input_ids=tok.input_ids,
            selfies_attention_mask=tok.attention_mask,
            atom_vec=atype,
            coordinates=coord,
            atoms_mask=mask,
            timesteps=t_tensor,
        )

        pred_coord_0 = pred_coord_0 * mask.unsqueeze(-1)

        if t_int > 0:
            noise = (coord - sqrt_ab[t_int] * pred_coord_0) / sqrt_one_minus_ab[t_int]
            mean = (
                1.0
                / torch.sqrt(alphas[t_int])
                * (coord - (1 - alphas[t_int]) / sqrt_one_minus_ab[t_int] * noise)
            )
            var = (
                (1 - a_bar[t_int - 1]) / (1 - a_bar[t_int]) * betas[t_int]
            )
            coord = mean + torch.sqrt(var) * torch.randn_like(coord)
        else:
            coord = pred_coord_0

        temp = getattr(cfg, "generation_temperature_atom_type", 1.0)
        sample = _add_gumbel(atype_logits, temp).argmax(-1)
        atype = (sample * mask).long()

    return coord[0], atype[0]


# -----------------------------------------------------------------------------#
# end‑to‑end interface used by training/eval
# -----------------------------------------------------------------------------#
@torch.no_grad()
def generate_for_evaluation(
    model: MMadaModelLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    step: int | None = None,  
) -> List[Chem.Mol]:
    out = []
    for p in tqdm(prompts, desc="eval‑gen", leave=False):
        try:
            c, t = generate_mol_3d(model, tokenizer, p, device)
            from rdkit.Chem import SanitizeMol
            mol = _coords_types_to_mol(t, c)
            SanitizeMol(mol)                   
            out.append(mol)
        except Exception as e:
            tag = f" step={step}" if step is not None else ""
            print(f"[GEN‑ERR]{tag} prompt='{p[:30]}...' -> {e}", flush=True)
            out.append(None)
    return out


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/mmada_pretraining_stage2_llada_instruct.yaml") as f:
        cfg_yaml = yaml.safe_load(f)

    class C:
        def __init__(self, d):  # dotted‑access
            for k, v in d.items():
                setattr(self, k, C(v) if isinstance(v, dict) else v)

    args = C(cfg_yaml)

    model_cfg = MMadaConfig(**args.model.__dict__)
    ckpt_dir = args.experiment.resume_from_checkpoint
    weights = os.path.join(ckpt_dir, "model.safetensors")

    model = MMadaModelLM(model_cfg)
    sd = safetensors.torch.load_file(weights, device="cpu")
    model.load_state_dict({k.lstrip("module."): v for k, v in sd.items()})
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model.llm_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_parquet(args.model.data_path)
    selfies_col = df["selfies_string"].astype(str).tolist()

    results = []
    for idx, s in enumerate(tqdm(selfies_col, desc="generate‑all", unit="mol")):
        try:
            c, t = generate_mol_3d(model, tokenizer, s, device)
            mask = (t.cpu().numpy() > 0) & (t.cpu().numpy() <= 118)
            results.append(
                {
                    "id": int(df.loc[idx, "id"]) if "id" in df.columns else idx,
                    "selfies": s,
                    "coords": c.cpu().numpy()[mask].tolist(),
                    "types": t.cpu().numpy()[mask].tolist(),
                }
            )
        except Exception as e:
            print(f"[WARN] idx={idx} failed: {e}")

    out_path = "/projects/bezp/yfeng7/data/generated_mols.parquet"
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"saved {len(results)} molecules to {out_path}")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
