import os
import gc
from typing import List, Tuple, Iterable
from collections.abc import Mapping

import torch
import numpy as np
import pandas as pd
import safetensors.torch
import traceback
from rdkit import Chem
from rdkit.Geometry import Point3D
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import yaml
import glob
from pathlib import Path

from models.modeling_mmada import MMadaConfig, MMadaModelLM

# -------------------- RDKit helpers --------------------
_pt = Chem.GetPeriodicTable()


def _atomic_number_to_symbol(z: int) -> str:
    return _pt.GetElementSymbol(int(z))


def _coords_types_to_mol(types: torch.Tensor, coords: torch.Tensor) -> Chem.Mol:
    mol = Chem.RWMol()
    conf = Chem.Conformer()

    types_np = types.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()

    for i, z in enumerate(types_np):
        if z < 1 or z > 118:
            continue
        rd_idx = mol.AddAtom(Chem.Atom(_atomic_number_to_symbol(int(z))))
        conf.SetAtomPosition(rd_idx, Point3D(*map(float, coords_np[i])))

    mol.AddConformer(conf, assignId=True)
    return mol.GetMol()


# -------------------- misc tensor helpers --------------------
def _add_gumbel(logits: torch.Tensor, temp: float) -> torch.Tensor:
    if temp == 0:
        return logits
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    return logits + g * temp


def _to_tensor(x, device, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device)
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device=device)
    if isinstance(x, (list, tuple)):
        return torch.as_tensor(x, device=device, dtype=dtype)
    raise TypeError(f"Cannot convert type {type(x)} to tensor")


def _iter_tensors(obj) -> Iterable[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, np.ndarray):
        yield torch.from_numpy(obj)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)
    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from _iter_tensors(v)
    # ignore other types


# -------------------- output picking by shape --------------------
def _pick_outputs_by_shape(out, cfg, bsz, max_atoms) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (pred_coord_0, atype_logits) by shape:
      - coords: last dim == cfg.output_atom_coords_dim (usually 3)
      - types : last dim == cfg.output_atom_type_dim (usually 120)
    Prefer tensors whose second-to-last dim equals max_atoms.
    """
    coord_cands = []
    type_cands = []
    for t in _iter_tensors(out):
        if not isinstance(t, torch.Tensor):
            continue
        if t.dim() >= 2 and t.shape[-1] == getattr(cfg, "output_atom_coords_dim", 3):
            coord_cands.append(t)
        if t.dim() >= 2 and t.shape[-1] == getattr(cfg, "output_atom_type_dim", 120):
            type_cands.append(t)

    if not coord_cands:
        raise RuntimeError("No coordinate-like tensor (last dim == 3) found in model outputs.")
    if not type_cands:
        raise RuntimeError("No atom-type-logits-like tensor (last dim == output_atom_type_dim) found in model outputs.")

    def _prefer_atoms_dim(cands):
        # prefer [..., max_atoms, feat]
        for t in cands:
            if t.dim() >= 3 and t.shape[-2] == max_atoms:
                return t
        # otherwise just first
        return cands[0]

    coord = _prefer_atoms_dim(coord_cands)
    atype_logits = _prefer_atoms_dim(type_cands)

    # make sure batch dimension exists
    if coord.dim() == 2:
        coord = coord.unsqueeze(0)
    if atype_logits.dim() == 2:
        atype_logits = atype_logits.unsqueeze(0)

    # move batch to front if not already
    if coord.shape[0] != bsz and coord.dim() >= 3:
        # try to find a dimension equal to bsz and move it to front
        for d in range(coord.dim()):
            if coord.shape[d] == bsz:
                coord = coord.movedim(d, 0)
                break
    if atype_logits.shape[0] != bsz and atype_logits.dim() >= 3:
        for d in range(atype_logits.dim()):
            if atype_logits.shape[d] == bsz:
                atype_logits = atype_logits.movedim(d, 0)
                break

    return coord, atype_logits


# -------------------- diffusion-based 3D generation --------------------
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
        bsz, cfg.max_atoms, cfg.output_atom_coords_dim, device=device, dtype=torch.float32
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
    time_grid = [int(x) for x in torch.linspace(
        cfg.diffusion_timesteps - 1, 0, n_steps, dtype=torch.long, device=device
    ).tolist()]

    warned_types_len = False

    for t_int in time_grid:
        t_batch = torch.full((bsz,), t_int, dtype=torch.long, device=device)
        try:
            out = model(
                selfies_input_ids=tok.input_ids,
                selfies_attention_mask=tok.attention_mask,
                atom_vec=atype,
                coordinates=coord,
                atoms_mask=mask,
                timesteps=t_batch,
            )
        except Exception:
            out = model(
                selfies_input_ids=tok.input_ids,
                selfies_attention_mask=tok.attention_mask,
                atom_vec=atype,
                coordinates=coord,
                atoms_mask=mask,
                timesteps=t_int,
            )

        pred_coord_0, atype_logits = _pick_outputs_by_shape(out, cfg, bsz=bsz, max_atoms=cfg.max_atoms)
        pred_coord_0 = _to_tensor(pred_coord_0, device, dtype=torch.float32)
        atype_logits = _to_tensor(atype_logits, device)

        # Expect shapes: [bsz, max_atoms, 3] and [bsz, max_atoms, output_atom_type_dim]
        if pred_coord_0.shape[-1] != cfg.output_atom_coords_dim:
            raise RuntimeError(
                f"pred_coord_0 last dim mismatch: expected {cfg.output_atom_coords_dim}, got {pred_coord_0.shape[-1]}"
            )

        # If type logits length does not equal max_atoms, safely trim/pad once.
        if atype_logits.dim() >= 3 and atype_logits.shape[-2] != cfg.max_atoms:
            if not warned_types_len:
                print(
                    f"[WARN] atype_logits len {atype_logits.shape[-2]} != max_atoms {cfg.max_atoms}; "
                    f"{'truncating' if atype_logits.shape[-2] > cfg.max_atoms else 'padding'} to match.",
                    flush=True,
                )
                warned_types_len = True
            if atype_logits.shape[-2] > cfg.max_atoms:
                atype_logits = atype_logits[..., :cfg.max_atoms, :]
            else:
                pad = cfg.max_atoms - atype_logits.shape[-2]
                pad_tail = torch.zeros(
                    (*atype_logits.shape[:-2], pad, atype_logits.shape[-1]),
                    device=atype_logits.device, dtype=atype_logits.dtype
                )
                atype_logits = torch.cat([atype_logits, pad_tail], dim=-2)

        pred_coord_0 = pred_coord_0 * mask.unsqueeze(-1)

        if t_int > 0:
            noise = (coord - sqrt_ab[t_int] * pred_coord_0) / sqrt_one_minus_ab[t_int]
            mean = (1.0 / torch.sqrt(alphas[t_int])) * (
                coord - (1 - alphas[t_int]) / sqrt_one_minus_ab[t_int] * noise
            )
            var = ((1 - a_bar[t_int - 1]) / (1 - a_bar[t_int])) * betas[t_int]
            coord = mean + torch.sqrt(var) * torch.randn_like(coord)
        else:
            coord = pred_coord_0

        temp = getattr(cfg, "generation_temperature_atom_type", 1.0)
        sample = _add_gumbel(atype_logits, temp).argmax(-1)  # [bsz, max_atoms]
        atype = (sample * mask).long()

    return coord[0], atype[0]


# -------------------- eval wrapper --------------------
@torch.no_grad()
def generate_for_evaluation(
    model: MMadaModelLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    step: int | None = None,
) -> List[Chem.Mol]:
    out = []
    for p in tqdm(prompts, desc="eval-gen", leave=False):
        try:
            c, t = generate_mol_3d(model, tokenizer, p, device)
            from rdkit.Chem import SanitizeMol
            mol = _coords_types_to_mol(t, c)
            SanitizeMol(mol)
            out.append(mol)
        except Exception as e:
            traceback.print_exc()
            tag = f" step={step}" if step is not None else ""
            print(f"[GEN-ERR]{tag} prompt='{p[:30]}...' -> {e}", flush=True)
            out.append(None)
    return out


# -------------------- tokenizer/weights helpers --------------------
def _has_tokenizer_files(p: str) -> bool:
    if not isinstance(p, str) or not os.path.isdir(p):
        return False
    names = set(os.listdir(p))
    needed = {"tokenizer.json", "tokenizer_config.json"}
    alt = {"vocab.json", "merges.txt"}
    return (needed <= names) or (alt <= names)


def _resolve_weights_path(ckpt_dir: str) -> str:
    cand1 = os.path.join(ckpt_dir, "model.safetensors")
    cand2 = os.path.join(ckpt_dir, "pytorch_model.safetensors")
    if os.path.isfile(cand1):
        return cand1
    if os.path.isfile(cand2):
        return cand2
    idx = os.path.join(ckpt_dir, "pytorch_model.safetensors.index.json")
    shards = glob.glob(os.path.join(ckpt_dir, "pytorch_model-*.safetensors"))
    if os.path.isfile(idx) and shards:
        raise FileNotFoundError(
            f"Found sharded weights in {ckpt_dir} but no single-file safetensors. "
            f"Merge shards into model.safetensors or modify the loader to use sharded weights."
        )
    raise FileNotFoundError(f"No safetensors found under {ckpt_dir}")


# -------------------- main --------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/mmada_pretraining_stage1_llada_instruct.yaml") as f:
        cfg_yaml = yaml.safe_load(f)

    class C:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, C(v) if isinstance(v, dict) else v)

    args = C(cfg_yaml)

    ckpt_dir = args.experiment.resume_from_checkpoint
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"resume_from_checkpoint is not a directory: {ckpt_dir}")

    # tokenizer
    tok_dir_candidates = [
        ckpt_dir,
        str(Path(ckpt_dir).parent),
        args.model.llm_model_name_or_path,
    ]
    for _p in tok_dir_candidates:
        if _has_tokenizer_files(_p):
            tokenizer_dir = _p
            break
    else:
        tokenizer_dir = args.model.llm_model_name_or_path
    print(f"[INFO] loading tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"[INFO] tokenizer vocab size = {vocab_size}")

    # model
    model_cfg = MMadaConfig(**args.model.__dict__)
    model = MMadaModelLM(model_cfg)

    # load weights + align embedding rows
    weights = _resolve_weights_path(ckpt_dir)
    print(f"[INFO] loading weights: {weights}")
    sd = safetensors.torch.load_file(weights, device="cpu")
    state = {k.removeprefix("module."): v for k, v in sd.items()}

    wte_key = "llm_backbone.model.transformer.wte.weight"
    if wte_key in state:
        ckpt_vocab, dim = state[wte_key].shape
        emb = model.llm_backbone.model.transformer.wte
        old_vocab, old_dim = emb.weight.shape
        if old_dim != dim:
            raise RuntimeError(
                f"Embedding dim mismatch: model={old_dim}, ckpt={dim}. Check backbone hidden size."
            )
        if old_vocab != ckpt_vocab:
            print(f"[INFO] resizing token embeddings: model_vocab={old_vocab} -> ckpt_vocab={ckpt_vocab}")
            new_emb = torch.nn.Embedding(ckpt_vocab, dim, dtype=emb.weight.dtype, device=emb.weight.device)
            with torch.no_grad():
                num = min(old_vocab, ckpt_vocab)
                new_emb.weight[:num].copy_(emb.weight[:num])
                if ckpt_vocab > old_vocab:
                    torch.nn.init.normal_(new_emb.weight[num:], mean=0.0, std=0.02)
            model.llm_backbone.model.transformer.wte = new_emb

            # try to resize a language head if present
            obj = model.llm_backbone.model
            for path in ("lm_head", "transformer.lm_head"):
                cur = obj
                ok = True
                for part in path.split("."):
                    if hasattr(cur, part):
                        cur = getattr(cur, part)
                    else:
                        ok = False
                        break
                if ok and hasattr(cur, "weight"):
                    head_w = cur.weight
                    if head_w.shape[1] == dim and head_w.shape[0] != ckpt_vocab:
                        print(f"[INFO] resizing {path}: {tuple(head_w.shape)} -> ({ckpt_vocab}, {dim})")
                        new_w = torch.empty((ckpt_vocab, dim), dtype=head_w.dtype, device=head_w.device)
                        with torch.no_grad():
                            n2 = min(head_w.shape[0], ckpt_vocab)
                            new_w[:n2].copy_(head_w[:n2])
                            if ckpt_vocab > head_w.shape[0]:
                                torch.nn.init.normal_(new_w[n2:], mean=0.0, std=0.02)
                        cur.weight = torch.nn.Parameter(new_w)
                    break

    if wte_key in state:
        print(f"[INFO] checkpoint embedding shape = {tuple(state[wte_key].shape)}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))

    model.to(device).eval()

    # dataset
    data_path = args.model.data_path
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")
    df = pd.read_parquet(data_path, engine="pyarrow")

    # ---- random sampling controls (no change to diffusion steps) ----
    sample_n = int(os.environ.get("SAMPLE_N", "0")) or None
    sample_frac = float(os.environ.get("SAMPLE_FRAC", "0") or 0) or None
    sample_every = int(os.environ.get("SAMPLE_EVERY", "0")) or None

    if sample_n is not None:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
    elif sample_frac is not None:
        df = df.sample(frac=min(max(sample_frac, 0.0), 1.0), random_state=42)
    elif sample_every is not None and sample_every > 1:
        df = df.iloc[::sample_every]

    selfies_col = df["selfies_string"].astype(str).tolist()

    results = []
    for idx, s in enumerate(tqdm(selfies_col, desc="generate-all", unit="mol")):
        try:
            c, t = generate_mol_3d(model, tokenizer, s, device)
            t_np = t.detach().cpu().numpy()
            c_np = c.detach().cpu().numpy()
            mask = (t_np > 0) & (t_np <= 118)
            results.append(
                {
                    "id": int(df.iloc[idx]["id"]) if "id" in df.columns else idx,
                    "selfies": s,
                    "coords": c_np[mask].tolist(),
                    "types": t_np[mask].tolist(),
                }
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[WARN] idx={idx} failed: {e}")

    out_path = os.environ.get("OUT_PATH", "/projects/bezp/yfeng7/data/generated_mols.parquet")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"saved {len(results)} molecules to {out_path}")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
