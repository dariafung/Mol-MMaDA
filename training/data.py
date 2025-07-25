# training/data.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Any, Tuple

from parquet.my_dataset import MolecularUnifiedDataset
from selfies import get_semantic_robust_alphabet


def get_pretraining_dataloader(config: Dict[str, Any]) -> Tuple[DataLoader, Any]:
    """
    Build and return a DataLoader (and tokenizer) for Chem-MMaDA pretraining.

    Required config keys:
        data_path (str)
        tokenizer_name_or_path (str)
        diffusion_timesteps (int)
        mask_schedule_name (str)
        mask_schedule_start (float)
        mask_schedule_end (float)
        atom_type_mask_prob (float)
        max_text_length (int)
        max_selfies_length (int)
        max_atoms (int)
        train_batch_size (int)
        num_workers (int)

    Optional keys:
        selfies_mask_ratio (float)
        include_edge_bond_dist (bool)
        include_rdmol2selfies (bool)
        shuffle (bool)
        repeat (bool)
        buffer_size (int)
        rank (int)
        world_size (int)
        pin_memory (bool)
        persistent_workers (bool)
        atom_mask_token_id (int)   # for masking atom types specifically
    """
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name_or_path"])

    # Add SELFIES tokens if missing
    selfies_tokens = set(get_semantic_robust_alphabet())
    existing = set(tokenizer.get_vocab().keys())
    to_add = list(selfies_tokens - existing)
    if to_add:
        tokenizer.add_tokens(to_add)

    # Ensure mask_token_id is defined
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        if tokenizer.pad_token_id is not None:
            mask_token_id = tokenizer.pad_token_id
        else:
            # fallback: pick an unused id (rarely needed if you just add a mask token)
            for i in range(len(tokenizer), len(tokenizer) + 1000):
                if i not in tokenizer.get_vocab().values():
                    mask_token_id = i
                    break
            else:
                raise ValueError("No valid mask_token_id or pad_token_id found.")

    # Atom-mask ID may differ from text/SELFIES mask
    atom_mask_token_id = config.get("atom_mask_token_id", 0)

    # 2. Dataset
    dataset = MolecularUnifiedDataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        diffusion_timesteps=config["diffusion_timesteps"],
        mask_schedule_name=config["mask_schedule_name"],
        mask_schedule_start=config["mask_schedule_start"],
        mask_schedule_end=config["mask_schedule_end"],
        selfies_mask_ratio=config.get("selfies_mask_ratio"),
        atom_type_mask_prob=config["atom_type_mask_prob"],
        rank=config.get("rank", 0),
        world_size=config.get("world_size", 1),
        shuffle=config.get("shuffle", True),
        repeat=config.get("repeat", True),
        buffer_size=config.get("buffer_size", 100),
        max_text_length=config["max_text_length"],
        max_selfies_length=config["max_selfies_length"],
        max_atoms=config["max_atoms"],
        include_edge_bond_dist=config.get("include_edge_bond_dist", False),
        include_rdmol2selfies=config.get("include_rdmol2selfies", False),
        atom_mask_token_id=atom_mask_token_id,
    )

    # 3. DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        collate_fn=dataset.collate_fn,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", False),
    )

    return dataloader, tokenizer


if __name__ == "__main__":
    # Basic smoke test
    cfg = {
        "data_path": "/projects/bezp/yfeng7/data/m3_molecular_data.parquet",
        "tokenizer_name_or_path": "GSAI-ML/LLaDA-8B-Instruct",
        "diffusion_timesteps": 1000,
        "mask_schedule_name": "linear",
        "mask_schedule_start": 0.0001,
        "mask_schedule_end": 0.02,
        "selfies_mask_ratio": 0.15,
        "atom_type_mask_prob": 0.15,
        "max_text_length": 512,
        "max_selfies_length": 256,
        "max_atoms": 128,
        "include_edge_bond_dist": True,
        "include_rdmol2selfies": False,
        "train_batch_size": 4,
        "num_workers": 2,
        "shuffle": True,
        "repeat": False,
        "buffer_size": 10,
        "rank": 0,
        "world_size": 1,
        "atom_mask_token_id": 0,
    }

    print("---- DataLoader Smoke Test ----")
    dl, tok = get_pretraining_dataloader(cfg)
    for i, batch in enumerate(dl):
        print(f"\nBatch {i+1}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}, {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
        if i >= 1:
            break
    print("Smoke test complete.")
