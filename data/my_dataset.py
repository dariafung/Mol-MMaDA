import collections
import glob
import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from rdkit import Chem
from selfies import get_semantic_robust_alphabet
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from training.training_utils import get_mask_schedule, mask_or_random_replace_tokens


# ---- Atom vocabulary for QM9 (compact indices) ----
# 0 = PAD/MASK, 1 = H, 2 = C, 3 = N, 4 = O, 5 = F
QM9_Z2IDX = {0: 0, 1: 1, 6: 2, 7: 3, 8: 4, 9: 5}


def atom_to_id(symbol: str) -> int:
    """Convert atomic symbol to atomic number (Z)."""
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    except Exception:
        return 0


def parse_molecular_3d_data(raw_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Parse one row/dict into tensors:
      - atom_vec: long (N,) containing atomic numbers (Z)
      - coordinates: float (N,3)
      - (optional) edge_type, bond_type, dist, rdmol2selfies
    NOTE: we return Z here; the Z->idx mapping happens inside the dataset __iter__.
    """
    try:
        atom_raw = json.loads(raw_data.get("atom_vec_str", "[]"))
        coords_list = json.loads(raw_data.get("coordinates_str", "[]"))

        atom_ids = [
            int(x) if isinstance(x, int) or str(
                x).isdigit() else atom_to_id(str(x))
            for x in atom_raw
        ]
        atom_vec = torch.tensor(atom_ids, dtype=torch.long)
        coordinates = torch.tensor(coords_list, dtype=torch.float32)

        edge_type = None
        if raw_data.get("edge_type_str"):
            edge_type = torch.tensor(json.loads(
                raw_data["edge_type_str"]), dtype=torch.long)

        bond_type = None
        if raw_data.get("bond_type_str"):
            bond_type = torch.tensor(json.loads(
                raw_data["bond_type_str"]), dtype=torch.long)

        dist = None
        if raw_data.get("dist_str"):
            dist = torch.tensor(json.loads(
                raw_data["dist_str"]), dtype=torch.float32)

        rdmol2selfies = None
        if raw_data.get("rdmol2selfies_str"):
            rdmol2selfies = torch.tensor(json.loads(
                raw_data["rdmol2selfies_str"]), dtype=torch.float32)

        return {
            "atom_vec": atom_vec,             # Z (atomic number)
            "coordinates": coordinates,       # (N,3)
            "edge_type": edge_type,
            "bond_type": bond_type,
            "dist": dist,
            "rdmol2selfies": rdmol2selfies,
        }
    except Exception:
        return {}


class MolecularUnifiedDataset(IterableDataset):
    """
    Iterable dataset that:
      - Reads parquet rows
      - Tokenizes SELFIES/text
      - Centers coordinates
      - Converts atomic number Z -> compact idx (QM9_Z2IDX)
      - Applies masking to atom indices (0 is PAD/MASK class)
      - Pads/truncates to max_atoms and collates into batches
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        mask_token_id: int,          # <-- text/SELFIES mask id (NOT atom type)
        diffusion_timesteps: int,
        mask_schedule_name: str,
        mask_schedule_start: float,
        mask_schedule_end: float,
        selfies_mask_ratio: Optional[float] = None,
        atom_type_mask_prob: float = 0.15,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        repeat: bool = True,
        buffer_size: int = 100,
        max_text_length: int = 512,
        max_selfies_length: int = 256,
        max_atoms: int = 256,
        include_edge_bond_dist: bool = False,
        include_rdmol2selfies: bool = False,
        atom_mask_token_id: int = 0,  # <-- atom-type MASK/PAD idx (QM9 uses 0)
    ):
        super().__init__()

        self.data_path = data_path
        if os.path.isdir(data_path):
            self.files = sorted(
                glob.glob(os.path.join(data_path, "*.parquet")))
        else:
            self.files = sorted(glob.glob(data_path))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found at {data_path}")

        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_text_length = max_text_length
        self.max_selfies_length = max_selfies_length
        self.max_atoms = max_atoms

        self.tokenizer = tokenizer
        self.include_edge_bond_dist = include_edge_bond_dist
        self.include_rdmol2selfies = include_rdmol2selfies

        # mask_token_id is for text/SELFIES diffusion; atom_mask_token_id is for atom class masking (0)
        self.mask_token_id = mask_token_id
        self.atom_mask_token_id = atom_mask_token_id

        self.diffusion_timesteps = diffusion_timesteps
        self.mask_schedule_values = get_mask_schedule(
            mask_schedule_name,
            timesteps=diffusion_timesteps,
            start=mask_schedule_start,
            end=mask_schedule_end,
        )
        self.selfies_mask_ratio = selfies_mask_ratio
        self.atom_type_mask_prob = atom_type_mask_prob

        self.prop_cols = ["mu", "alpha", "homo", "lumo", "gap", "cv"]

    def read_parquet_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            df = df[df["selfies_string"].notna() & (
                df["selfies_string"] != "")]
            for _, row in df.iterrows():
                yield row.to_dict()
        except Exception as e:
            print(f"Error reading parquet file {file_path}: {e}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        files_for_worker = self.files[self.rank::self.world_size]

        seed = self.rank + worker_id + os.getpid()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        while True:
            file_list = list(files_for_worker)
            if self.shuffle:
                random.shuffle(file_list)

            for file in file_list:
                data_gen = self.read_parquet_file(file)
                buffer = []

                for raw in data_gen:
                    try:
                        selfies_str = raw.get("selfies_string", "")
                        if not selfies_str:
                            continue

                        # ----- Tokenize SELFIES -----
                        tok_selfies = self.tokenizer(
                            selfies_str,
                            truncation=True,
                            max_length=self.max_selfies_length,
                            padding="do_not_pad",
                            return_tensors="pt",
                        )
                        selfies_ids_clean = tok_selfies.input_ids[0]
                        selfies_attn_mask = tok_selfies.attention_mask[0]

                        # Diffusion timestep & mask ratio for SELFIES tokens
                        timestep = torch.randint(
                            0, self.diffusion_timesteps, (1,)).item()
                        current_ratio = self.mask_schedule_values[timestep].item(
                        )
                        if self.selfies_mask_ratio is not None:
                            current_ratio = self.selfies_mask_ratio

                        masked_selfies_ids, true_selfies_labels, _ = mask_or_random_replace_tokens(
                            selfies_ids_clean.unsqueeze(0),
                            # this is tokenizer.mask_token_id (text side)
                            self.mask_token_id,
                            mask_ratio=current_ratio,
                            vocab_size=getattr(self.tokenizer, "vocab_size", None) or len(
                                self.tokenizer),
                            is_train=True,
                        )
                        masked_selfies_ids = masked_selfies_ids.squeeze(0)
                        true_selfies_labels = true_selfies_labels.squeeze(0)

                        # ----- Tokenize text -----
                        text_str = raw.get("text_description", "")
                        if not isinstance(text_str, str):
                            text_str = ""
                        tok_text = self.tokenizer(
                            text_str,
                            truncation=True,
                            max_length=self.max_text_length,
                            padding="do_not_pad",
                            return_tensors="pt",
                        )
                        text_input_ids = tok_text.input_ids[0]
                        text_attention_mask = tok_text.attention_mask[0]

                        # ----- Parse molecule 3D (Z + coords) -----
                        mol_3d = parse_molecular_3d_data(raw)
                        if not mol_3d:
                            continue

                        atom_vec_Z = mol_3d.get("atom_vec")           # Z (N,)
                        coordinates = mol_3d.get("coordinates")       # (N,3)
                        if atom_vec_Z is None or coordinates is None or atom_vec_Z.shape[0] == 0:
                            continue
                        if atom_vec_Z.shape[0] > self.max_atoms:
                            # skip overly large molecules (keep data/model consistent)
                            continue

                        # Center and normalize coordinates per molecule
                        if coordinates.shape[0] > 0:
                            # Center coordinates
                            centroid = coordinates.mean(dim=0)
                            coordinates = coordinates - centroid

                            # Normalize coordinates to reasonable scale
                            distances = torch.norm(coordinates, dim=1)
                            if distances.numel() > 0:
                                scale_factor = torch.quantile(distances, 0.95)
                                if scale_factor > 1e-6:
                                    coordinates = coordinates / scale_factor * \
                                        2.0  # Scale to [-2, 2] range

                        # ----- Z -> compact idx (QM9) -----
                        # Any atom Z not in QM9_Z2IDX will be mapped to 0 (PAD/MASK) and effectively ignored.
                        atom_idx = torch.tensor(
                            [QM9_Z2IDX.get(int(z), 0)
                             for z in atom_vec_Z.tolist()],
                            dtype=torch.long,
                        )

                        # ----- Pad atoms (idx) and coordinates -----
                        num_atoms = atom_idx.shape[0]

                        padded_atom_idx = torch.full(
                            (self.max_atoms,), 0, dtype=torch.long)  # 0=PAD/MASK
                        padded_atom_idx[:num_atoms] = atom_idx

                        padded_coords = torch.zeros(
                            (self.max_atoms, 3), dtype=torch.float32)
                        padded_coords[:num_atoms] = coordinates

                        atoms_mask = torch.zeros(
                            (self.max_atoms,), dtype=torch.bool)
                        atoms_mask[:num_atoms] = True

                        # ----- Mask atom indices for discrete diffusion (atom types) -----
                        masked_atom_idx = padded_atom_idx.clone()
                        if self.atom_type_mask_prob > 0:
                            rand_mask = torch.rand_like(
                                masked_atom_idx, dtype=torch.float32) < self.atom_type_mask_prob
                            final_mask = rand_mask & atoms_mask
                            # 0
                            masked_atom_idx[final_mask] = self.atom_mask_token_id

                            # Ensure at least one real (non-mask) atom remains when atoms exist
                            if atoms_mask.any() and (masked_atom_idx != self.atom_mask_token_id)[atoms_mask].sum() == 0:
                                real_idxs = atoms_mask.nonzero(
                                    as_tuple=True)[0]
                                ridx = real_idxs[torch.randint(
                                    len(real_idxs), (1,)).item()]
                                masked_atom_idx[ridx] = padded_atom_idx[ridx]

                        sample = {
                            "id": raw.get("id", str(random.randint(0, 10**9))),
                            # SELFIES (masked input + labels)
                            "selfies_input_ids": masked_selfies_ids,
                            "selfies_attention_mask": selfies_attn_mask,
                            "true_selfies_labels": true_selfies_labels,
                            # text
                            "text_input_ids": text_input_ids,
                            "text_attention_mask": text_attention_mask,
                            # atom types (idx) and coords
                            # input (masked idx)
                            "atom_vec": masked_atom_idx,
                            # labels (idx)
                            "true_atom_vec": padded_atom_idx,
                            "coordinates": padded_coords,          # input coords
                            "true_coordinates": padded_coords.clone(),  # labels coords
                            "atoms_mask": atoms_mask,
                            "timesteps": torch.tensor([timestep], dtype=torch.long),
                        }

                        # Optionals
                        if self.include_edge_bond_dist:
                            if mol_3d.get("edge_type") is not None:
                                et = mol_3d["edge_type"]
                                pad = torch.zeros(
                                    (self.max_atoms, self.max_atoms), dtype=torch.long)
                                pad[: et.shape[0], : et.shape[1]] = et
                                sample["edge_type"] = pad

                            if mol_3d.get("bond_type") is not None:
                                bt = mol_3d["bond_type"]
                                pad = torch.zeros(
                                    (self.max_atoms, self.max_atoms), dtype=torch.long)
                                pad[: bt.shape[0], : bt.shape[1]] = bt
                                sample["bond_type"] = pad

                            if mol_3d.get("dist") is not None:
                                dist = mol_3d["dist"]
                                pad = torch.zeros(
                                    (self.max_atoms, self.max_atoms), dtype=torch.float32)
                                pad[: dist.shape[0], : dist.shape[1]] = dist
                                sample["dist"] = pad

                        if self.include_rdmol2selfies and mol_3d.get("rdmol2selfies") is not None:
                            r2s = mol_3d["rdmol2selfies"]
                            pad = torch.zeros(
                                (self.max_atoms, self.max_selfies_length), dtype=torch.float32)
                            a_dim = min(r2s.shape[0], self.max_atoms)
                            s_dim = min(r2s.shape[1], self.max_selfies_length)
                            pad[:a_dim, :s_dim] = r2s[:a_dim, :s_dim]
                            sample["rdmol2selfies"] = pad

                        # properties (optional)
                        try:
                            props = []
                            for c in self.prop_cols:
                                v = raw.get(c, None)
                                if v is None or (isinstance(v, float) and np.isnan(v)):
                                    props.append(float("nan"))
                                else:
                                    props.append(float(v))
                            sample["true_props"] = torch.tensor(
                                props, dtype=torch.float32)
                        except Exception:
                            sample["true_props"] = torch.tensor(
                                [float("nan")] * len(self.prop_cols), dtype=torch.float32)

                        buffer.append(sample)
                        if len(buffer) >= self.buffer_size:
                            if self.shuffle:
                                random.shuffle(buffer)
                            for it in buffer:
                                yield it
                            buffer = []

                    except Exception as e:
                        print(f"Error processing sample: {e}")
                        continue

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for it in buffer:
                        yield it

            if not self.repeat:
                break

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        buckets = collections.defaultdict(list)
        for item in batch:
            for k, v in item.items():
                if v is not None:
                    buckets[k].append(v)

        out: Dict[str, Any] = {}

        # SELFIES
        if "selfies_input_ids" in buckets:
            padded = self.tokenizer.pad(
                {"input_ids": buckets["selfies_input_ids"],
                    "attention_mask": buckets["selfies_attention_mask"]},
                padding=True,
                return_tensors="pt",
            )
            out["selfies_input_ids"] = padded["input_ids"]
            out["selfies_attention_mask"] = padded["attention_mask"]
        else:
            out["selfies_input_ids"] = torch.empty(0, dtype=torch.long)
            out["selfies_attention_mask"] = torch.empty(0, dtype=torch.long)

        if "true_selfies_labels" in buckets:
            max_len = max(s.size(0) for s in buckets["true_selfies_labels"])
            padded_labels = []
            for s in buckets["true_selfies_labels"]:
                pad_len = max_len - s.size(0)
                # -100 for CE ignore_index
                padded_labels.append(F.pad(s, (0, pad_len), "constant", -100))
            out["true_selfies_labels"] = torch.stack(padded_labels, dim=0)
        else:
            out["true_selfies_labels"] = torch.empty(0, dtype=torch.long)

        # text
        if "text_input_ids" in buckets:
            padded = self.tokenizer.pad(
                {"input_ids": buckets["text_input_ids"],
                    "attention_mask": buckets["text_attention_mask"]},
                padding=True,
                return_tensors="pt",
            )
            out["text_input_ids"] = padded["input_ids"]
            out["text_attention_mask"] = padded["attention_mask"]
        else:
            out["text_input_ids"] = torch.empty(0, dtype=torch.long)
            out["text_attention_mask"] = torch.empty(0, dtype=torch.long)

        # props
        if "true_props" in buckets and len(buckets["true_props"]) > 0:
            kept = [t for t in buckets["true_props"]
                    if isinstance(t, torch.Tensor)]
            if len(kept) == len(buckets["true_props"]) and len(kept) > 0:
                out["true_props"] = torch.stack(kept, dim=0).float()  # (B,6)
            else:
                out["true_props"] = torch.empty(
                    len(batch), 0, dtype=torch.float32)
        else:
            out["true_props"] = torch.empty(len(batch), 0, dtype=torch.float32)

        # atoms/coords (+ optionals)
        keys_to_stack = ["atom_vec", "coordinates", "atoms_mask",
                         "timesteps", "true_atom_vec", "true_coordinates"]
        if self.include_edge_bond_dist:
            keys_to_stack.extend(["edge_type", "bond_type", "dist"])
        if self.include_rdmol2selfies:
            keys_to_stack.append("rdmol2selfies")

        for k in keys_to_stack:
            if k in buckets and len(buckets[k]) > 0:
                out[k] = torch.stack(buckets[k], dim=0)
            else:
                if k in ["coordinates", "true_coordinates"]:
                    out[k] = torch.empty(
                        len(batch), self.max_atoms, 3, dtype=torch.float32)
                elif k == "atoms_mask":
                    out[k] = torch.empty(
                        len(batch), self.max_atoms, dtype=torch.bool)
                elif k == "timesteps":
                    out[k] = torch.empty(len(batch), 1, dtype=torch.long)
                elif k in ["edge_type", "bond_type"]:
                    out[k] = torch.empty(
                        len(batch), self.max_atoms, self.max_atoms, dtype=torch.long)
                elif k == "dist":
                    out[k] = torch.empty(
                        len(batch), self.max_atoms, self.max_atoms, dtype=torch.float32)
                elif k == "rdmol2selfies":
                    out[k] = torch.empty(
                        len(batch), self.max_atoms, self.max_selfies_length, dtype=torch.float32)
                else:
                    out[k] = torch.empty(
                        len(batch), self.max_atoms, dtype=torch.long)

        return out


if __name__ == "__main__":
    parquet_path = "/projects/bezp/yfeng7/data/qm9_molecular_data.parquet"

    try:
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    except Exception:
        print("Fallback to bert-base-uncased tokenizer for smoke test.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Extend tokenizer with SELFIES alphabet if needed (text side only)
    new_tokens = set(get_semantic_robust_alphabet()) - \
        set(tokenizer.get_vocab().keys())
    if new_tokens:
        tokenizer.add_tokens(list(new_tokens))

    # Text/SELFIES mask token id (unrelated to atom-type mask=0)
    mask_token_id_test = tokenizer.mask_token_id
    if mask_token_id_test is None:
        mask_token_id_test = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    dataset = MolecularUnifiedDataset(
        data_path=parquet_path,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id_test,  # text/SELFIES mask id
        diffusion_timesteps=1000,
        mask_schedule_name="linear",
        mask_schedule_start=0.0001,
        mask_schedule_end=0.02,
        selfies_mask_ratio=0.15,
        max_text_length=512,
        max_selfies_length=256,
        max_atoms=32,                # keep consistent with your YAML
        include_edge_bond_dist=True,
        include_rdmol2selfies=False,
        buffer_size=10,
        shuffle=True,
        repeat=False,
        rank=0,
        world_size=1,
        atom_mask_token_id=0,        # atom-type PAD/MASK = 0
    )

    dl = DataLoader(dataset, batch_size=32,
                    collate_fn=dataset.collate_fn, num_workers=4)

    print("Testing dataloader...")
    for i, batch in enumerate(dl):
        print(f"\nBatch {i+1}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}, dtype={v.dtype}")
            else:
                print(f"{k}: {type(v)}")
        if i >= 2:
            break
    print("Done.")
