import collections
import os
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import glob
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Iterator
import pyarrow.parquet as pq
from transformers import AutoTokenizer 
import json
import numpy as np 
from rdkit import Chem 
from selfies import encoder, get_semantic_robust_alphabet

from training.utils import mask_or_random_replace_tokens, get_mask_schedule


def atom_to_id(symbol: str) -> int:
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    except Exception:
        return 0 

def parse_molecular_3d_data(raw_data_dict: Dict[str, Any]) -> Any:
    try:
        atom_raw = json.loads(raw_data_dict.get("atom_vec_str", "[]"))
        coordinates_list = json.loads(raw_data_dict.get("coordinates_str", "[]"))

        atom_ids = [
            int(x) if isinstance(x, int) or str(x).isdigit() else atom_to_id(str(x))
            for x in atom_raw
        ]
        
        atom_vec_tensor = torch.tensor(atom_ids, dtype=torch.long)
        coordinates_tensor = torch.tensor(coordinates_list, dtype=torch.float32)

        edge_type_tensor = None
        if raw_data_dict.get('edge_type_str'):
            edge_type_tensor = torch.tensor(json.loads(raw_data_dict['edge_type_str']), dtype=torch.long)
        
        bond_type_tensor = None
        if raw_data_dict.get('bond_type_str'):
            bond_type_tensor = torch.tensor(json.loads(raw_data_dict['bond_type_str']), dtype=torch.long)
        
        dist_tensor = None
        if raw_data_dict.get('dist_str'):
            dist_tensor = torch.tensor(json.loads(raw_data_dict['dist_str']), dtype=torch.float32)
        
        rdmol2selfies_tensor = None
        if raw_data_dict.get('rdmol2selfies_str'):
            rdmol2selfies_tensor = torch.tensor(json.loads(raw_data_dict['rdmol2selfies_str']), dtype=torch.float32)

        return {
            "atom_vec": atom_vec_tensor,
            "coordinates": coordinates_tensor,
            "edge_type": edge_type_tensor,
            "bond_type": bond_type_tensor,
            "dist": dist_tensor,
            "rdmol2selfies": rdmol2selfies_tensor
        }

    except Exception as e:
        # logging.error(f"Error parsing 3D data for ID {raw_data_dict.get('id', 'unknown')}: {e}")
        # print(f"Error parsing 3D data for ID {raw_data_dict.get('id', 'unknown')}: {e}") # Debugging
        return {}


class MolecularUnifiedDataset(IterableDataset):
    def __init__(self,
                 data_path: str, 
                 tokenizer, 
                 mask_token_id: int, 
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
                 include_rdmol2selfies: bool = False): 
        super().__init__()
        
        self.data_path = data_path 
        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
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

        self.mask_token_id = mask_token_id
        self.diffusion_timesteps = diffusion_timesteps 

        self.mask_schedule_values = get_mask_schedule(
            mask_schedule_name, 
            timesteps=self.diffusion_timesteps, 
            start=mask_schedule_start,
            end=mask_schedule_end
        )
        self.selfies_mask_ratio = selfies_mask_ratio 
        self.atom_type_mask_prob = atom_type_mask_prob

    def read_parquet_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """从 Parquet 文件读取所有列数据。"""
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            df = df[df['selfies_string'].notna() & (df['selfies_string'] != '')]
            for _, row in df.iterrows():
                yield row.to_dict() 
        except Exception as e:
            print(f"Error reading parquet file {file_path}: {e}")


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        self_files_for_worker = self.files[self.rank::self.world_size]
        
        worker_seed = self.rank + worker_id + os.getpid()
        random.seed(worker_seed)
        np.random.seed(worker_seed) 

        while True:
            file_list_shuffled = list(self_files_for_worker)
            if self.shuffle:
                random.shuffle(file_list_shuffled)

            for file in file_list_shuffled:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for raw_data_item in data_generator:
                    try:
                        selfies_str = raw_data_item.get("selfies_string", "")
                        if not selfies_str or not isinstance(selfies_str, str):
                            continue 

                        selfies_tokenized_clean = self.tokenizer(
                            selfies_str,
                            truncation=True,
                            max_length=self.max_selfies_length,
                            padding="do_not_pad",
                            return_tensors="pt"
                        )
                        selfies_input_ids_clean = selfies_tokenized_clean.input_ids[0]
                        selfies_attention_mask = selfies_tokenized_clean.attention_mask[0]
                        
                        timestep = torch.randint(0, self.diffusion_timesteps, (1,)).item()
                        
                        current_mask_ratio = self.mask_schedule_values[timestep].item()
                        
                        masked_selfies_input_ids, true_selfies_labels_for_loss, _ = mask_or_random_replace_tokens(
                            selfies_input_ids_clean.unsqueeze(0),
                            self.mask_token_id,
                            mask_ratio=current_mask_ratio, 
                            tokenizer_vocab_size=self.tokenizer.vocab_size,
                            is_train=True 
                        )
                        selfies_input_ids = masked_selfies_input_ids.squeeze(0) 
                        true_selfies_labels = true_selfies_labels_for_loss.squeeze(0) 

                        text_str = raw_data_item.get("text_description", "")
                        if not text_str or not isinstance(text_str, str):
                            text_str = "" 

                        text_tokenized = self.tokenizer(
                            text_str,
                            truncation=True,
                            max_length=self.max_text_length,
                            padding="do_not_pad",
                            return_tensors="pt"
                        )
                        text_input_ids = text_tokenized.input_ids[0]
                        text_attention_mask = text_tokenized.attention_mask[0]

                        processed_3d_data_tensors = parse_molecular_3d_data(raw_data_item)
                        if not processed_3d_data_tensors: 
                            continue 
                        
                        atom_vec = processed_3d_data_tensors.get("atom_vec")
                        coordinates = processed_3d_data_tensors.get("coordinates")
                        
                        if atom_vec is None or coordinates is None or atom_vec.shape[0] == 0:
                            continue 

                        num_atoms = atom_vec.shape[0]
                        if num_atoms > self.max_atoms:
                            continue 
                        
                        padded_atom_vec = torch.full((self.max_atoms,), 0, dtype=torch.long) 
                        padded_atom_vec[:num_atoms] = atom_vec
                        
                        padded_coordinates = torch.zeros((self.max_atoms, 3), dtype=torch.float32)
                        padded_coordinates[:num_atoms, :] = coordinates
                        
                        atoms_mask = torch.zeros((self.max_atoms,), dtype=torch.bool)
                        atoms_mask[:num_atoms] = True
                        masked_atom_vec = padded_atom_vec.clone()

                        if self.atom_type_mask_prob > 0:
                            prob_mask = torch.rand_like(masked_atom_vec, dtype=torch.float32) < self.atom_type_mask_prob
                            final_mask = prob_mask & atoms_mask
                            
                            masked_atom_vec[final_mask] = 0

                            if (masked_atom_vec != 0).sum() == 0:
                                real_idxs = (atoms_mask != 0).nonzero(as_tuple=True)[0]
                                rand_idx = real_idxs[ torch.randint(len(real_idxs), (1,)).item() ]
                                masked_atom_vec[rand_idx] = padded_atom_vec[rand_idx]

                        sample = {
                            "id": raw_data_item.get("id", str(random.randint(0, 1000000))),
                            "selfies_input_ids": selfies_input_ids,
                            "selfies_attention_mask": selfies_attention_mask,
                            "true_selfies_labels": true_selfies_labels,
                            "text_input_ids": text_input_ids,
                            "text_attention_mask": text_attention_mask,
                            "atom_vec": masked_atom_vec,          
                            "true_atom_vec": padded_atom_vec,     
                            
                            "coordinates": padded_coordinates,    
                            "true_coordinates": padded_coordinates.clone(), 
                            
                            "atoms_mask": atoms_mask,
                            "timesteps": torch.tensor([timestep], dtype=torch.long), 
                        }

                        if self.include_edge_bond_dist:
                            if processed_3d_data_tensors.get('edge_type') is not None and processed_3d_data_tensors['edge_type'].numel() > 0:
                                current_edge_shape = processed_3d_data_tensors['edge_type'].shape
                                padded_edge_type = torch.zeros((self.max_atoms, self.max_atoms), dtype=torch.long)
                                padded_edge_type[:current_edge_shape[0], :current_edge_shape[1]] = processed_3d_data_tensors['edge_type']
                                sample['edge_type'] = padded_edge_type

                            if processed_3d_data_tensors.get('bond_type') is not None and processed_3d_data_tensors['bond_type'].numel() > 0:
                                current_bond_shape = processed_3d_data_tensors['bond_type'].shape
                                padded_bond_type = torch.zeros((self.max_atoms, self.max_atoms), dtype=torch.long)
                                padded_bond_type[:current_bond_shape[0], :current_bond_shape[1]] = processed_3d_data_tensors['bond_type']
                                sample['bond_type'] = padded_bond_type

                            if processed_3d_data_tensors.get('dist') is not None and processed_3d_data_tensors['dist'].numel() > 0:
                                current_dist_shape = processed_3d_data_tensors['dist'].shape
                                padded_dist = torch.zeros((self.max_atoms, self.max_atoms), dtype=torch.float32)
                                padded_dist[:current_dist_shape[0], :current_dist_shape[1]] = processed_3d_data_tensors['dist']
                                sample['dist'] = padded_dist
                            
                        if self.include_rdmol2selfies and processed_3d_data_tensors.get('rdmol2selfies') is not None and processed_3d_data_tensors['rdmol2selfies'].numel() > 0:
                            rdmol2selfies_current_shape = processed_3d_data_tensors['rdmol2selfies'].shape
                            padded_rdmol2selfies = torch.zeros((self.max_atoms, self.max_selfies_length), dtype=torch.float32) 
                            
                            copy_atoms_dim = min(rdmol2selfies_current_shape[0], self.max_atoms)
                            copy_selfies_token_dim = min(rdmol2selfies_current_shape[1], self.max_selfies_length)

                            padded_rdmol2selfies[:copy_atoms_dim, :copy_selfies_token_dim] = \
                                processed_3d_data_tensors['rdmol2selfies'][:copy_atoms_dim, :copy_selfies_token_dim]
                            sample['rdmol2selfies'] = padded_rdmol2selfies


                        buffer.append(sample)

                        if len(buffer) >= self.buffer_size:
                            if self.shuffle:
                                random.shuffle(buffer)
                            for item in buffer:
                                yield item
                            buffer = []

                    except Exception as e:
                        print(f"Error processing sample ID {raw_data_item.get('id', 'unknown')}: {e}")
                        continue

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item

            if not self.repeat:
                break

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched_data = collections.defaultdict(list)
        for item in batch:
            for k, v in item.items():
                if v is not None:
                    batched_data[k].append(v)
        
        final_batch = {}

        # --- 对文本和 SELFIES 序列进行填充 ---
        if "selfies_input_ids" in batched_data and len(batched_data["selfies_input_ids"]) > 0:
            padded_selfies = self.tokenizer.pad(
                {"input_ids": batched_data["selfies_input_ids"], 
                 "attention_mask": batched_data["selfies_attention_mask"]},
                padding=True, 
                return_tensors="pt",
            )
            final_batch["selfies_input_ids"] = padded_selfies["input_ids"]
            final_batch["selfies_attention_mask"] = padded_selfies["attention_mask"]
        else:
            final_batch["selfies_input_ids"] = torch.empty(0, dtype=torch.long)
            final_batch["selfies_attention_mask"] = torch.empty(0, dtype=torch.long)
        
        if "true_selfies_labels" in batched_data and len(batched_data["true_selfies_labels"]) > 0:
            max_len_selfies_labels = max([s.size(0) for s in batched_data["true_selfies_labels"]])
            padded_true_selfies_labels = []
            for s in batched_data["true_selfies_labels"]:
                padding_len = max_len_selfies_labels - s.size(0)
                padded_s = F.pad(s, (0, padding_len), "constant", -100)
                padded_true_selfies_labels.append(padded_s)
            final_batch["true_selfies_labels"] = torch.stack(padded_true_selfies_labels, dim=0)
        else:
            final_batch["true_selfies_labels"] = torch.empty(0, dtype=torch.long)


        if "text_input_ids" in batched_data and len(batched_data["text_input_ids"]) > 0:
            padded_text = self.tokenizer.pad(
                {"input_ids": batched_data["text_input_ids"], 
                 "attention_mask": batched_data["text_attention_mask"]},
                padding=True,
                return_tensors="pt",
            )
            final_batch["text_input_ids"] = padded_text["input_ids"]
            final_batch["text_attention_mask"] = padded_text["attention_mask"]
        else:
            final_batch["text_input_ids"] = torch.empty(0, dtype=torch.long)
            final_batch["text_attention_mask"] = torch.empty(0, dtype=torch.long)


        keys_to_stack = ["atom_vec", "coordinates", "atoms_mask", "timesteps", "true_atom_vec", "true_coordinates"] 
        if self.include_edge_bond_dist:
            keys_to_stack.extend(["edge_type", "bond_type", "dist"])
        if self.include_rdmol2selfies:
            keys_to_stack.append("rdmol2selfies")

        for k in keys_to_stack:
            if k in batched_data and len(batched_data[k]) > 0:
                final_batch[k] = torch.stack(batched_data[k], dim=0)
            else:
                if k in ["coordinates", "true_coordinates"]:
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, 3, dtype=torch.float32)
                elif k == "atoms_mask":
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, dtype=torch.bool)
                elif k == "timesteps":
                    final_batch[k] = torch.empty(len(batch), 1, dtype=torch.long)
                elif k in ["edge_type", "bond_type"]:
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, self.max_atoms, dtype=torch.long)
                elif k == "dist":
                     final_batch[k] = torch.empty(len(batch), self.max_atoms, self.max_atoms, dtype=torch.float32)
                elif k == "rdmol2selfies":
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, self.max_selfies_length, dtype=torch.float32)
                else: 
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, dtype=torch.long)

        return final_batch


if __name__ == '__main__':
    parquet_path = "/projects/bezp/yfeng7/data/m3_molecular_data.parquet"
    
    try:
        example_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct") 
    except Exception:
        print("Could not load LLaDA tokenizer, falling back to bert-base-uncased for testing.")
        example_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    current_vocab_size = len(example_tokenizer)
    new_selfies_tokens_set = set(get_semantic_robust_alphabet())
    existing_tokens = set(example_tokenizer.get_vocab().keys())
    tokens_to_add = list(new_selfies_tokens_set - existing_tokens)
    if tokens_to_add:
        example_tokenizer.add_tokens(tokens_to_add)

    mask_token_id_test = example_tokenizer.mask_token_id
    if mask_token_id_test is None:
        print("Tokenizer does not have a mask token. Using a dummy mask_token_id=0 (or another unused ID).")
        mask_token_id_test = example_tokenizer.pad_token_id 


    dataset = MolecularUnifiedDataset(
        data_path=parquet_path,
        tokenizer=example_tokenizer,
        mask_token_id=mask_token_id_test,
        diffusion_timesteps=1000, 
        mask_schedule_name="linear", 
        mask_schedule_start=0.0001, 
        mask_schedule_end=0.02, 
        selfies_mask_ratio=0.15, 
        max_text_length=512,
        max_selfies_length=256,
        max_atoms=100,
        include_edge_bond_dist=True,
        include_rdmol2selfies=False,
        buffer_size=10,
        shuffle=True,
        repeat=False,
        rank=0,
        world_size=1
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=dataset.collate_fn,
        num_workers=8
    )
    
    print("Starting data loading test...")
    for i, batch in enumerate(train_dataloader):
        print(f"\nBatch {i+1} loaded.")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"   {k}: {v.shape}, dtype={v.dtype}")
            elif isinstance(v, str):
                print(f"   {k}: {v}")
            else:
                print(f"   {k}: {type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
        if i >= 2:
            break
    print("Data loading test complete.")