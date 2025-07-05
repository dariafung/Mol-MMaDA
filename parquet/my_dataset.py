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
from transformers import AutoTokenizer # 用于示例和加载LLM tokenizer
from torchvision import transforms # 可能是原有文件中的遗留导入，如果不需要可以移除
import json
import numpy as np 
from rdkit import Chem 
from selfies import encoder, get_semantic_robust_alphabet

# 从 training.utils 导入 mask_or_random_replace_tokens 和 get_mask_schedule
# 确保这个路径正确
from training.utils import mask_or_random_replace_tokens, get_mask_schedule


def atom_to_id(symbol: str) -> int:
    """
    将原子符号映射到其原子序数 (Z)。
    如果符号无法识别，返回 0（或你定义的未知原子 ID）。
    """
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    except Exception:
        return 0 # 对于无法识别的原子符号，返回0

# --- 3D 数据解析函数（基于您最新提供的内容，并修正原子符号转换） ---
def parse_molecular_3d_data(raw_data_dict: Dict[str, Any]) -> Any:
    """
    解析并转换 3D 相关的原始数据为 PyTorch Tensor。
    raw_data_dict 应该包含从 Parquet 行读取的 'atom_vec_str', 'coordinates_str' 等序列化字符串。
    注意：这里的 atom_vec_str 预期是原子符号列表的JSON字符串，例如 "[\"C\", \"O\"]"。
    """
    try:
        # 从 JSON 字符串反序列化为 Python 列表
        atom_vec_symbols = json.loads(raw_data_dict.get('atom_vec_str', '[]'))
        coordinates_list = json.loads(raw_data_dict.get('coordinates_str', '[]'))

        # 将原子符号列表转换为原子ID列表
        atom_ids = [atom_to_id(symbol) for symbol in atom_vec_symbols]
        
        # 将数据转换为 Numpy 数组，再转换为 PyTorch Tensor
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


# --- 修正后的 MolecularUnifiedDataset 类 ---
class MolecularUnifiedDataset(IterableDataset):
    def __init__(self,
                 data_path: str, # 指向你的 Parquet 文件或包含 Parquet 文件的目录
                 tokenizer, # HuggingFace tokenizer 实例
                 mask_token_id: int, # 新增：掩码 token 的 ID
                 diffusion_timesteps: int,
                 mask_schedule_name: str, # 新增：掩码调度名称
                 mask_schedule_start: float, # 新增：掩码调度开始值
                 mask_schedule_end: float, # 新增：掩码调度结束值
                 selfies_mask_ratio: Optional[float] = None,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = True,
                 repeat: bool = True,
                 buffer_size: int = 100,
                 max_text_length: int = 512, # 文本最大长度
                 max_selfies_length: int = 256, # SELFIES 最大长度
                 max_atoms: int = 256, # 为原子数设置最大值，用于 padding 3D 数据
                 include_edge_bond_dist: bool = False, # 是否包含边类型、键类型和距离矩阵
                 include_rdmol2selfies: bool = False): # 是否包含rdmol2selfies
        super().__init__()
        
        self.data_path = data_path # 保存 data_path
        # 查找 Parquet 文件
        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        else:
            self.files = sorted(glob.glob(data_path)) # 允许直接传入文件路径模式

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

        # 掩码相关参数
        self.mask_token_id = mask_token_id
        self.diffusion_timesteps = diffusion_timesteps # 存储总的扩散步长

        # mask_schedule_fn 用于获取给定 timestep 的 mask ratio
        # 这里传入 diffusion_timesteps 作为 schedule 的总长度
        self.mask_schedule_values = get_mask_schedule(
            mask_schedule_name, # 例如 "linear", "cosine"
            timesteps=self.diffusion_timesteps, # 传递总的扩散步长
            start=mask_schedule_start,
            end=mask_schedule_end
        )
        self.selfies_mask_ratio = selfies_mask_ratio # 如果不是动态的，可以使用固定值

    def read_parquet_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """从 Parquet 文件读取所有列数据。"""
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            # 过滤掉 SELFIES 字符串为空的行，避免后续错误
            df = df[df['selfies_string'].notna() & (df['selfies_string'] != '')]
            for _, row in df.iterrows():
                yield row.to_dict() # 返回包含所有列的字典
        except Exception as e:
            print(f"Error reading parquet file {file_path}: {e}")
            # 可以选择跳过文件或记录错误

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        self_files_for_worker = self.files[self.rank::self.world_size]
        
        # 确保每个 worker 有一个独立的随机种子，以确保数据 shuffle 和 mask 的随机性
        # 使用 os.getpid() 增加随机性，避免同一机器上不同 worker 种子相同
        worker_seed = self.rank + worker_id + os.getpid()
        random.seed(worker_seed)
        np.random.seed(worker_seed) 

        while True:
            file_list_shuffled = list(self_files_for_worker) # 复制一份，防止 random.shuffle 修改原始列表
            if self.shuffle:
                random.shuffle(file_list_shuffled)

            for file in file_list_shuffled:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for raw_data_item in data_generator:
                    try:
                        # --- 1. 处理 SELFIES ---
                        selfies_str = raw_data_item.get("selfies_string", "")
                        if not selfies_str or not isinstance(selfies_str, str):
                            continue # 跳过空或非字符串 SELFIES 样本

                        # 原始的 SELFIES token ID (作为 true_selfies_labels 的来源)
                        selfies_tokenized_clean = self.tokenizer(
                            selfies_str,
                            truncation=True,
                            max_length=self.max_selfies_length,
                            padding="do_not_pad",
                            return_tensors="pt"
                        )
                        selfies_input_ids_clean = selfies_tokenized_clean.input_ids[0]
                        selfies_attention_mask = selfies_tokenized_clean.attention_mask[0]
                        
                        # --- 为 SELFIES 应用掩码 (L_Diff-disc) ---
                        # 随机采样一个整数 timestep (0 到 total_diffusion_timesteps - 1)
                        # 这个 timestep 也将传递给模型
                        timestep = torch.randint(0, self.diffusion_timesteps, (1,)).item()
                        
                        # 根据 timestep 从预计算的 schedule 中获取 mask ratio
                        # 注意：mask_schedule_values 应该是一个 tensor，直接索引即可
                        current_mask_ratio = self.mask_schedule_values[timestep].item()
                        
                        # 应用掩码和随机替换
                        masked_selfies_input_ids, true_selfies_labels_for_loss, _ = mask_or_random_replace_tokens(
                            selfies_input_ids_clean.unsqueeze(0), # mask_or_random_replace_tokens 期望批次维度
                            self.mask_token_id,
                            mask_ratio=current_mask_ratio, # 使用动态采样的掩码比例
                            tokenizer_vocab_size=self.tokenizer.vocab_size,
                            is_train=True # 假设在训练模式
                        )
                        selfies_input_ids = masked_selfies_input_ids.squeeze(0) # 移除批次维度
                        true_selfies_labels = true_selfies_labels_for_loss.squeeze(0) # 移除批次维度

                        # --- 2. 处理其他文本 (如果需要的话) ---
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

                        # --- 3. 处理 3D 数据 ---
                        processed_3d_data_tensors = parse_molecular_3d_data(raw_data_item)
                        if not processed_3d_data_tensors: # 检查是否是空字典
                            continue # 跳过 3D 数据解析失败的样本
                        
                        atom_vec = processed_3d_data_tensors.get("atom_vec")
                        coordinates = processed_3d_data_tensors.get("coordinates")
                        
                        if atom_vec is None or coordinates is None or atom_vec.shape[0] == 0:
                            continue # 跳过没有原子或坐标的样本

                        num_atoms = atom_vec.shape[0]
                        if num_atoms > self.max_atoms:
                            continue # 跳过原子数过多的分子
                        
                        # Padding 3D 数据到 max_atoms
                        padded_atom_vec = torch.full((self.max_atoms,), 0, dtype=torch.long) 
                        padded_atom_vec[:num_atoms] = atom_vec
                        
                        padded_coordinates = torch.zeros((self.max_atoms, 3), dtype=torch.float32)
                        padded_coordinates[:num_atoms, :] = coordinates
                        
                        atoms_mask = torch.zeros((self.max_atoms,), dtype=torch.bool)
                        atoms_mask[:num_atoms] = True

                        sample = {
                            "id": raw_data_item.get("id", str(random.randint(0, 1000000))),
                            "selfies_input_ids": selfies_input_ids,
                            "selfies_attention_mask": selfies_attention_mask,
                            "true_selfies_labels": true_selfies_labels,
                            "text_input_ids": text_input_ids,
                            "text_attention_mask": text_attention_mask,
                            "atom_vec": padded_atom_vec,
                            "coordinates": padded_coordinates,
                            "atoms_mask": atoms_mask,
                            "timesteps": torch.tensor([timestep], dtype=torch.long), # 传递 timestep
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
        """
        批处理函数，对文本和 SELFIES 序列进行填充，对 3D 数据进行堆叠。
        排除 'id' 字段以避免 accelerate 的拼接错误。
        """
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


        keys_to_stack = ["atom_vec", "coordinates", "atoms_mask", "timesteps"] # 添加 timesteps
        if self.include_edge_bond_dist:
            keys_to_stack.extend(["edge_type", "bond_type", "dist"])
        if self.include_rdmol2selfies:
            keys_to_stack.append("rdmol2selfies")

        for k in keys_to_stack:
            if k in batched_data and len(batched_data[k]) > 0:
                final_batch[k] = torch.stack(batched_data[k], dim=0)
            else:
                if k == "coordinates":
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, 3, dtype=torch.float32) # 使用 len(batch)
                elif k == "atoms_mask":
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, dtype=torch.bool)
                elif k == "timesteps": # 针对 timesteps 添加处理
                    final_batch[k] = torch.empty(len(batch), 1, dtype=torch.long) # timesteps 是 (batch_size, 1)
                elif k in ["edge_type", "bond_type", "dist"]:
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, self.max_atoms, dtype=torch.long if k != "dist" else torch.float32)
                elif k == "rdmol2selfies":
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, self.max_selfies_length, dtype=torch.float32)
                else: # atom_vec 或其他 1D 数组
                    final_batch[k] = torch.empty(len(batch), self.max_atoms, dtype=torch.long)

        return final_batch


if __name__ == '__main__':
    parquet_path = "/home/exouser/MMaDA/m3_molecular_data.parquet"
    
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
        diffusion_timesteps=1000, # 传递总扩散步长
        mask_schedule_name="linear", # 示例值
        mask_schedule_start=0.0001, # 示例值
        mask_schedule_end=0.02, # 示例值
        selfies_mask_ratio=0.15, # 示例值，现在这个值只在 mask_schedule_name 不是动态的情况下使用
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
        batch_size=2,
        collate_fn=dataset.collate_fn,
        num_workers=0
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