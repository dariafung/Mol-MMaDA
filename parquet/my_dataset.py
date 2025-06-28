import collections
import os
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import glob
from typing import List, Dict, Any, Optional, Iterator
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from torchvision import transforms
import json
# from PIL import Image
import numpy as np 
from rdkit import Chem 
from selfies import encoder, get_semantic_robust_alphabet

def atom_to_id(symbol: str) -> int:
    """
    将原子符号映射到其原子序数 (Z)。
    如果符号无法识别，返回 0（或你定义的未知原子 ID）。
    """
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    except Exception:
        return 0 # 对于无法识别的原子符号，返回0

# --- 新增的 3D 数据解析函数 ---
def parse_molecular_3d_data(raw_data_dict: Dict[str, Any]) -> Any:
    """
    解析并转换 3D 相关的原始数据为 PyTorch Tensor。
    raw_data_dict 应该包含从 Parquet 行读取的 'atom_vec_str', 'coordinates_str' 等序列化字符串。
    """
    try:
        # 从 JSON 字符串反序列化为 Python 列表/Numpy 数组
        # 使用 .get() 并提供默认空列表，以防某些键缺失
        atom_vec = np.array(json.loads(raw_data_dict.get('atom_vec_str', '[]')), dtype=np.int64)
        coordinates = np.array(json.loads(raw_data_dict.get('coordinates_str', '[]')), dtype=np.float32)
        
        # 将 Numpy 数组转换为 PyTorch Tensor
        atom_vec_tensor = torch.tensor(atom_vec, dtype=torch.long)
        coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)

        edge_type_tensor = None
        if 'edge_type_str' in raw_data_dict and raw_data_dict['edge_type_str']:
            edge_type_tensor = torch.tensor(json.loads(raw_data_dict['edge_type_str']), dtype=torch.long)
        
        bond_type_tensor = None
        if 'bond_type_str' in raw_data_dict and raw_data_dict['bond_type_str']:
            bond_type_tensor = torch.tensor(json.loads(raw_data_dict['bond_type_str']), dtype=torch.long)
        
        dist_tensor = None
        if 'dist_str' in raw_data_dict and raw_data_dict['dist_str']:
            dist_tensor = torch.tensor(json.loads(raw_data_dict['dist_str']), dtype=torch.float32)
        
        rdmol2selfies_tensor = None
        if 'rdmol2selfies_str' in raw_data_dict and raw_data_dict['rdmol2selfies_str']:
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
        print(f"Error parsing 3D data for ID {raw_data_dict.get('id', 'unknown')}: {e}")
        return None # 返回 None 表示解析失败，在 __iter__ 中跳过该样本

# --- 新增的 MolecularUnifiedDataset 类 ---
class MolecularUnifiedDataset(IterableDataset):
    def __init__(self,
                 data_path: str, # 指向你的 Parquet 文件或包含 Parquet 文件的目录
                 tokenizer, # HuggingFace tokenizer
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
        
        # 查找 Parquet 文件
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

        # 分配给当前进程的文件
        self.files = self.files[self.rank::self.world_size]

    def read_parquet_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """从 Parquet 文件读取所有列数据。"""
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            for _, row in df.iterrows():
                yield row.to_dict() # 返回包含所有列的字典
        except Exception as e:
            print(f"Error reading parquet file {file_path}: {e}")
            # 可以选择跳过文件或记录错误

    def __iter__(self):
        # 获取当前 worker 信息，用于分布式数据加载
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        while True:
            file_list = self.files
            if self.shuffle:
                random.shuffle(file_list)

            # 每个 worker 处理自己的文件子集
            current_worker_files = file_list[worker_id::num_workers]

            for file in current_worker_files:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for raw_data_item in data_generator:
                    try:
                        # --- 1. 处理 SELFIES ---
                        selfies_str = raw_data_item.get("selfies_string", "")
                        # 确保 SELFIES 字符串有效且不是空字符串
                        if not selfies_str or not isinstance(selfies_str, str): # 确保是字符串类型
                            continue # 跳过空或非字符串 SELFIES 样本

                        selfies_tokenized = self.tokenizer(
                            selfies_str,
                            truncation=True,
                            max_length=self.max_selfies_length,
                            padding="do_not_pad", # 在 collate_fn 中 padding
                            return_tensors="pt"
                        )
                        selfies_input_ids = selfies_tokenized.input_ids[0]
                        selfies_attention_mask = selfies_tokenized.attention_mask[0]

                        # --- 2. 处理其他文本 ---
                        text_str = raw_data_item.get("text_description", "")
                        # 确保文本字符串有效
                        if not text_str or not isinstance(text_str, str): # 确保是字符串类型
                            text_str = "" # 文本可以为空，但不跳过，将其置为空字符串

                        text_tokenized = self.tokenizer(
                            text_str,
                            truncation=True,
                            max_length=self.max_text_length,
                            padding="do_not_pad", # 在 collate_fn 中 padding
                            return_tensors="pt"
                        )
                        text_input_ids = text_tokenized.input_ids[0]
                        text_attention_mask = text_tokenized.attention_mask[0]

                        # --- 3. 处理 3D 数据 ---
                        # parse_molecular_3d_data 接收一个字典，返回一个字典包含 Tensor
                        processed_3d_data_tensors = parse_molecular_3d_data(raw_data_item)
                        if processed_3d_data_tensors is None: # 解析失败
                            continue # 跳过 3D 数据解析失败的样本
                        
                        atom_vec = processed_3d_data_tensors.get("atom_vec")
                        coordinates = processed_3d_data_tensors.get("coordinates")
                        
                        if atom_vec is None or coordinates is None or atom_vec.shape[0] == 0:
                            print(f"Warning: 3D data incomplete or empty for ID {raw_data_item.get('id', 'unknown')}, skipping.")
                            continue # 跳过没有原子或坐标的样本

                        num_atoms = atom_vec.shape[0]
                        if num_atoms > self.max_atoms:
                            # print(f"Skipping molecule with {num_atoms} atoms (max_atoms={self.max_atoms})")
                            continue # 跳过原子数过多的分子
                        
                        # Padding 3D 数据到 max_atoms
                        # atom_vec
                        padded_atom_vec = torch.full((self.max_atoms,), self.tokenizer.pad_token_id, dtype=torch.long)
                        padded_atom_vec[:num_atoms] = atom_vec
                        
                        # coordinates
                        padded_coordinates = torch.zeros((self.max_atoms, 3), dtype=torch.float32)
                        padded_coordinates[:num_atoms, :] = coordinates
                        
                        # 3D 数据的掩码 (指示哪些是真实原子，哪些是填充)
                        # 1 for real atoms, 0 for padded
                        atoms_mask = torch.zeros((self.max_atoms,), dtype=torch.bool)
                        atoms_mask[:num_atoms] = True

                        sample = {
                            "id": raw_data_item.get("id", str(random.randint(0, 1000000))), # 确保有ID
                            "selfies_input_ids": selfies_input_ids,
                            "selfies_attention_mask": selfies_attention_mask,
                            "text_input_ids": text_input_ids,
                            "text_attention_mask": text_attention_mask,
                            "atom_vec": padded_atom_vec,
                            "coordinates": padded_coordinates,
                            "atoms_mask": atoms_mask,
                        }

                        if self.include_edge_bond_dist:
                            # 确保这些张量不是None且形状正确
                            if processed_3d_data_tensors.get('edge_type') is not None and processed_3d_data_tensors['edge_type'].numel() > 0:
                                padded_edge_type = torch.zeros((self.max_atoms, self.max_atoms), dtype=torch.long)
                                padded_edge_type[:num_atoms, :num_atoms] = processed_3d_data_tensors['edge_type']
                                sample['edge_type'] = padded_edge_type
                            if processed_3d_data_tensors.get('bond_type') is not None and processed_3d_data_tensors['bond_type'].numel() > 0:
                                padded_bond_type = torch.zeros((self.max_atoms, self.max_atoms), dtype=torch.long)
                                padded_bond_type[:num_atoms, :num_atoms] = processed_3d_data_tensors['bond_type']
                                sample['bond_type'] = padded_bond_type
                            if processed_3d_data_tensors.get('dist') is not None and processed_3d_data_tensors['dist'].numel() > 0:
                                padded_dist = torch.zeros((self.max_atoms, self.max_atoms), dtype=torch.float32)
                                padded_dist[:num_atoms, :num_atoms] = processed_3d_data_tensors['dist']
                                sample['dist'] = padded_dist
                        
                        if self.include_rdmol2selfies and processed_3d_data_tensors.get('rdmol2selfies') is not None and processed_3d_data_tensors['rdmol2selfies'].numel() > 0:
                            # rdmol2selfies 是 [N, L] 格式，需要 N 和 L 都 padding
                            max_selfies_seq_length = self.max_selfies_length # L 是 SELFIES token 的最大长度
                            padded_rdmol2selfies = torch.zeros((self.max_atoms, max_selfies_seq_length), dtype=torch.float32) # 根据 rdmol2selfies 的具体 dtype 调整
                            
                            copy_atoms = min(num_atoms, self.max_atoms)
                            copy_selfies_len = min(processed_3d_data_tensors['rdmol2selfies'].shape[1], max_selfies_seq_length)

                            padded_rdmol2selfies[:copy_atoms, :copy_selfies_len] = \
                                processed_3d_data_tensors['rdmol2selfies'][:copy_atoms, :copy_selfies_len]
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
                        continue # 跳过错误样本

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item

            if not self.repeat:
                break # 如果不重复，在所有文件处理完后退出循环

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


        # --- 对 3D 相关的固定尺寸张量进行堆叠 ---
        keys_to_stack = ["atom_vec", "coordinates", "atoms_mask"]
        if self.include_edge_bond_dist:
            keys_to_stack.extend(["edge_type", "bond_type", "dist"])
        if self.include_rdmol2selfies:
            keys_to_stack.append("rdmol2selfies")

        for k in keys_to_stack:
            if k in batched_data and len(batched_data[k]) > 0:
                final_batch[k] = torch.stack(batched_data[k], dim=0)
            else:
                # 返回形状合适的空张量，避免后续模型输入问题
                if k == "coordinates":
                    final_batch[k] = torch.empty(0, self.max_atoms, 3, dtype=torch.float32)
                elif k == "atoms_mask":
                    final_batch[k] = torch.empty(0, self.max_atoms, dtype=torch.bool)
                elif k in ["edge_type", "bond_type", "dist"]:
                    final_batch[k] = torch.empty(0, self.max_atoms, self.max_atoms, dtype=torch.long if k != "dist" else torch.float32)
                elif k == "rdmol2selfies":
                    final_batch[k] = torch.empty(0, self.max_atoms, self.max_selfies_length, dtype=torch.float32)
                else: # atom_vec 或其他 1D 数组
                    final_batch[k] = torch.empty(0, self.max_atoms, dtype=torch.long)

        # --- 不再将 'id' 字段添加到 final_batch 中 ---
        # 如果训练过程中需要访问 id，则需要修改 prepare_molecular_inputs_and_labels 函数，
        # 让它从 batch 中直接获取，而不是从 collate_fn 的 final_batch 中。
        # 但通常 id 字段只用于记录，不用于模型输入，所以暂时移除是安全的。

        return final_batch


# --- 你可以在这里添加一个简单的测试，但在训练脚本中调用更合适 ---
if __name__ == '__main__':
    # 假设你已经生成了 m3_molecular_data_test.parquet
    parquet_path = "/home/exouser/MMaDA/m3_molecular_data.parquet"
    
    # 需要一个 tokenizer 来测试
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # 示例 tokenizer
    
    # 扩展 tokenizer 词汇表以包含 SELFIES 符号
    new_selfies_tokens = list(get_semantic_robust_alphabet())
    tokenizer.add_tokens(new_selfies_tokens)

    dataset = MolecularUnifiedDataset(
        data_path=parquet_path,
        tokenizer=tokenizer,
        max_text_length=512,
        max_selfies_length=256,
        max_atoms=100, # 假设你的测试数据中分子原子数不超过100
        include_edge_bond_dist=True,
        include_rdmol2selfies=False,
        buffer_size=10,
        shuffle=False,
        repeat=False,
        rank=0,
        world_size=1
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=2, # 批处理大小
        collate_fn=dataset.collate_fn,
        num_workers=0 # 测试时设置为0，避免多进程问题
    )
    
    print("Starting data loading test...")
    for i, batch in enumerate(train_dataloader):
        print(f"\nBatch {i+1} loaded.")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)}, len={len(v)}")
        if i >= 2: # 打印几个批次就停止
            break
    print("Data loading test complete.")

""" class RefinedWebDataset(IterableDataset):
    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=True,
                 repeat=True,
                 buffer_size=1000,
                 max_length=8000,
                 num_workers=1):
        super().__init__()
        self.files = sorted(glob.glob(data_path))  
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.files = self.files[self.rank::self.world_size]

    def read_parquet_file(self, file_path):
        table = pq.read_table(file_path, columns=["content"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            yield {"content": row["content"]}

    def __iter__(self):
        while True:  
            file_list = self.files
            if self.shuffle:
                random.shuffle(file_list)  

            for file in file_list:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for data in data_generator:
                    text = data["content"].replace("\n", "")
                    if len(text) > self.max_length:
                        start_index = random.randint(0, len(text) - self.max_length - 1)
                        selected_text = text[start_index:start_index + self.max_length]
                    else:
                        selected_text = text

                    buffer.append({"input_ids": selected_text})

                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item

            if not self.repeat:
                break  

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

class ChatDataset(IterableDataset):
    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=True,
                 repeat=True,
                 buffer_size=1000,
                 max_length=8000,
                 num_workers=1,
                 tokenizer=None):
        super().__init__()
        self.files = sorted(glob.glob(data_path))  
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.files = self.files[self.rank::self.world_size]

    def read_parquet_file(self, file_path):
        table = pq.read_table(file_path, columns=["content"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            yield {"content": row["content"]}

    def __iter__(self):
        while True:  
            file_list = self.files
            if self.shuffle:
                random.shuffle(file_list)  

            for file in file_list:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for data in data_generator:
                    text = data["content"]
                    if  self.tokenizer is None:
                        if len(text) > self.max_length:
                            start_index = random.randint(0, len(text) - self.max_length - 1)
                            selected_text = text[start_index:start_index + self.max_length]
                        else:
                            selected_text = text
                    else:
                        if len(self.tokenizer(text)['input_ids']) < self.max_length:
                            selected_text = text
                        else:
                            continue

                    buffer.append({"input_ids": selected_text})

                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item

            if not self.repeat:
                break  

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

class R2iDataset(IterableDataset):
    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=True,
                 repeat=True,
                 buffer_size=1000,
                 max_length=8000,
                 num_workers=1,
                 resolution=256,
                 tokenizer=None):
        super().__init__()
        self.data_path = data_path  
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.resolution = resolution

    def __iter__(self):
        while True:  
            subdirs = sorted([d for d in glob.glob(os.path.join(self.data_path, "*")) if os.path.isdir(d)])
            
            if self.shuffle:
                random.shuffle(subdirs)  
            
            subdirs = subdirs[self.rank::self.world_size]

            subdirs = ['/data_storage/lbw/datasets/laion-aesthetics-12m-images-2/00000']
            
            for subdir in subdirs:
                all_files = glob.glob(os.path.join(subdir, "*.*"))
                base_names = set()
                
                for file_path in all_files:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    base_names.add(base_name)
                
                base_names = list(base_names)
                if self.shuffle:
                    random.shuffle(base_names)
                
                buffer = []
                
                for base_name in base_names:
                    jpg_path = os.path.join(subdir, f"{base_name}.jpg")
                    caption_path = os.path.join(subdir, f"{base_name}.caption")
                    shortcaption_path = os.path.join(subdir, f"{base_name}.shortcaption")
                    
                    if not os.path.exists(jpg_path):
                        continue
                    
                    try:
                        image = Image.open(jpg_path).convert("RGB")
                        
                        caption = ""
                        if os.path.exists(caption_path):
                            with open(caption_path, "r", encoding="utf-8") as f:
                                caption = f.read().strip()
                        
                        short_caption = ""
                        if os.path.exists(shortcaption_path):
                            with open(shortcaption_path, "r", encoding="utf-8") as f:
                                short_caption = f.read().strip()
                        
                        transformed_image = image_transform_clip({"images": image}, resolution=self.resolution)["images"]
                        
                        if self.tokenizer is not None:
                            if len(self.tokenizer(caption)['input_ids']) > self.max_length - 2:
                                continue
                        
                        prompt = (
                            '<|start_header_id|>user<|end_header_id|>\n'
                            "You should first think out a more detailed version of the description and then provide the user with the image. The detailed description is enclosed within <think> </think> tags, i.e. <think> detailed description here </think> image here\n"
                            f"{short_caption}"
                            '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
                            f"<think>{caption}</think>"
                        )

                        sample = {
                            "images": transformed_image,
                            "input_ids": prompt,
                        }
                        
                        buffer.append(sample)
                        
                        if len(buffer) >= self.buffer_size:
                            if self.shuffle:
                                random.shuffle(buffer)
                            for item in buffer:
                                yield item
                            buffer = []
                    
                    except Exception as e:
                        print(f"Error processing {jpg_path}: {e}")
                        continue
                
                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item
            
            if not self.repeat:
                break  

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

class VQADataset(IterableDataset):
    def __init__(self,
                 json_path: str,
                 image_root: str,
                 tokenizer = None,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = True,
                 repeat: bool = True,
                 buffer_size: int = 100,
                 resolution: int = 256,
                 max_length: int = 8000,
                 num_workers: int = 1,
                 image_transform_method: str = "squash"):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.resolution = resolution
        self.max_length = max_length 
        self.num_workers = num_workers
        self.image_transform_method = image_transform_method
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.json_path}")
            self.list_data_dict = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.json_path}")
            self.list_data_dict = []
        else:
            self.list_data_dict = [item for item in raw_data if 'image' in item and 'conversations' in item]
        self.list_data_dict = self.list_data_dict[self.rank::self.world_size]
    def __iter__(self):
        sot_token = '<|startoftext|>'
        assistant_prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
        while True:
            current_data_list = list(self.list_data_dict) 
            if self.shuffle:
                random.shuffle(current_data_list)
            buffer = []
            for item in current_data_list:
                image_relative_path = item.get('image')
                conversations = item.get('conversations', [])
                if not image_relative_path or not conversations or len(conversations) < 2:
                    continue
                num_total_messages = len(conversations)
                if num_total_messages % 2 != 0:
                     conversations = conversations[:-1]
                     num_total_messages -= 1
                     if num_total_messages < 2: continue 
                num_turns = num_total_messages // 2
                if num_turns == 0:
                    continue
                selected_num_turns = random.randint(1, num_turns)
                selected_conversations = conversations[:selected_num_turns * 2]
                image_path = os.path.join(self.image_root, image_relative_path)
                try:
                    image = Image.open(image_path).convert("RGB")
                    if self.image_transform_method == "squash":
                        transformed_image = image_transform_squash({"images": image}, resolution=self.resolution)["images"]
                    elif self.image_transform_method == "pad":
                        transformed_image = image_transform_pad({"images": image}, resolution=self.resolution)["images"]
                    else:
                        transformed_image = image_transform_clip({"images": image}, resolution=self.resolution)["images"]
                    first_human_message = selected_conversations[0]['value']
                    processed_message = first_human_message.replace('<image>\n', '').replace('\n<image>', '')
                    current_selection_messages = list(selected_conversations)
                    current_selection_messages[0] = dict(current_selection_messages[0]) 
                    current_selection_messages[0]['value'] = processed_message
                    messages = []
                    for turn in current_selection_messages:
                        role = "user" if turn["from"] == "human" else "assistant"
                        messages.append({"role": role, "content": turn["value"]})
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True 
                    )
                    if formatted_text.startswith(sot_token):
                         formatted_text = formatted_text[len(sot_token):]
                    if formatted_text.endswith(assistant_prompt_suffix):
                        formatted_text = formatted_text[:-len(assistant_prompt_suffix)]
                    token_ids = self.tokenizer(formatted_text)['input_ids'] 
                    if len(token_ids) > self.max_length:
                        continue 
                    sample = {
                        "images": transformed_image,
                        "input_ids": formatted_text,
                    }
                    buffer.append(sample)
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for buf_item in buffer:
                            yield buf_item
                        buffer = []
                except FileNotFoundError:
                    print(f"Warning: Image file not found at {image_path}, skipping item.")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing item with image {image_path}: {e}, skipping.")
                    continue
            if buffer:
                if self.shuffle:
                    random.shuffle(buffer)
                for buf_item in buffer:
                    yield buf_item
            if not self.repeat:
                break
    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)
        return batched

def image_transform_clip(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    sample["images"] = image
    return sample

def image_transform_squash(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(image)
    sample["images"] = image
    return sample

def image_transform_pad(sample, resolution=256, fill_color=(255, 255, 255)):
    image = sample["images"]
    w, h = image.size
    if w == h:
        padded_image = image
    elif w < h:
        padding_needed = h - w
        padding_left = padding_needed // 2
        padding_right = padding_needed - padding_left
        pad_transform = transforms.Pad((padding_left, 0, padding_right, 0), fill=fill_color, padding_mode='constant')
        padded_image = pad_transform(image)
    else:
        padding_needed = w - h
        padding_top = padding_needed // 2
        padding_bottom = padding_needed - padding_top
        pad_transform = transforms.Pad((0, padding_top, 0, padding_bottom), fill=fill_color, padding_mode='constant')
        padded_image = pad_transform(image)
    image_resized = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC)(padded_image)
    image_tensor = transforms.ToTensor()(image_resized)
    image_normalized = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)
    sample["images"] = image_normalized
    return sample

if __name__ == '__main__':
    data_path = "/data_storage/shared/datasets/falcon-refinedweb/data/data/*.parquet"
    dataset = RefinedWebDataset(
        data_path=data_path,
        max_length=8000,
        buffer_size=0,
    )
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset, 
        batch_size=1,
        sampler=None, 
        collate_fn=dataset.collate_fn,
        num_workers=0
    )
    
    print("Starting data loading test...")
    for i, batch in enumerate(train_dataloader):
        if i == 0:
            print(batch)
            print(f"Batch size: {len(batch['input_ids'])}")
            print(f"First sample length: {len(batch['input_ids'][0])}")
        if i >= 5:
            break
    print("Data loading test complete") """