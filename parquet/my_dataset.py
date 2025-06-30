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
        return None # 返回 None 表示解析失败，在 __iter__ 中跳过该样本


# --- 修正后的 MolecularUnifiedDataset 类 ---
class MolecularUnifiedDataset(IterableDataset):
    def __init__(self,
                 data_path: str, # 指向你的 Parquet 文件或包含 Parquet 文件的目录
                 tokenizer, # HuggingFace tokenizer 实例
                 mask_token_id: int, # 新增：掩码 token 的 ID
                 mask_schedule_name: str, # 新增：掩码调度名称
                 mask_schedule_start: float, # 新增：掩码调度开始值
                 mask_schedule_end: float, # 新增：掩码调度结束值
                 selfies_mask_ratio: float, # 新增：SELFIES 掩码的比例
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
        self.tokenizer = tokenizer # 接收 tokenizer 实例
        self.include_edge_bond_dist = include_edge_bond_dist
        self.include_rdmol2selfies = include_rdmol2selfies

        # 掩码相关参数
        self.mask_token_id = mask_token_id
        # 使用 get_mask_schedule 初始化掩码调度函数
        self.mask_schedule_fn = get_mask_schedule(
            mask_schedule_name, mask_schedule_start, mask_schedule_end, timesteps=1000 # 假设 timesteps 为 1000
        )
        self.selfies_mask_ratio = selfies_mask_ratio # selfies_mask_ratio 用于控制掩码量，即使有 schedule 也可以额外使用


        # 分配给当前进程的文件
        self.files = self.files[self.rank::self.world_size]

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
        # 获取当前 worker 信息，用于分布式数据加载
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # 每个 worker 负责处理其分配到的文件子集
        # 确保每个 worker 有一个独立的随机种子，以确保数据 shuffle 和 mask 的随机性
        worker_seed = self.rank + worker_id # 基于 rank 和 worker_id 生成种子
        random.seed(worker_seed)
        np.random.seed(worker_seed) # For numpy if used internally

        while True:
            file_list = self.files
            if self.shuffle:
                random.shuffle(file_list)

            for file in file_list: # worker 已经分配了文件，所以直接遍历 file_list
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
                        # 从调度器获取掩码概率，或者使用固定的 selfies_mask_ratio
                        # 这里使用随机采样的浮点时间步 t
                        t_float_for_mask = torch.rand(1).item() # 0到1之间的随机浮点数
                        # selfies_mask_ratio_for_sample = self.mask_schedule_fn(torch.tensor([t_float_for_mask])).item()
                        # 简化：直接使用配置的 selfies_mask_ratio 或 get_mask_schedule 返回的函数
                        selfies_mask_ratio_for_sample = self.selfies_mask_ratio 
                        
                        # 应用掩码和随机替换
                        masked_selfies_input_ids, true_selfies_labels_for_loss, _ = mask_or_random_replace_tokens(
                            selfies_input_ids_clean.unsqueeze(0), # mask_or_random_replace_tokens 期望批次维度
                            self.mask_token_id,
                            mask_ratio=selfies_mask_ratio_for_sample, # 使用采样的掩码比例
                            tokenizer_vocab_size=self.tokenizer.vocab_size,
                            is_train=True # 假设在训练模式
                        )
                        selfies_input_ids = masked_selfies_input_ids.squeeze(0) # 移除批次维度
                        true_selfies_labels = true_selfies_labels_for_loss.squeeze(0) # 移除批次维度

                        # --- 2. 处理其他文本 (如果需要的话) ---
                        text_str = raw_data_item.get("text_description", "")
                        if not text_str or not isinstance(text_str, str):
                            text_str = "" # 文本可以为空，但不跳过，将其置为空字符串

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
                        if processed_3d_data_tensors is None:
                            continue # 跳过 3D 数据解析失败的样本
                        
                        atom_vec = processed_3d_data_tensors.get("atom_vec")
                        coordinates = processed_3d_data_tensors.get("coordinates")
                        
                        if atom_vec is None or coordinates is None or atom_vec.shape[0] == 0:
                            # print(f"Warning: 3D data incomplete or empty for ID {raw_data_item.get('id', 'unknown')}, skipping.")
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
                            "id": raw_data_item.get("id", str(random.randint(0, 1000000))), # 确保有ID
                            "selfies_input_ids": selfies_input_ids,
                            "selfies_attention_mask": selfies_attention_mask,
                            "true_selfies_labels": true_selfies_labels, # 新增：真实 SELFIES 标签
                            "text_input_ids": text_input_ids,
                            "text_attention_mask": text_attention_mask,
                            "atom_vec": padded_atom_vec,
                            "coordinates": padded_coordinates,
                            "atoms_mask": atoms_mask,
                            "task_type": "1d_to_3d" # 明确任务类型
                        }

                        if self.include_edge_bond_dist:
                            # 确保这些张量不是None且形状正确
                            # 假设 edge_type, bond_type, dist 也是原子维度的
                            # num_atoms_original = processed_3d_data_tensors['atom_vec'].shape[0] # 获取原始原子数

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
                            # rdmol2selfies 是 [N_atoms, L_selfies_token] 格式，需要 N_atoms 和 L_selfies_token 都 padding
                            rdmol2selfies_current_shape = processed_3d_data_tensors['rdmol2selfies'].shape
                            # L_selfies_token 应该与 max_selfies_length 匹配
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
                if v is not None: # 排除 None 值，避免在 append 时出错
                    batched_data[k].append(v)
        
        final_batch = {}

        # --- 对文本和 SELFIES 序列进行填充 ---
        # selfies_input_ids 和 selfies_attention_mask
        if "selfies_input_ids" in batched_data and len(batched_data["selfies_input_ids"]) > 0:
            padded_selfies = self.tokenizer.pad(
                {"input_ids": batched_data["selfies_input_ids"], 
                 "attention_mask": batched_data["selfies_attention_mask"]},
                padding=True, 
                return_tensors="pt",
            )
            final_batch["selfies_input_ids"] = padded_selfies["input_ids"]
            final_batch["selfies_attention_mask"] = padded_selfies["attention_mask"]
        else: # 处理批次中没有 SELFIES 的情况，返回空张量
            final_batch["selfies_input_ids"] = torch.empty(0, dtype=torch.long)
            final_batch["selfies_attention_mask"] = torch.empty(0, dtype=torch.long)
        
        # true_selfies_labels
        if "true_selfies_labels" in batched_data and len(batched_data["true_selfies_labels"]) > 0:
            # For true_selfies_labels, pad with -100 (ignore_index for CE loss)
            # Find max length in batch for true_selfies_labels
            max_len_selfies_labels = max([s.size(0) for s in batched_data["true_selfies_labels"]])
            padded_true_selfies_labels = []
            for s in batched_data["true_selfies_labels"]:
                padding_len = max_len_selfies_labels - s.size(0)
                padded_s = F.pad(s, (0, padding_len), "constant", -100)
                padded_true_selfies_labels.append(padded_s)
            final_batch["true_selfies_labels"] = torch.stack(padded_true_selfies_labels, dim=0)
        else:
            final_batch["true_selfies_labels"] = torch.empty(0, dtype=torch.long)


        # text_input_ids 和 text_attention_mask
        if "text_input_ids" in batched_data and len(batched_data["text_input_ids"]) > 0:
            padded_text = self.tokenizer.pad(
                {"input_ids": batched_data["text_input_ids"], 
                 "attention_mask": batched_data["text_attention_mask"]},
                padding=True,
                return_tensors="pt",
            )
            final_batch["text_input_ids"] = padded_text["input_ids"]
            final_batch["text_attention_mask"] = padded_text["attention_mask"]
        else: # 处理批次中没有文本的情况，返回空张量
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
                    # 这里需要根据原始数据中的维度来确定，假设是 (max_atoms, max_atoms)
                    final_batch[k] = torch.empty(0, self.max_atoms, self.max_atoms, dtype=torch.long if k != "dist" else torch.float32)
                elif k == "rdmol2selfies":
                    final_batch[k] = torch.empty(0, self.max_atoms, self.max_selfies_length, dtype=torch.float32)
                else: # atom_vec 或其他 1D 数组
                    final_batch[k] = torch.empty(0, self.max_atoms, dtype=torch.long)

        # 任务类型通常在每个样本中是固定的，所以可以直接取第一个
        if "task_type" in batched_data and len(batched_data["task_type"]) > 0:
            final_batch["task_type"] = batched_data["task_type"][0]
        else:
            final_batch["task_type"] = "1d_to_3d" # 默认任务类型


        return final_batch


# --- 你可以在这里添加一个简单的测试，但在训练脚本中调用更合适 ---
if __name__ == '__main__':
    # 假设你已经生成了 m3_molecular_data_test.parquet
    parquet_path = "/home/exouser/MMaDA/m3_molecular_data.parquet"
    
    # 需要一个 tokenizer 来测试
    # llm_model_name_or_path 应该在你的配置文件中定义
    # 对于测试，我们可以使用一个通用的 BERT tokenizer，但实际训练应使用 LLaDA 的 tokenizer
    try:
        # 使用 LLaDA 的 tokenizer，但在此处作为示例，如果路径不可用，会尝试其他
        example_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct") 
    except Exception:
        print("Could not load LLaDA tokenizer, falling back to bert-base-uncased for testing.")
        example_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    # 扩展 tokenizer 词汇表以包含 SELFIES 符号
    # 仅当 tokenizer 没有这些 token 时才添加，并调整模型 embedding
    # 这部分通常在训练脚本中处理更合适
    current_vocab_size = len(example_tokenizer)
    new_selfies_tokens_set = set(get_semantic_robust_alphabet())
    existing_tokens = set(example_tokenizer.get_vocab().keys())
    tokens_to_add = list(new_selfies_tokens_set - existing_tokens)
    if tokens_to_add:
        example_tokenizer.add_tokens(tokens_to_add)
        # print(f"Added {len(tokens_to_add)} new SELFIES tokens. New vocab size: {len(example_tokenizer)}")


    # 假设 mask_token_id 存在于 tokenizer 中，或手动指定一个
    mask_token_id_test = example_tokenizer.mask_token_id
    if mask_token_id_test is None:
        print("Tokenizer does not have a mask token. Using a dummy mask_token_id=0 (or another unused ID).")
        # 实际训练中，需要确保 mask_token_id 是一个模型能理解的有效 ID
        mask_token_id_test = example_tokenizer.pad_token_id # Fallback to pad token or another unused ID


    dataset = MolecularUnifiedDataset(
        data_path=parquet_path,
        tokenizer=example_tokenizer,
        mask_token_id=mask_token_id_test,
        mask_schedule_name="linear", # 示例值
        mask_schedule_start=0.0001, # 示例值
        mask_schedule_end=0.02, # 示例值
        selfies_mask_ratio=0.15, # 示例值
        max_text_length=512,
        max_selfies_length=256,
        max_atoms=100, # 假设你的测试数据中分子原子数不超过100
        include_edge_bond_dist=True,
        include_rdmol2selfies=False,
        buffer_size=10,
        shuffle=True, # 测试时可以设置为 True 看看效果
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
                print(f"   {k}: {v.shape}, dtype={v.dtype}")
            elif isinstance(v, str): # task_type is string
                print(f"   {k}: {v}")
            else:
                print(f"   {k}: {type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
        if i >= 2: # 打印几个批次就停止
            break
    print("Data loading test complete.")