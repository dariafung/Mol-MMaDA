import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple, Union, Optional
import argparse
import sys 

def get_mask_schedule(schedule_name: str, timesteps: int, start: float, end: float, x=None):
    if schedule_name == "linear":
        return torch.linspace(start, end, timesteps, dtype=torch.float32)
    elif schedule_name == "cosine":
        if x is None: # If not providing a specific timestep, return the full schedule
            t = torch.arange(timesteps, dtype=torch.float32)
        else: # For a specific timestep 'x'
            t = x.float() # Ensure t is float for calculation

        f_t = torch.cos(((t / timesteps) + 0.008) / 1.008 * (math.pi / 2)) ** 2
        alphas_bar = f_t / f_t[0]
        # For masking, we often want the opposite: low mask at t=0, high mask at t=timesteps
        # So we might use 1 - alphas_bar for mask ratio, or a custom function.
        # Let's define it such that it gives mask ratio directly
        
        # Example for mask ratio: starts low, increases
        # Using 1 - sqrt(alphas_bar) can be a simple way to get a masking schedule
        mask_ratio_schedule = 1.0 - torch.sqrt(alphas_bar)
        
        # Scale to start and end
        scaled_mask_ratio = start + (end - start) * mask_ratio_schedule
        return scaled_mask_ratio.clamp(0.0, 1.0) # Ensure it's between 0 and 1
    else:
        raise ValueError(f"Unknown schedule name: {schedule_name}")

def apply_selfies_masking(
    original_selfies_ids: torch.LongTensor,
    mask_token_id: int,
    timestep: int, # 当前的扩散步长
    total_diffusion_timesteps: int, # 总的扩散步长，来自 MMadaConfig.diffusion_timesteps
    mask_schedule_name: str, # 例如 "linear", "cosine"
    mask_schedule_start: float, # 例如 0.0001
    mask_schedule_end: float, # 例如 0.02
    device: torch.device,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    根据扩散步长对 SELFIES token 进行 masking。
    
    Args:
        original_selfies_ids: 原始的 SELFIES token ID 序列 (batch_size, seq_len)。
        mask_token_id: SELFIES tokenizer 的 [MASK] token ID。
        timestep: 当前的扩散步长 (0 到 total_diffusion_timesteps - 1)。
        total_diffusion_timesteps: 总的扩散步长。
        mask_schedule_name: mask 比例的调度名称。
        mask_schedule_start: mask 调度起始值。
        mask_schedule_end: mask 调度结束值。
        device: tensor 所在的设备。

    Returns:
        masked_selfies_ids: 经过 masking 的 SELFIES token ID 序列。
        true_selfies_labels: 原始的 SELFIES token ID 序列 (作为ground truth)。
                              未被 mask 的位置通常用 -100 标记，以便在计算交叉熵时忽略。
    """
    batch_size, seq_len = original_selfies_ids.shape
    
    # 1. 获取当前 timestep 对应的 masking 概率

    
    # 映射 timestep 到 [0, 1] 范围，用于 schedule 计算
    normalized_timestep = timestep / (total_diffusion_timesteps - 1)
    
    # 模拟 get_mask_schedule 对单个 timestep 的行为
    if mask_schedule_name == "linear":
        mask_ratio = mask_schedule_start + (mask_schedule_end - mask_schedule_start) * normalized_timestep
    elif mask_schedule_name == "cosine":
        # 简化版：从 0 到 1 的归一化时间 t
        t_normalized = (timestep.float() + 1e-5) / total_diffusion_timesteps # Avoid division by zero
        # 假设一个从低到高变化的mask比例
        mask_ratio = mask_schedule_start + (mask_schedule_end - mask_schedule_start) * (1 - torch.cos(t_normalized * math.pi)) / 2
    else:
        # Fallback for other schedules or default to a fixed ratio
        mask_ratio = mask_schedule_start # Or a fixed self.config.mask_replace_ratio

    mask_ratio = torch.clamp(mask_ratio, 0.0, 1.0).item() # 确保在 [0, 1] 之间并转为标量


    # 2. 生成 mask
    masked_selfies_ids = original_selfies_ids.clone()
    true_selfies_labels = original_selfies_ids.clone() # 初始时，标签就是原始序列
    
    for i in range(batch_size):
        seq_len_i = (original_selfies_ids[i] != 0).sum().item() # 假设 0 是 padding token
        if seq_len_i == 0:
            continue
            
        num_mask_tokens = int(seq_len_i * mask_ratio)
        
        # 可mask的 token 索引
        non_special_token_indices = (original_selfies_ids[i] != 0).nonzero(as_tuple=True)[0]
        
        if len(non_special_token_indices) == 0:
            continue

        # 随机选择要 mask 的 token 索引
        mask_indices = random.sample(
            non_special_token_indices.tolist(), 
            min(num_mask_tokens, len(non_special_token_indices))
        )
        
        # 对选定的索引进行 mask
        masked_selfies_ids[i, mask_indices] = mask_token_id
        
        # 对于未被 mask 的 token，在 true_selfies_labels 中设置为 -100，以便在交叉熵计算时忽略
        unmasked_indices = list(set(non_special_token_indices.tolist()) - set(mask_indices))
        true_selfies_labels[i, unmasked_indices] = -100
        
        # 如果有 padding token，也设置为 -100
        padding_indices = (original_selfies_ids[i] == 0).nonzero(as_tuple=True)[0]
        true_selfies_labels[i, padding_indices] = -100

    return masked_selfies_ids.to(device), true_selfies_labels.to(device)

def get_noise_schedule(name: str, beta_start: float, beta_end: float, timesteps: int):
    """
    Returns a noise schedule for continuous diffusion.
    """
    if name == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    elif name == "cosine":
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(f"Noise schedule '{name}' not supported.")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Return a function that maps an integer timestep to alpha_bar_sqrt
    def schedule_fn(t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(alphas_cumprod[t])

    return schedule_fn

##################################################
#              config utils
##################################################
def get_config():
    parser = argparse.ArgumentParser(description="LLaDA training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file.")
    parser.add_argument("--local_rank", type=int, default=0) 
    
    args, unknown_args = parser.parse_known_args() 
    
    yaml_conf = OmegaConf.load(args.config)
    
    cli_overrides = OmegaConf.from_cli(unknown_args) 

    config = OmegaConf.merge(yaml_conf, cli_overrides)
    
    return config


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              training utils
##################################################
def soft_target_cross_entropy(logits, targets, soft_targets):
    # ignore the first token from logits and targets (class id token)
    logits = logits[:, 1:]
    targets = targets[:, 1:]

    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    padding_mask = targets.eq(-100)

    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    loss = loss.sum() / num_active_elements
    return loss


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]

def mask_or_random_replace_tokens(
    tokens: torch.Tensor, 
    mask_id: int, 
    mask_ratio: float, # 掩码的比例，浮点数，例如 0.15
    tokenizer_vocab_size: int, # 分词器的词汇表大小
    is_train: bool = True, 
    seed: Optional[int] = None # 用于确定性掩码的种子
):
    """
    对 token 序列进行掩码或随机替换操作，用于离散扩散。
    
    Args:
        tokens (torch.Tensor): 输入的 token ID 序列，形状 (batch_size, seq_len)。
        mask_id (int): 掩码 token 的 ID。
        mask_ratio (float): 每个 token 被掩码的概率。
        tokenizer_vocab_size (int): 分词器的总词汇量大小，用于随机替换。
        is_train (bool): 是否处于训练模式。影响随机性行为。
        seed (Optional[int]): 用于生成确定性掩码的种子（仅在非训练模式下使用）。
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        - input_ids: 掩码或随机替换后的 token ID 序列，作为模型输入。
        - labels: 原始的 token ID 序列，被掩码的位置保留原始 ID，未被掩码的位置为 -100 (用于交叉熵损失忽略)。
        - loss_weight: 损失权重（可选，如果需要根据掩码概率加权）。
    """
    batch_size, seq_len = tokens.shape

    if not is_train and seed is not None:
        # 保存当前随机状态
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        python_rng_state = random.getstate()
        
        # 设置固定种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed) # For numpy if used internally


    # 创建一个随机掩码
    # mask_ratio 是每个 token 被掩码的概率
    mask = torch.rand(batch_size, seq_len, device=tokens.device) < mask_ratio
    mask = mask.to(torch.bool)

    # input_ids 是模型实际看到的输入
    input_ids = torch.where(mask, mask_id, tokens)
    
    # labels 是损失函数的目标。
    # 只有被掩码的位置的原始 token ID 才被用来计算损失，其他位置设为 -100。
    labels = torch.where(mask, tokens, -100) # -100 是 PyTorch F.cross_entropy 的 ignore_index 默认值


    # loss_weight 可以根据 mask_ratio 和 mask 本身来定义，如果需要加权损失的话
    # 对于简单的掩码语言模型任务，通常不需要额外的 loss_weight
    loss_weight = None # For now, set to None as it's optional

    if not is_train and seed is not None:
        # 恢复随机状态
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        random.setstate(python_rng_state)
        # np.random.set_state(numpy_rng_state) # If numpy was used, restore its state

    return input_ids, labels, loss_weight


##################################################
#              misc
##################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

