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

def get_mask_schedule(name: str, start: float, end: float, timesteps: int):
    """
    Returns a masking schedule function based on the given name and parameters.
    This schedule maps a float timestep (0 to 1) to a masking probability.
    """
    betas = None
    if name == "linear":
        betas = torch.linspace(start, end, timesteps, dtype=torch.float64)
    elif name == "cosine":
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999) # Clamp to avoid issues

    else:
        raise NotImplementedError(f"Mask schedule '{name}' not supported.")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Return a function that maps a float timestep (0 to 1) to a mask probability
    # For a mask schedule, we often want the probability of being masked at a given "noise level" or timestep.
    # This might be directly related to 1 - sqrt(alpha_bar_t) for noise, or a linear/cosine interpolation.
    
    # Let's create a callable that takes a float t (0 to 1) and returns a mask ratio.
    # We will interpolate the alphas_cumprod values for given t.
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def schedule_fn(t_float: torch.Tensor) -> torch.Tensor:
        # t_float is expected to be a tensor of floats between 0 and 1.
        # Scale t_float to be an index in [0, timesteps-1]
        t_indices = (t_float * (timesteps - 1)).long().clamp(0, timesteps - 1)
        
        # The mask probability can be defined in various ways.
        # A common approach for discrete diffusion masking is related to the noise schedule.
        # For example, the probability of a token being replaced by MASK or randomized.
        
        # Here, let's return 1 - sqrt_alphas_cumprod as a proxy for masking probability,
        # which is akin to the noise level for continuous diffusion.
        # This means, as t increases, more masking.
        return sqrt_one_minus_alphas_cumprod[t_indices]

    return schedule_fn

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

