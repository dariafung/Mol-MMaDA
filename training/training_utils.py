import math
import random
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf


# ----------------------------------------------------------------------
# Mask‑schedule utilities
# ----------------------------------------------------------------------
def get_mask_schedule(
    schedule_name: str,
    timesteps: int,
    start: float,
    end: float,
    x: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a 1‑D tensor with `timesteps` values in [start, end]."""
    if schedule_name == "linear":
        return torch.linspace(start, end, timesteps, dtype=torch.float32)
    if schedule_name == "cosine":
        t = torch.arange(timesteps, dtype=torch.float32) if x is None else x.float()
        f_t = torch.cos(((t / timesteps) + 0.008) / 1.008 * (math.pi / 2)) ** 2
        alpha_bar = f_t / f_t[0]
        mask_ratio = 1.0 - torch.sqrt(alpha_bar)           # low → high
        return (start + (end - start) * mask_ratio).clamp(0.0, 1.0)
    raise ValueError(f"Unknown schedule name: {schedule_name}")


# ----------------------------------------------------------------------
# SELFIES masking helper
# ----------------------------------------------------------------------
def apply_selfies_masking(
    original_ids: torch.LongTensor,
    mask_token_id: int,
    timestep: int,
    total_timesteps: int,
    schedule_name: str,
    schedule_start: float,
    schedule_end: float,
    device: torch.device,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Mask SELFIES tokens according to the current diffusion timestep."""
    bsz, seq_len = original_ids.shape
    norm_t = timestep / (total_timesteps - 1)

    if schedule_name == "linear":
        mask_ratio = schedule_start + (schedule_end - schedule_start) * norm_t
    elif schedule_name == "cosine":
        t_norm = (timestep + 1e-5) / total_timesteps
        mask_ratio = schedule_start + (schedule_end - schedule_start) * (1 - math.cos(t_norm * math.pi)) / 2
    else:
        mask_ratio = schedule_start

    mask_ratio = float(torch.clamp(torch.tensor(mask_ratio), 0.0, 1.0))

    masked_ids = original_ids.clone()
    labels = original_ids.clone()

    for i in range(bsz):
        valid_idx = (original_ids[i] != 0).nonzero(as_tuple=True)[0]  # assume 0 is PAD
        if len(valid_idx) == 0:
            continue
        num_mask = int(len(valid_idx) * mask_ratio)
        mask_idx = random.sample(valid_idx.tolist(), min(num_mask, len(valid_idx)))

        masked_ids[i, mask_idx] = mask_token_id
        labels[i, list(set(valid_idx.tolist()) - set(mask_idx))] = -100
        labels[i, (original_ids[i] == 0).nonzero(as_tuple=True)[0]] = -100

    return masked_ids.to(device), labels.to(device)


# ----------------------------------------------------------------------
# Continuous diffusion noise schedule
# ----------------------------------------------------------------------
def get_noise_schedule(
    name: str,
    beta_start: float,
    beta_end: float,
    timesteps: int,
    device: torch.device,
):
    """Return a callable mapping integer timesteps to sqrt(alpha_bar)."""
    if name == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    elif name == "cosine":
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1, device=device)
        alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / float(alpha_bar[0].item())
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Noise schedule '{name}' is not supported.")

    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).to(device)

    def schedule_fn(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device)
        return torch.sqrt(alphas_cumprod[t])

    return schedule_fn


# ----------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------
def get_config():
    import argparse

    parser = argparse.ArgumentParser(description="LLaDA training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--local_rank", type=int, default=0)
    args, unknown = parser.parse_known_args()

    yaml_conf = OmegaConf.load(args.config)
    cli_overrides = OmegaConf.from_cli(unknown)
    return OmegaConf.merge(yaml_conf, cli_overrides)


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    """Flatten an OmegaConf DictConfig/ListConfig into a list of (key, value) pairs."""
    out: List[Tuple[str, Any]] = []

    def recurse(prefix: str, value: Any):
        if isinstance(value, DictConfig):
            for k, v in value.items_ex(resolve=resolve):
                recurse(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, ListConfig):
            for idx, v in enumerate(value._iter_ex(resolve=resolve)):
                recurse(f"{prefix}.{idx}" if prefix else str(idx), v)
        else:
            out.append((prefix, value))

    recurse("", cfg)
    return out


# ----------------------------------------------------------------------
# Misc utilities
# ----------------------------------------------------------------------
def soft_target_cross_entropy(logits, targets, soft_targets):
    logits = logits[:, 1:]
    targets = targets[:, 1:]
    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    pad_mask = targets.eq(-100)
    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(pad_mask, 0.0)
    active = pad_mask.numel() - pad_mask.long().sum()
    return loss.sum() / active


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(
    tokens: torch.Tensor,
    mask_id: int,
    mask_ratio: float,
    vocab_size: int,
    is_train: bool = True,
    seed: Optional[int] = None,
):
    """Standard BERT‑style masking (mask token or random replacement)."""
    if not is_train and seed is not None:
        cpu_state, cuda_state, py_state = (
            torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            random.getstate(),
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    mask = torch.rand_like(tokens, dtype=torch.float32) < mask_ratio
    input_ids = tokens.clone()
    input_ids[mask] = mask_id
    labels = torch.where(mask, tokens, torch.tensor(-100, dtype=tokens.dtype, device=tokens.device))

    if not is_train and seed is not None:
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        random.setstate(py_state)

    return input_ids, labels, None


class AverageMeter:
    """Compute running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
