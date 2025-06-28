from __future__ import annotations

import logging
import math
import sys
# from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
# from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule
from transformers import PretrainedConfig


# --- 新增 3D 分子编码器 ---
class Molecular3DEncoder(nn.Module):
    def __init__(self, config: "MMadaConfig"): # 这里的 config 是 MMadaConfig
        super().__init__()
        # 使用 nn.Embedding 编码原子类型
        # config.num_atom_types 应该是原子序数最大值 + 1 + 1 (0 for unknown, max_Z+1 for padding)
        self.atom_embedding = nn.Embedding(
            config.num_atom_types, config.mol_atom_embedding_dim, padding_idx=0 # 假设0是pad token
        )
        # 将 3D 坐标投影到高维空间
        self.coord_projection = nn.Linear(3, config.mol_coord_embedding_dim)

        # 一个简单的 MLP 来融合原子和坐标特征，并输出分子级嵌入
        self.per_atom_mlp = nn.Sequential(
            nn.Linear(config.mol_atom_embedding_dim + config.mol_coord_embedding_dim, config.fusion_hidden_dim),
            nn.GELU(), # 使用 GELU 激活函数
            nn.Linear(config.fusion_hidden_dim, config.mol_3d_encoder_output_dim)
        )
        # 如果你计划使用更复杂的结构如 GNNs 或 Transformer，需要在这里替换或添加
        # 例如：self.gnn = GNNLayer(...)

    def forward(self, atom_vec: torch.LongTensor, coordinates: torch.FloatTensor, atoms_mask: torch.BoolTensor):
        """
        Args:
            atom_vec: 原子类型 ID, (batch_size, max_atoms)
            coordinates: 原子 3D 坐标, (batch_size, max_atoms, 3)
            atoms_mask: 真实原子掩码 (True 为真实原子), (batch_size, max_atoms)
        Returns:
            分子嵌入: (batch_size, mol_3d_encoder_output_dim)
        """
        # 1. 编码原子类型和坐标
        atom_embeds = self.atom_embedding(atom_vec) # (B, N_max, mol_atom_embedding_dim)
        coord_embeds = self.coord_projection(coordinates) # (B, N_max, mol_coord_embedding_dim)
        
        # 2. 拼接原子和坐标嵌入
        fused_atom_coord_embeds = torch.cat([atom_embeds, coord_embeds], dim=-1) # (B, N_max, combined_dim)
        
        # 3. 通过每个原子的 MLP
        per_atom_features = self.per_atom_mlp(fused_atom_coord_embeds) # (B, N_max, mol_3d_encoder_output_dim)

        # 4. 池化：对真实原子进行平均池化得到分子级嵌入
        # 确保掩码维度匹配 (B, N_max, 1)
        # 使用 float() 将 bool 转换为 float (1.0/0.0)
        masked_features = per_atom_features * atoms_mask.unsqueeze(-1).float()
        molecular_embedding = masked_features.sum(dim=1) / (atoms_mask.sum(dim=1, keepdim=True).float() + 1e-5) # 避免除以零

        return molecular_embedding
    
class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            # "codebook_size",
            # "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
             # --- 新增配置项，以便在 config.yaml 中定义 3D 编码器和融合层的参数 ---
            "mol_atom_embedding_dim", # 分子原子嵌入维度
            "mol_coord_embedding_dim", # 分子坐标嵌入维度
            "mol_3d_encoder_output_dim", # 3D 编码器输出维度
            "fusion_hidden_dim", # 融合层隐藏维度
            "final_condition_dim", # 最终融合后传递给 LLM 的条件维度
            "num_atom_types", # 原子类型词汇表大小 (例如，Z=0到max_Z+1)
            "max_atoms", # 最大原子数 (与 dataset 保持一致)
            "output_atom_coords_dim", # 输出原子坐标的维度 (通常是3)
            "output_atom_type_dim", # 输出原子类型的维度 (通常是原子词汇表大小)
            "diffusion_timesteps", # 扩散过程的总时间步 (例如 1000)
            "noise_schedule_beta_start", # 扩散噪声调度参数
            "noise_schedule_beta_end",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])

        # 从父类 LLaDAConfig 复制一些关键参数，或者确保 MMadaConfig 包含它们
        # 这些参数会被 super().__init__ 使用
        self.d_model = kwargs.get("d_model", 4096) # LLM 模型的隐藏层维度
        self.n_heads = kwargs.get("n_heads", 32)
        self.n_layers = kwargs.get("n_layers", 32)
        self.vocab_size = kwargs.get("vocab_size", 126464) # 确保这里的 vocab_size 是扩展后的总词汇表大小
        self.mask_token_id = kwargs.get("mask_token_id", 126336)
        self.pad_token_id = kwargs.get("pad_token_id", 126081)

class MMadaModelLM(LLaDAModelLM):
    config_class = MMadaConfig
    base_model_prefix = "model"

    def __init__(self, config: MMadaConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

        self.molecular_3d_encoder = Molecular3DEncoder(config)

        # --- 多模态融合层 ---
        # 融合层的输入维度：LLM_d_model (text/selfies) + mol_3d_encoder_output_dim
        fusion_input_dim = config.d_model * 2 + config.mol_3d_encoder_output_dim # selfies_embeds + text_embeds + mol_3d_embeds
        
        self.multimodal_fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(config.fusion_hidden_dim, config.final_condition_dim) # 输出到 LLM 的条件维度
        )
        
        # --- 新增输出头：用于预测 3D 坐标和原子类型 ---
        # 这些头将作用于融合后的条件嵌入 (final_condition_embeds)，并输出扁平化的预测
        
        # 预测原子类型（分类任务）
        self.atom_type_prediction_head = nn.Linear(config.final_condition_dim, config.num_atom_types)
        
        # 预测原子坐标（回归任务）
        self.coordinates_prediction_head = nn.Linear(config.final_condition_dim, config.output_atom_coords_dim)
        
        # 为扩散过程定义噪声调度 (beta 值)
        self.betas = torch.linspace(config.noise_schedule_beta_start, config.noise_schedule_beta_end, config.diffusion_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) # 累积乘积
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def forward(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        text_input_ids: torch.LongTensor,
        text_attention_mask: torch.LongTensor,
        atom_vec: torch.LongTensor, # 真实的原子类型
        coordinates: torch.FloatTensor, # 真实的 3D 坐标
        atoms_mask: torch.BoolTensor,
        edge_type: Optional[torch.LongTensor] = None,
        bond_type: Optional[torch.LongTensor] = None,
        dist: Optional[torch.FloatTensor] = None,
        rdmol2selfies: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None, # 扩散时间步 (训练时添加噪声用)
        # LLM forward 的原始参数，不直接在这里传递给 super().forward()
        # **kwargs 用于捕获其他可能的参数，确保兼容性
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]: # 返回预测坐标和原子类型 logits
        """
        MMadaModelLM 的主要前向传播方法。
        接收多模态输入，融合后通过 LLM 主干，并预测 3D 结构。
        注意：在这个简化版本中，LLM 主干 (self.model) 主要用于获取 token 嵌入，
        然后预测头直接作用于多模模态融合后的特征。
        """
        batch_size = selfies_input_ids.shape[0] # 获取批次大小

        # 1. 编码 SELFIES 和文本
        selfies_embeds = self.model.embed_tokens(selfies_input_ids) # (B, L_selfies, D_model)
        text_embeds = self.model.embed_tokens(text_input_ids) # (B, L_text, D_model)
        
        # 平均池化（或更复杂的聚合）
        selfies_sum = (selfies_embeds * selfies_attention_mask.unsqueeze(-1).float()).sum(dim=1)
        selfies_count = selfies_attention_mask.sum(dim=1, keepdim=True).float() + 1e-5
        selfies_context_embeds = selfies_sum / selfies_count # (B, D_model)

        text_sum = (text_embeds * text_attention_mask.unsqueeze(-1).float()).sum(dim=1)
        text_count = text_attention_mask.sum(dim=1, keepdim=True).float() + 1e-5
        text_context_embeds = text_sum / text_count # (B, D_model)
        
        # 2. 编码 3D 数据
        # self.molecular_3d_encoder 期望 atom_vec, coordinates, atoms_mask
        mol_3d_embeds = self.molecular_3d_encoder(atom_vec, coordinates, atoms_mask)

        # 3. 融合所有模态的嵌入
        fused_multimodal_embeds = torch.cat(
            [selfies_context_embeds, text_context_embeds, mol_3d_embeds], dim=-1
        )
        
        # 通过融合 MLP 得到最终的条件嵌入
        final_condition_embeds = self.multimodal_fusion_mlp(fused_multimodal_embeds)
        
        # 4. 应用预测头
        # 预测原子类型 (分类任务)
        # (B, max_atoms * num_atom_types) -> reshape to (B, max_atoms, num_atom_types)
        predicted_atom_type_logits_flat = self.atom_type_prediction_head(final_condition_embeds)
        predicted_atom_type_logits = predicted_atom_type_logits_flat.view(batch_size, self.config.max_atoms, self.config.num_atom_types)
        
        # 预测原子坐标 (回归任务)
        # (B, max_atoms * output_atom_coords_dim) -> reshape to (B, max_atoms, output_atom_coords_dim)
        predicted_coordinates_flat = self.coordinates_prediction_head(final_condition_embeds)
        predicted_coordinates = predicted_coordinates_flat.view(batch_size, self.config.max_atoms, self.config.output_atom_coords_dim)

        return predicted_coordinates, predicted_atom_type_logits

    def get_alpha_bar(self, timesteps: torch.LongTensor) -> torch.FloatTensor:
        """从预计算的 alpha_bars 中获取指定时间步的值。"""
        # 确保 alpha_bars 在正确的设备上
        return self.alpha_bars[timesteps].to(timesteps.device)

    # --- 彻底重写 forward_process 函数 ---
    def forward_process(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        text_input_ids: torch.LongTensor,
        text_attention_mask: torch.LongTensor,
        atom_vec: torch.LongTensor, # 真实的原子类型
        coordinates: torch.FloatTensor, # 真实的 3D 坐标
        atoms_mask: torch.BoolTensor,
        edge_type: Optional[torch.LongTensor] = None,
        bond_type: Optional[torch.LongTensor] = None,
        dist: Optional[torch.FloatTensor] = None,
        rdmol2selfies: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None, # 这里的 labels 接收的是真实坐标
        **kwargs,
    ):
        batch_size = coordinates.shape[0]
        
        # 随机采样一个噪声时间步
        timesteps = torch.randint(0, self.config.diffusion_timesteps, (batch_size,), device=coordinates.device).long()
        
        # 对坐标添加高斯噪声 (DDPM 风格)
        noise = torch.randn_like(coordinates)
        alpha_bar_t = self.get_alpha_bar(timesteps) 
        
        # (B, 1, 1) or (B, 1)
        sqrt_alpha_bar_t = alpha_bar_t.sqrt().unsqueeze(-1) 
        sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt().unsqueeze(-1)
        
        noisy_coordinates = sqrt_alpha_bar_t * coordinates + \
                            sqrt_one_minus_alpha_bar_t * noise
        
        # 调用 MMadaModelLM 的主要前向方法 (self.forward)
        # 它将返回预测的 coordinates 和 atom_type_logits
        predicted_coordinates, predicted_atom_type_logits = self.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            atom_vec=atom_vec, # 传入真实的原子类型作为条件
            coordinates=noisy_coordinates, # 传入噪声化的坐标
            atoms_mask=atoms_mask,
            edge_type=edge_type, 
            bond_type=bond_type, 
            dist=dist,           
            rdmol2selfies=rdmol2selfies,
            timesteps=timesteps, # 传递时间步
        )

        # --- 损失计算 ---
        # true_coordinates 是真实的干净坐标 (来自 DataLoader，通过 labels 传入)
        true_coordinates = labels 
        true_atom_vec = atom_vec # 真实的原子类型 ID

        # 1. 坐标预测损失 (MSE Loss)
        # 只在真实原子位置计算损失
        coords_loss = F.mse_loss(
            predicted_coordinates * atoms_mask.unsqueeze(-1).float(),
            true_coordinates * atoms_mask.unsqueeze(-1).float(),
            reduction='sum'
        ) / (atoms_mask.sum().float() + 1e-5) # 平均每个真实原子的损失

        # 2. 原子类型预测损失 (Cross-Entropy Loss)
        # 只在真实原子位置计算损失
        # predicted_atom_type_logits: (B, N_max, num_atom_types)
        # true_atom_vec: (B, N_max)
        
        # 将 logits 和 labels 展平，只考虑真实原子
        atom_type_logits_flat = predicted_atom_type_logits[atoms_mask].contiguous().view(-1, self.config.num_atom_types)
        true_atom_vec_flat = true_atom_vec[atoms_mask].contiguous().view(-1)

        # 确保 target 不会是 padding_idx，或者 ignore_index 设置正确
        # config.pad_token_id 是LLM的pad token，不一定是原子类型的padding_idx
        # 如果 atom_to_id 0 是 unknown，且不希望它参与损失计算，可以 ignore_index=0
        atom_type_loss = F.cross_entropy(
            atom_type_logits_flat,
            true_atom_vec_flat,
            ignore_index=0, # 假设原子 ID 0 是未知/填充原子，不参与损失计算
            reduction='mean'
        )

        # 总损失
        # 你可以为坐标损失和原子类型损失设置权重
        loss = coords_loss + atom_type_loss # 简单相加

        # 返回总损失
        return loss

AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)