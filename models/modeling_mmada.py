# models/modeling_mmada.py

from __future__ import annotations

import logging
import math
import sys
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
from transformers.modeling_outputs import CausalLMOutputWithPast 
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache


from .modeling_llada import LLaDAModelLM 

from .sampling import cosine_schedule 
from transformers import PretrainedConfig

from .common_modules import MLP

class Molecular3DEncoder(nn.Module):
    def __init__(self, config: "MMadaConfig"): 
        super().__init__()
        self.atom_embedding = nn.Embedding(
            config.num_atom_types, config.mol_atom_embedding_dim, padding_idx=0 
        )
        self.coord_projection = nn.Linear(3, config.mol_coord_embedding_dim)

        self.per_atom_mlp = nn.Sequential(
            nn.Linear(config.mol_atom_embedding_dim + config.mol_coord_embedding_dim, config.fusion_hidden_dim),
            nn.GELU(), 
            nn.Linear(config.fusion_hidden_dim, config.mol_3d_encoder_output_dim)
        )

    def forward(self, atom_vec: torch.LongTensor, coordinates: torch.FloatTensor, atoms_mask: torch.BoolTensor):
        atom_embeds = self.atom_embedding(atom_vec) 
        coord_embeds = self.coord_projection(coordinates)
        
        fused_atom_coord_embeds = torch.cat([atom_embeds, coord_embeds], dim=-1)
        
        per_atom_features = self.per_atom_mlp(fused_atom_coord_embeds)

        masked_features = per_atom_features * atoms_mask.unsqueeze(-1).float()
        molecular_embedding = masked_features.sum(dim=1) / (atoms_mask.sum(dim=1, keepdim=True).float() + 1e-5)

        return molecular_embedding


class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "d_model", 
            "vocab_size", 
            "llm_vocab_size", # LLM 的原始词汇表大小
            "llm_model_path", # LLM 预训练模型路径
            "mask_token_id", 
            "pad_token_id",
            "max_sequence_length", # LLM 的最大序列长度
            
            "num_new_special_tokens", 
            "gradient_checkpointing",
            "new_vocab_size", 

            "mol_atom_embedding_dim", 
            "mol_coord_embedding_dim", 
            "mol_3d_encoder_output_dim", 
            "fusion_hidden_dim", 
            "final_condition_dim", 
            "num_atom_types", 
            "max_atoms", 
            "output_atom_coords_dim", 
            "output_atom_type_dim", 
            "diffusion_timesteps", 
            "noise_schedule_beta_start", 
            "noise_schedule_beta_end",

            "coords_coeff",
            "atom_type_coeff",
            "alignment_coeff",
            "selfies_coeff",
            "hierarchical_coeff",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])


class MMadaModelLM(nn.Module): 

    def __init__(self, config: "MMadaConfig", base_llm_model: LLaDAModelLM, *args, **kwargs): # <-- 接收 base_llm_model
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__()
        
        self.config = config 
        self.llm_backbone = base_llm_model 

        self.molecular_3d_encoder = Molecular3DEncoder(config)

        fusion_input_dim = config.d_model * 2 + config.mol_3d_encoder_output_dim 
        
        self.multimodal_fusion_mlp = MLP( # 确保 MLP 已导入
            input_dim=fusion_input_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.final_condition_dim,
            num_layers=2 # 假设有两层，根据config 调整
        )
        
        self.condition_to_llm_projection = nn.Linear(config.final_condition_dim, config.d_model)
        
        self.atom_type_prediction_head = nn.Linear(config.d_model, config.max_atoms * config.num_atom_types)
        self.coordinates_prediction_head = nn.Linear(config.d_model, config.max_atoms * config.output_atom_coords_dim)
        
        self.betas = torch.linspace(config.noise_schedule_beta_start, config.noise_schedule_beta_end, config.diffusion_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.mask_token_id = config.mask_token_id


    def forward(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        text_input_ids: torch.LongTensor,
        text_attention_mask: torch.LongTensor,
        atom_vec: torch.LongTensor, 
        coordinates: torch.FloatTensor, 
        atoms_mask: torch.BoolTensor,
        edge_type: Optional[torch.LongTensor] = None, 
        bond_type: Optional[torch.LongTensor] = None,
        dist: Optional[torch.FloatTensor] = None,
        rdmol2selfies: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None, 
        **kwargs, 
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        MMadaModelLM 的主要前向传播方法。
        接收多模态输入，融合后通过 LLM 主干，并预测 3D 结构。
        """
        batch_size = selfies_input_ids.shape[0]

        # 1. 编码 SELFIES 和文本
        selfies_embeds = self.llm_backbone.get_input_embeddings()(selfies_input_ids)# (B, L_selfies, D_model)
        text_embeds = self.llm_backbone.get_input_embeddings()(text_input_ids) # (B, L_text, D_model)

        # --- DEBUG PRINT ---
        print(f"DEBUG_FWD: selfies_embeds shape: {selfies_embeds.shape}, device: {selfies_embeds.device}")
        print(f"DEBUG_FWD: text_embeds shape: {text_embeds.shape}, device: {text_embeds.device}")
        # --- END DEBUG PRINT ---
        
        # 平均池化
        selfies_sum = (selfies_embeds * selfies_attention_mask.unsqueeze(-1).float()).sum(dim=1)
        selfies_count = selfies_attention_mask.sum(dim=1, keepdim=True).float() + 1e-5
        selfies_context_embeds = selfies_sum / selfies_count # (B, D_model)

        text_sum = (text_embeds * text_attention_mask.unsqueeze(-1).float()).sum(dim=1)
        text_count = text_attention_mask.sum(dim=1, keepdim=True).float() + 1e-5
        text_context_embeds = text_sum / text_count # (B, D_model)
        
        # 2. 编码 3D 数据
        mol_3d_embeds = self.molecular_3d_encoder(atom_vec, coordinates, atoms_mask)

        # --- DEBUG PRINT ---
        print(f"DEBUG_FWD: mol_3d_embeds shape: {mol_3d_embeds.shape}, device: {mol_3d_embeds.device}")
        # --- END DEBUG PRINT ---

        # 3. 融合所有模态的嵌入
        fused_multimodal_embeds = torch.cat(
            [selfies_context_embeds, text_context_embeds, mol_3d_embeds], dim=-1
        )
        
        final_condition_embeds = self.multimodal_fusion_mlp(fused_multimodal_embeds)

        # --- DEBUG PRINT ---
        print(f"DEBUG_FWD: final_condition_embeds shape: {final_condition_embeds.shape}, device: {final_condition_embeds.device}")
        # --- END DEBUG PRINT ---
        
    # 4. 将融合后的条件嵌入作为 LLM 的条件（注入到 LLM 的输入中）
        cond_llm_embeds = self.condition_to_llm_projection(final_condition_embeds) 

        condition_token_id = self.config.pad_token_id 
        cond_input_ids = torch.full((batch_size, 1), condition_token_id, dtype=torch.long, device=selfies_input_ids.device)
        cond_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=selfies_input_ids.device)

        llm_text_selfies_input_ids = torch.cat([selfies_input_ids, text_input_ids], dim=1) 
        llm_text_selfies_attention_mask = torch.cat([selfies_attention_mask, text_attention_mask], dim=1)
        llm_text_selfies_embeds = self.llm_backbone.get_input_embeddings()(llm_text_selfies_input_ids) 

        combined_inputs_embeds = torch.cat([cond_llm_embeds.unsqueeze(1), llm_text_selfies_embeds], dim=1) 
        combined_attention_mask = torch.cat([cond_attention_mask, llm_text_selfies_attention_mask], dim=1) 

        combined_attention_bias = (combined_attention_mask[:, :, None] & combined_attention_mask[:, None, :]).bool().unsqueeze(1)

        # --- DEBUG PRINT ---
        print(f"DEBUG_FWD: LLM backbone input embeds shape: {combined_inputs_embeds.shape}, device: {combined_inputs_embeds.device}")
        print(f"DEBUG_FWD: Calling llm_backbone forward...")
        # --- END DEBUG PRINT ---

        # 调用 LLaDAModelLM 骨干网络 (self.llm_backbone) 的 forward 方法
        llm_output: CausalLMOutputWithPast = self.llm_backbone( 
            inputs_embeds=combined_inputs_embeds, 
            attention_mask=combined_attention_mask,
            attention_bias=combined_attention_bias,
            output_hidden_states=True, 
            return_dict=True,
        )

        # --- DEBUG PRINT ---
        print(f"DEBUG_FWD: llm_backbone forward completed.")
        # --- END DEBUG PRINT ---

        llm_last_hidden_state = llm_output.hidden_states[-1] 

        llm_output_for_prediction = llm_last_hidden_state[:, 1:, :] 

        pooled_llm_output = (llm_output_for_prediction * llm_text_selfies_attention_mask.unsqueeze(-1).float()).sum(dim=1) / \
                            (llm_text_selfies_attention_mask.sum(dim=1, keepdim=True).float() + 1e-5) 

        # --- DEBUG PRINT ---
        print(f"DEBUG_FWD: pooled_llm_output shape: {pooled_llm_output.shape}, device: {pooled_llm_output.device}")
        print(f"DEBUG_FWD: Applying prediction heads...")
        # --- END DEBUG PRINT ---

        # 5. 应用预测头
        predicted_atom_type_logits_flat = self.atom_type_prediction_head(pooled_llm_output)
        predicted_atom_type_logits = predicted_atom_type_logits_flat.view(batch_size, self.config.max_atoms, self.config.num_atom_types)

        predicted_coordinates_flat = self.coordinates_prediction_head(pooled_llm_output)
        predicted_coordinates = predicted_coordinates_flat.view(batch_size, self.config.max_atoms, self.config.output_atom_coords_dim)

        # --- DEBUG PRINT ---
        print(f"DEBUG_FWD: Prediction heads completed.")
        # --- END DEBUG PRINT ---

        return predicted_coordinates, predicted_atom_type_logits

    def get_alpha_bar(self, timesteps: torch.LongTensor) -> torch.FloatTensor:
        """从预计算的 alpha_bars 中获取指定时间步的值。"""
        # 确保 alpha_bars 在正确的设备上
        return self.alpha_bars[timesteps].to(timesteps.device)

    def forward_process( # 这个函数就是 train_mmada_stage2.py 中调用的
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        text_input_ids: torch.LongTensor,
        text_attention_mask: torch.LongTensor,
        atom_vec: torch.LongTensor, # 真实的原子类型
        coordinates: torch.FloatTensor, # 真实的 3D 坐标
        atoms_mask: torch.BoolTensor,
        task_type: str,
        true_coordinates: torch.FloatTensor, 
        true_atom_vec: torch.LongTensor,
        mask_schedule_coords: Callable,

        edge_type: Optional[torch.LongTensor] = None, 
        bond_type: Optional[torch.LongTensor] = None,
        dist: Optional[torch.FloatTensor] = None,
        rdmol2selfies: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
    ):
        batch_size = coordinates.shape[0]

        losses = {}
        
        # ------------------------------------------------------------------------
        # 1D (SELFIES/Text) to 3D Generation Task (去噪坐标 & 原子类型)
        # 明确只处理 '1d_to_3d' 任务
        if task_type == '1d_to_3d':
            # 直接调用 self.forward，传入加噪后的 coordinates 作为 3D 编码器输入
            # 注意：这里不再对 coordinates 加噪，因为 prepare_molecular_inputs_and_labels 已经处理了
            predicted_coordinates, predicted_atom_type_logits = self.forward(
                selfies_input_ids=selfies_input_ids,
                selfies_attention_mask=selfies_attention_mask,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                atom_vec=atom_vec, # 传入 atom_vec
                coordinates=coordinates, # <-- 传入已经噪声化后的坐标 (来自 prepare_molecular_inputs_and_labels)
                atoms_mask=atoms_mask,
                # timesteps 参数不需要传入 forward
            )

            # --- 损失计算 ---
            # 1. 坐标预测损失 (MSE Loss)
            coords_loss = F.mse_loss(
                predicted_coordinates * atoms_mask.unsqueeze(-1).float(),
                true_coordinates * atoms_mask.unsqueeze(-1).float(), # 目标是真实（干净）坐标
                reduction='sum'
            ) / (atoms_mask.sum().float() + 1e-5)
            losses['coords_loss'] = coords_loss

            # 2. 原子类型预测损失 (Cross-Entropy Loss)
            atom_type_logits_flat = predicted_atom_type_logits[atoms_mask].contiguous().view(-1, self.config.num_atom_types)
            true_atom_vec_flat = true_atom_vec[atoms_mask].contiguous().view(-1) # 使用传入的 true_atom_vec 作为目标

            atom_type_loss = F.cross_entropy(
                atom_type_logits_flat,
                true_atom_vec_flat,
                ignore_index=0, # 假设原子 ID 0 是填充/未知原子，不参与损失计算
                reduction='mean'
            )
            losses['atom_type_loss'] = atom_type_loss
            
            # --- 对齐损失 (可选，推荐) ---
            # 为了计算对齐损失，forward 方法需要返回 selfies_context_embeds 和 mol_3d_embeds
            # 如果你在 forward 方法的返回类型中添加了这些，并且 return_dict=True，
            # 可以在这里通过 model_outputs["selfies_context_embeds"] 等获取并计算损失。
            # 这里为了简化，暂时不在此处添加对齐损失的计算，专注于核心功能。
            # 如果未来需要，可以修改 forward 的返回类型为 dict，并在其中包含这些嵌入。

        # ------------------------------------------------------------------------
        # 结合所有损失，并应用系数
        total_loss = torch.tensor(0.0, device=coordinates.device)
        if not losses: # 如果没有任何损失被计算，说明 task_type 不匹配
            raise ValueError(f"No loss calculated for task_type: {task_type}. Check your task_type logic and data.")

        for loss_name, loss_value in losses.items():
            # 从 config 中获取系数 (您需要在 config.yaml 和 MMadaConfig 中添加这些系数)
            # 例如 config.coords_coeff, config.atom_type_coeff, config.alignment_coeff
            coeff = getattr(self.config, loss_name.replace('_loss', '_coeff'), 1.0)
            total_loss += coeff * loss_value

        return total_loss, losses # 返回总损失和各项损失的字典以供日志记录

AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)