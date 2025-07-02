import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing import Optional, Callable, Dict, Any, Tuple

from .modeling_llada import LLaDAModelLM
from .common_modules import MLP, SinusoidalPositionalEmbedding
from training.utils import get_mask_schedule


# --- 1. MMadaConfig 类 ---
class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(
        self,
        # LLM backbone config
        llm_config_path: str = "llada-8b-instruct",
        llm_model_name_or_path: str = None,

        # Molecular 3D Encoder (原有的 Molecular3DEncoder 参数)
        mol_atom_embedding_dim: int = 128,
        mol_coord_embedding_dim: int = 128,
        mol_3d_encoder_output_dim: int = 768, # Should match LLM hidden size or be projected
        num_atom_types: int = 120, # Example: up to U, with 0 for padding
        max_atoms: int = 256, # Max atoms in a molecule
        output_atom_coords_dim: int = 3, # x, y, z coordinates
        output_atom_type_dim: int = 120, # Number of atom types for classification

        # Fusion network parameters (原有的融合网络参数)
        d_model: int = 768, # Typically LLM hidden size
        fusion_hidden_dim: int = 2048,
        final_condition_dim: int = 768,

        # Task specific parameters
        diffusion_timesteps: int = 1000,
        noise_schedule_beta_start: float = 0.0001,
        noise_schedule_beta_end: float = 0.02,
        
        # Loss coefficients for molecular and other tasks
        coords_coeff: float = 1.0,
        atom_type_coeff: float = 1.0,
        selfies_coeff: float = 0.0,  # SELFIES预测损失系数
        alignment_coeff: float = 0.0, # 对齐损失系数
        hierarchical_coeff: float = 0.0, # 分层对齐损失系数

        # Masking parameters for discrete diffusion
        mask_token_id: int = -1, # To be set by tokenizer vocab size + 1
        mask_replace_ratio: float = 0.1,
        mask_schedule_name: str = "linear", # linear, cosine etc.
        mask_schedule_start: float = 0.0001,
        mask_schedule_end: float = 0.02,

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_config_path = llm_config_path
        self.llm_model_name_or_path = llm_model_name_or_path
        
        # 移除图像相关参数的赋值
        # self.image_tokenizer_path = image_tokenizer_path
        # self.image_tokenizer_config_file = image_tokenizer_config_file
        # self.image_tokenizer_model_file = image_tokenizer_model_file
        # self.image_unet_config_path = image_unet_config_path
        # self.image_unet_model_path = image_unet_model_path

        self.mol_atom_embedding_dim = mol_atom_embedding_dim
        self.mol_coord_embedding_dim = mol_coord_embedding_dim
        self.mol_3d_encoder_output_dim = mol_3d_encoder_output_dim
        self.num_atom_types = num_atom_types
        self.max_atoms = max_atoms
        self.output_atom_coords_dim = output_atom_coords_dim
        self.output_atom_type_dim = output_atom_type_dim

        self.d_model = d_model
        self.fusion_hidden_dim = fusion_hidden_dim
        self.final_condition_dim = final_condition_dim

        self.diffusion_timesteps = diffusion_timesteps
        self.noise_schedule_beta_start = noise_schedule_beta_start
        self.noise_schedule_beta_end = noise_schedule_beta_end

        self.coords_coeff = coords_coeff
        self.atom_type_coeff = atom_type_coeff
        self.selfies_coeff = selfies_coeff
        self.alignment_coeff = alignment_coeff
        self.hierarchical_coeff = hierarchical_coeff

        self.mask_token_id = mask_token_id
        self.mask_replace_ratio = mask_replace_ratio
        self.mask_schedule_name = mask_schedule_name
        self.mask_schedule_start = mask_schedule_start
        self.mask_schedule_end = mask_schedule_end


# --- 2. Molecular3DEncoder 类 ---
class Molecular3DEncoder(nn.Module):
    def __init__(self, config: MMadaConfig):
        super().__init__()
        self.config = config
        self.atom_embedding = nn.Embedding(config.num_atom_types, config.mol_atom_embedding_dim)
        self.coord_projection = nn.Linear(config.output_atom_coords_dim, config.mol_coord_embedding_dim)
        
        # Combined feature dimension for per-atom MLP
        combined_atom_feat_dim = config.mol_atom_embedding_dim + config.mol_coord_embedding_dim
        
        self.per_atom_mlp = MLP(
            input_dim=combined_atom_feat_dim,
            hidden_dim=combined_atom_feat_dim * 2,
            output_dim=config.mol_3d_encoder_output_dim,
            num_layers=2
        )
        # Positional embedding for atoms based on their index in the sequence
        self.position_embeddings = SinusoidalPositionalEmbedding(
            config.mol_3d_encoder_output_dim, init_range=config.max_atoms
        )

    def forward(self, atom_vec: torch.LongTensor, coordinates: torch.FloatTensor, atoms_mask: torch.BoolTensor):
        atom_embeds = self.atom_embedding(atom_vec)
        coord_embeds = self.coord_projection(coordinates)
        
        # Concatenate atom type and coordinate embeddings
        combined_embeds = torch.cat([atom_embeds, coord_embeds], dim=-1)
        
        # Apply per-atom MLP
        per_atom_features = self.per_atom_mlp(combined_embeds)
        
        # Add positional embeddings
        position_ids = torch.arange(per_atom_features.shape[1], device=per_atom_features.device).unsqueeze(0)
        per_atom_features = per_atom_features + self.position_embeddings(position_ids)

        # Apply mask and average pool to get a single molecular embedding
        # Ensure mask is broadcastable (batch_size, seq_len, 1)
        masked_features = per_atom_features * atoms_mask.unsqueeze(-1).float()
        
        # Average pooling over valid atoms
        molecular_embedding = masked_features.sum(dim=1) / (atoms_mask.sum(dim=1, keepdim=True).float() + 1e-5)
        
        return molecular_embedding, per_atom_features # 返回per_atom_features用于潜在的分层对齐


# --- 3. MMadaModelLM 类 ---
class MMadaModelLM(nn.Module):
    def __init__(self, config: MMadaConfig):
        super().__init__()
        self.config = config

        # Load LLaDA backbone
        self.llm_backbone = LLaDAModelLM.from_pretrained(config.llm_model_name_or_path)
        self.llm_backbone.eval() # Typically freeze LLM backbone for multimodal fine-tuning stages
        for param in self.llm_backbone.parameters():
            param.requires_grad = False

        # Molecular 3D Encoder (保留)
        self.molecular_3d_encoder = Molecular3DEncoder(config)

        # Multimodal fusion MLP (保留)
        # Input dim for fusion: LLM output dim (from selfies context) + Mol3D encoder output dim
        fusion_input_dim = config.d_model + config.mol_3d_encoder_output_dim
        self.multimodal_fusion_mlp = MLP(
            input_dim=fusion_input_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.final_condition_dim,
            num_layers=2
        )

        # Prediction heads for 3D molecular generation (保留)
        self.coordinates_prediction_head = nn.Linear(config.final_condition_dim, config.output_atom_coords_dim)
        self.atom_type_prediction_head = nn.Linear(config.final_condition_dim, config.output_atom_type_dim)

        # New: Projection layer for alignment loss if dimensions don't match (保留)
        if config.mol_3d_encoder_output_dim != config.d_model:
            self.mol_embed_projection_for_alignment = nn.Linear(config.mol_3d_encoder_output_dim, config.d_model)
        else:
            self.mol_embed_projection_for_alignment = nn.Identity()

        # 旧的 SELFIES prediction head (如果 LLM 的 lm_head 不直接用于 SELFIES)
        # self.selfies_prediction_head = nn.Linear(config.d_model, config.selfies_vocab_size) # Example if separate vocab


    def forward(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        atom_vec: torch.LongTensor,
        coordinates: torch.FloatTensor,
        atoms_mask: torch.BoolTensor,
        text_input_ids: Optional[torch.LongTensor] = None, # Optional for 1d_to_3d
        text_attention_mask: Optional[torch.LongTensor] = None, # Optional for 1d_to_3d
        timesteps: Optional[torch.LongTensor] = None,
        # 移除 image_input_ids 和 image_attention_mask
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_size = selfies_input_ids.shape[0]
        
        # 1. Process SELFIES input using LLM backbone
        llm_selfies_output = self.llm_backbone(
            input_ids=selfies_input_ids,
            attention_mask=selfies_attention_mask,
            output_hidden_states=True # We need hidden states for SELFIES context
        )
        # Get the last hidden state from the LLM
        # Shape: (batch_size, sequence_length, hidden_size)
        selfies_hidden_states = llm_selfies_output.hidden_states[-1]

        # Average pool SELFIES hidden states to get a single context embedding
        # Apply mask to sum and then divide by count of non-padded tokens
        selfies_context_embeds = (selfies_hidden_states * selfies_attention_mask.unsqueeze(-1).float()).sum(dim=1) / \
                                (selfies_attention_mask.sum(dim=1, keepdim=True).float() + 1e-5)
        # Shape: (batch_size, hidden_size)


        # 2. Process 3D molecular input (保留 Molecular3DEncoder)
        mol_3d_embeds, per_atom_features = self.molecular_3d_encoder(atom_vec, coordinates, atoms_mask)
        # mol_3d_embeds shape: (batch_size, mol_3d_encoder_output_dim)
        # per_atom_features shape: (batch_size, max_atoms, mol_3d_encoder_output_dim)


        # 3. Multimodal fusion (保留)
        fused_features = torch.cat([selfies_context_embeds, mol_3d_embeds], dim=-1)
        final_condition_embeds = self.multimodal_fusion_mlp(fused_features)
        # Shape: (batch_size, final_condition_dim)

        # 4. Prediction heads for 3D generation (保留)
        predicted_coordinates = self.coordinates_prediction_head(final_condition_embeds)
        predicted_atom_type_logits = self.atom_type_prediction_head(final_condition_embeds)

        # For SELFIES prediction (if selfies_coeff > 0)
        selfies_logits = llm_selfies_output.logits[:, :selfies_input_ids.shape[1], :]
        
        # Return all necessary outputs for loss calculation
        return predicted_coordinates, predicted_atom_type_logits, selfies_logits, selfies_context_embeds, mol_3d_embeds, per_atom_features


    def forward_process(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        text_input_ids: Optional[torch.LongTensor],
        text_attention_mask: Optional[torch.LongTensor],
        atom_vec: torch.LongTensor,
        coordinates: torch.FloatTensor, # 这里的 coordinates 应该是原始干净的 x_0
        atoms_mask: torch.BoolTensor,
        task_type: str,
        true_coordinates: torch.FloatTensor, # 真实的干净坐标 x_0
        true_atom_vec: torch.LongTensor, # 真实的原子类型
        mask_schedule_coords: Callable, # For continuous diffusion noise
        true_selfies_labels: Optional[torch.LongTensor] = None,
        timesteps: Optional[torch.LongTensor] = None, # Timestep for diffusion
        # 移除 image_input_ids, image_attention_mask, true_image_labels
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = coordinates.shape[0]
        losses = {}

        # Determine diffusion timestep for continuous coordinates
        if timesteps is None:
            timesteps = torch.randint(0, self.config.diffusion_timesteps, (batch_size,), device=coordinates.device).long()
        
        # Add noise to coordinates for continuous diffusion input (x_t)
        # 这里的 coordinates 是原始干净的 x_0
        noise = torch.randn_like(coordinates) * atoms_mask.unsqueeze(-1).float()
        alphas_bar_sqrt = mask_schedule_coords(timesteps).unsqueeze(-1).unsqueeze(-1)
        one_minus_alphas_bar_sqrt = (1.0 - alphas_bar_sqrt**2).sqrt()

        # Noisy coordinates for model input (x_t)
        noisy_coordinates = (alphas_bar_sqrt * coordinates + one_minus_alphas_bar_sqrt * noise) * atoms_mask.unsqueeze(-1).float()


        # Forward pass through the model
        predicted_coordinates, predicted_atom_type_logits, selfies_logits, \
        selfies_context_embeds, mol_3d_embeds, per_atom_features = self.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            atom_vec=atom_vec,
            coordinates=noisy_coordinates, # Use noisy coordinates as input
            atoms_mask=atoms_mask,
            timesteps=timesteps,
            # 移除 image_input_ids, image_attention_mask
        )

        # 1D (SELFIES/Text) to 3D Generation Task
        if task_type == '1d_to_3d':
            # 1. 坐标预测损失 (MSE Loss)
            # 我们预测的是 x_0，所以与 true_coordinates 比较
            coords_loss = F.mse_loss(
                predicted_coordinates * atoms_mask.unsqueeze(-1).float(),
                true_coordinates * atoms_mask.unsqueeze(-1).float(),
                reduction='sum'
            ) / (atoms_mask.sum().float() + 1e-5)
            losses['coords_loss'] = coords_loss

            # 2. 原子类型预测损失 (Cross-Entropy Loss)
            atom_type_logits_flat = predicted_atom_type_logits[atoms_mask].contiguous().view(-1, self.config.num_atom_types)
            true_atom_vec_flat = true_atom_vec[atoms_mask].contiguous().view(-1)
            atom_type_loss = F.cross_entropy(
                atom_type_logits_flat,
                true_atom_vec_flat,
                ignore_index=0, # Assuming atom ID 0 is padding/unknown
                reduction='mean'
            )
            losses['atom_type_loss'] = atom_type_loss

            # 3. SELFIES 预测损失 (Masked Cross-Entropy Loss)
            if self.config.selfies_coeff > 0 and true_selfies_labels is not None:
                selfies_loss = F.cross_entropy(
                    selfies_logits.view(-1, selfies_logits.size(-1)),
                    true_selfies_labels.view(-1),
                    ignore_index=-100, # Ignore padding and unmasked tokens (if -100 used)
                    reduction='mean'
                )
                losses['selfies_loss'] = selfies_loss

            # 4. 对齐损失 (Alignment Loss)
            if self.config.alignment_coeff > 0:
                # Project mol_3d_embeds to LLM hidden dim if needed for comparison
                projected_mol_embeds = self.mol_embed_projection_for_alignment(mol_3d_embeds)
                
                # Using MSE as a simple alignment loss.
                alignment_loss = F.mse_loss(selfies_context_embeds, projected_mol_embeds)
                losses['alignment_loss'] = alignment_loss

            # 5. 分层对齐 (Hierarchical Alignment Loss)
            if self.config.hierarchical_coeff > 0:
                # This would typically involve:
                # 1. Getting more granular features from LLM for SELFIES (e.g., token-level features)
                # 2. Getting `per_atom_features` from Molecular3DEncoder (already returned by `forward`)
                # 3. Defining how these granular features align (e.g., via attention, graph matching, etc.)
                # hierarchical_loss = calculate_hierarchical_loss(llm_token_features, per_atom_features)
                # losses['hierarchical_loss'] = hierarchical_loss
                pass # Implementation depends on specific definition of "hierarchical alignment"


        # Combine all losses with their coefficients
        total_loss = torch.tensor(0.0, device=coordinates.device)
        if not losses:
            raise ValueError(f"No loss calculated for task_type: {task_type}. Check your task_type logic and data.")

        for loss_name, loss_value in losses.items():
            coeff = getattr(self.config, loss_name.replace('_loss', '_coeff'), 1.0)
            total_loss += coeff * loss_value
        
        return total_loss, losses