import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing import Optional, Callable, Dict, Any, Tuple

from .modeling_llada import LLaDAModelLM
from .common_modules import MLP, SinusoidalPositionalEmbedding
# from training.utils import get_mask_schedule # Not used in this file


class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(
        self,
        # LLM backbone config
        llm_config_path: str = "llada-8b-instruct",
        llm_model_name_or_path: str = None,

        # Molecular 3D Encoder
        mol_atom_embedding_dim: int = 128,
        mol_coord_embedding_dim: int = 128,
        mol_3d_encoder_output_dim: int = 768, # Should match LLM hidden size or be projected
        num_atom_types: int = 120, # Example: up to U, with 0 for padding
        max_atoms: int = 256, # Max atoms in a molecule
        max_selfies_length: int = 256,
        output_atom_coords_dim: int = 3, # x, y, z coordinates
        output_atom_type_dim: int = 120, # Number of atom types for classification

        # Fusion network parameters 
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
        selfies_coeff: float = 1.0, 
        alignment_coeff: float = 1.0, 
        hierarchical_coeff: float = 0.0, 

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
        self.mol_atom_embedding_dim = mol_atom_embedding_dim
        self.mol_coord_embedding_dim = mol_coord_embedding_dim
        self.mol_3d_encoder_output_dim = mol_3d_encoder_output_dim
        self.num_atom_types = num_atom_types
        self.max_atoms = max_atoms
        self.max_selfies_length = max_selfies_length 
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

class Molecular3DEncoder(nn.Module):
    def __init__(self, config: MMadaConfig):
        super().__init__()
        self.config = config
        self.atom_embedding = nn.Embedding(config.num_atom_types, config.mol_atom_embedding_dim)
        self.coord_projection = nn.Linear(config.output_atom_coords_dim, config.mol_coord_embedding_dim)
        
        combined_atom_feat_dim = config.mol_atom_embedding_dim + config.mol_coord_embedding_dim
        
        self.per_atom_mlp = MLP(
            input_dim=combined_atom_feat_dim,
            hidden_dim=combined_atom_feat_dim * 2,
            output_dim=config.mol_3d_encoder_output_dim,
            num_layers=2
        )
        self.position_embeddings = SinusoidalPositionalEmbedding(
            config.mol_3d_encoder_output_dim, init_range=config.max_atoms
        )

    def forward(self, atom_vec: torch.LongTensor, coordinates: torch.FloatTensor, atoms_mask: torch.BoolTensor):
        coordinates = coordinates.float()
        atom_embeds = self.atom_embedding(atom_vec)
        coord_embeds = self.coord_projection(coordinates)
        coord_embeds = coord_embeds.to(atom_embeds.dtype)
        
        combined_embeds = torch.cat([atom_embeds, coord_embeds], dim=-1)
        per_atom_features = self.per_atom_mlp(combined_embeds)
        
        position_ids = torch.arange(per_atom_features.shape[1], device=per_atom_features.device).unsqueeze(0)
        per_atom_features = per_atom_features + self.position_embeddings(position_ids)

        masked_features = per_atom_features * atoms_mask.unsqueeze(-1).float()
        molecular_embedding = masked_features.sum(dim=1) / (atoms_mask.sum(dim=1, keepdim=True).float() + 1e-5)
        
        return molecular_embedding, per_atom_features


# --- 3. MMadaModelLM 类 ---
class MMadaModelLM(nn.Module):
    def __init__(self, config: MMadaConfig):
        super().__init__()
        self.config = config

        self.llm_backbone = LLaDAModelLM.from_pretrained(config.llm_model_name_or_path)

        self.molecular_3d_encoder = Molecular3DEncoder(config)

        fusion_input_dim = config.d_model + config.mol_3d_encoder_output_dim
        self.multimodal_fusion_mlp = MLP(
            input_dim=fusion_input_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.final_condition_dim,
            num_layers=2
        )

        self.coordinates_prediction_head = nn.Linear(
            config.final_condition_dim,
            config.max_atoms * config.output_atom_coords_dim
        )
        self.atom_type_prediction_head = nn.Linear(
            config.final_condition_dim,
            config.max_atoms * config.output_atom_type_dim
        )

        if config.mol_3d_encoder_output_dim != config.d_model:
            self.mol_embed_projection_for_alignment = nn.Linear(config.mol_3d_encoder_output_dim, config.d_model)
        else:
            self.mol_embed_projection_for_alignment = nn.Identity()

    def forward(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        atom_vec: torch.LongTensor,
        coordinates: torch.FloatTensor,
        atoms_mask: torch.BoolTensor,
        text_input_ids: Optional[torch.LongTensor] = None,     
        text_attention_mask: Optional[torch.LongTensor] = None, 
        timesteps: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        if text_input_ids is not None and text_attention_mask is not None:
             combined_input_ids = torch.cat([selfies_input_ids, text_input_ids], dim=1)
             combined_attention_mask = torch.cat([selfies_attention_mask, text_attention_mask], dim=1)
        else:
             combined_input_ids = selfies_input_ids
             combined_attention_mask = selfies_attention_mask
        
        # 1. Process SELFIES input using LLM backbone
        freeze_llm = getattr(self.config, "freeze_llm", False)
        if freeze_llm:
            with torch.no_grad():
                llm_output = self.llm_backbone(
                    input_ids=combined_input_ids,
                    attention_mask=combined_attention_mask,
                    output_hidden_states=True
                )
        else:
            llm_output = self.llm_backbone(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                output_hidden_states=True
            )
        hidden_states = llm_output.hidden_states[-1]

        selfies_len = selfies_input_ids.shape[1]
        selfies_hidden_states = hidden_states[:, :selfies_len, :]
        selfies_context_embeds = (selfies_hidden_states * selfies_attention_mask.unsqueeze(-1).float()).sum(dim=1) / \
                                (selfies_attention_mask.sum(dim=1, keepdim=True).float() + 1e-5)

        # 2. Process 3D molecular input
        mol_3d_embeds, per_atom_features = self.molecular_3d_encoder(atom_vec, coordinates, atoms_mask)

        # 3. Multimodal fusion
        fused_features = torch.cat([selfies_context_embeds, mol_3d_embeds], dim=-1)
        final_condition_embeds = self.multimodal_fusion_mlp(fused_features)

        # 4. Prediction heads
        B = final_condition_embeds.size(0)
        predicted_coordinates = self.coordinates_prediction_head(final_condition_embeds) \
                .view(B, self.config.max_atoms, self.config.output_atom_coords_dim)
        predicted_atom_type_logits = self.atom_type_prediction_head(final_condition_embeds) \
                .view(B, self.config.max_atoms, self.config.output_atom_type_dim)

        selfies_logits = llm_output.logits[:, :selfies_len, :]
        
        return predicted_coordinates, predicted_atom_type_logits, selfies_logits, selfies_context_embeds, mol_3d_embeds, per_atom_features

    def forward_process(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        atom_vec: torch.LongTensor,
        coordinates: torch.FloatTensor,
        atoms_mask: torch.BoolTensor,
        task_type: str,
        true_coordinates: torch.FloatTensor,
        true_atom_vec: torch.LongTensor,
        mask_schedule_coords: Callable,
        text_input_ids: Optional[torch.LongTensor] = None,  
        text_attention_mask: Optional[torch.LongTensor] = None, 
        true_selfies_labels: Optional[torch.LongTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size = coordinates.shape[0]
        losses = {}

        if timesteps is None:
            timesteps = torch.randint(0, self.config.diffusion_timesteps, (batch_size,), device=coordinates.device).long()
        
        timesteps = timesteps.view(-1)
        noise = torch.randn_like(coordinates) * atoms_mask.unsqueeze(-1).float()
        
        alphas_bar_sqrt = mask_schedule_coords(timesteps).unsqueeze(-1).unsqueeze(-1)
        one_minus_alphas_bar_sqrt = (1.0 - alphas_bar_sqrt**2).sqrt()

        noisy_coordinates = (alphas_bar_sqrt * coordinates + one_minus_alphas_bar_sqrt * noise) * atoms_mask.unsqueeze(-1).float()

        # Forward pass
        predicted_coordinates, predicted_atom_type_logits, selfies_logits, \
        selfies_context_embeds, mol_3d_embeds, per_atom_features = self.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            atom_vec=atom_vec,
            coordinates=noisy_coordinates,
            atoms_mask=atoms_mask,
            timesteps=timesteps,
        )

        if task_type == '1d_to_3d':

            coords_loss = F.mse_loss(
                predicted_coordinates * atoms_mask.unsqueeze(-1).float(),
                true_coordinates * atoms_mask.unsqueeze(-1).float(),
                reduction='sum'
            ) / (atoms_mask.sum().float() + 1e-5)
            losses['coords_loss'] = coords_loss

            atom_type_logits_flat = predicted_atom_type_logits[atoms_mask].contiguous().view(-1, self.config.num_atom_types)
            true_atom_vec_flat = true_atom_vec[atoms_mask].contiguous().view(-1)
  
            valid_mask = (true_atom_vec_flat != 0)          

            if valid_mask.sum() == 0:
                atom_type_loss = torch.tensor(0.0, device=coordinates.device)
            else:
                atom_type_loss = F.cross_entropy(
                    atom_type_logits_flat[valid_mask],
                    true_atom_vec_flat[valid_mask],
                    reduction='mean'
                )

            losses['atom_type_loss'] = atom_type_loss


            if self.global_step % 200 == 0 and self.training:
                print(f"[DBG] step={self.global_step}  valid_atom_type_labels="
                    f"{valid_mask.sum().item()}/{valid_mask.numel()}")
            if self.global_step % 200 == 0 and self.training:
                valid_selfies = (true_selfies_labels != -100).sum().item()
                print(f"[DBG] step={self.global_step}  valid_selfies_tokens={valid_selfies}/{true_selfies_labels.numel()}")
                
            # 3. SELFIES 损失
            if self.config.selfies_coeff > 0 and true_selfies_labels is not None:
                selfies_loss = F.cross_entropy(
                    selfies_logits.reshape(-1, selfies_logits.size(-1)),
                    true_selfies_labels.reshape(-1),
                    ignore_index=-100,
                    reduction='mean'
                )
                losses['selfies_loss'] = selfies_loss

            # 4. 对齐损失
            if self.config.alignment_coeff > 0:
                projected_mol_embeds = self.mol_embed_projection_for_alignment(mol_3d_embeds)
                alignment_loss = F.mse_loss(selfies_context_embeds, projected_mol_embeds)
                losses['alignment_loss'] = alignment_loss

        total_loss = torch.tensor(0.0, device=coordinates.device)
        if not losses:
            # 返回一个零损失和空字典，而不是报错，这样训练可以继续
            return total_loss, {}

        for loss_name, loss_value in losses.items():
            coeff = getattr(self.config, loss_name.replace('_loss', '_coeff'), 1.0)
            total_loss += coeff * loss_value
        
        return total_loss, losses