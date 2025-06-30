import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing import Optional, Callable, Dict, Any

from .modeling_llada import LLaDAModelLM
from .common_modules import MLP, SinusoidalPositionalEmbedding
from training.utils import get_mask_schedule


# --- 1. 修改 MMadaConfig 类 ---
class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(
        self,
        # LLM backbone config
        llm_config_path: str = "llada-8b-instruct",
        llm_model_name_or_path: str = None,

        # Image tokenizer config
        image_tokenizer_path: str = "magvit-v2-tokenizer",
        image_tokenizer_config_file: str = None,
        image_tokenizer_model_file: str = None,

        # Image diffusion
        image_unet_config_path: str = "show-o-image-unet",
        image_unet_model_path: str = None,

        # Molecular 3D Encoder
        mol_atom_embedding_dim: int = 128,
        mol_coord_embedding_dim: int = 128,
        mol_3d_encoder_output_dim: int = 768, # Should match LLM hidden size or be projected
        num_atom_types: int = 120, # Example: up to U, with 0 for padding
        max_atoms: int = 256, # Max atoms in a molecule
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
        selfies_coeff: float = 0.0,  # 新增：SELFIES预测损失系数
        alignment_coeff: float = 0.0, # 新增：对齐损失系数
        hierarchical_coeff: float = 0.0, # 新增：分层对齐损失系数 (暂时设为0，后续实现)

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
        self.image_tokenizer_path = image_tokenizer_path
        self.image_tokenizer_config_file = image_tokenizer_config_file
        self.image_tokenizer_model_file = image_tokenizer_model_file
        self.image_unet_config_path = image_unet_config_path
        self.image_unet_model_path = image_unet_model_path

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
        self.selfies_coeff = selfies_coeff      # 新增
        self.alignment_coeff = alignment_coeff  # 新增
        self.hierarchical_coeff = hierarchical_coeff # 新增

        self.mask_token_id = mask_token_id
        self.mask_replace_ratio = mask_replace_ratio
        self.mask_schedule_name = mask_schedule_name
        self.mask_schedule_start = mask_schedule_start
        self.mask_schedule_end = mask_schedule_end


# --- 2. 修改 Molecular3DEncoder 类 (无需太大改动，确保输出维度匹配) ---
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


# --- 3. 修改 MMadaModelLM 类 ---
class MMadaModelLM(nn.Module):
    def __init__(self, config: MMadaConfig):
        super().__init__()
        self.config = config

        # Load LLaDA backbone
        self.llm_backbone = LLaDAModelLM.from_pretrained(config.llm_model_name_or_path)
        self.llm_backbone.eval() # Typically freeze LLM backbone for multimodal fine-tuning stages
        for param in self.llm_backbone.parameters():
            param.requires_grad = False

        # Molecular 3D Encoder
        self.molecular_3d_encoder = Molecular3DEncoder(config)

        # Multimodal fusion MLP
        # Input dim for fusion: LLM output dim (from selfies context) + Mol3D encoder output dim
        # Assuming selfies context embed size is d_model
        fusion_input_dim = config.d_model + config.mol_3d_encoder_output_dim
        self.multimodal_fusion_mlp = MLP(
            input_dim=fusion_input_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.final_condition_dim,
            num_layers=2
        )

        # Prediction heads for 3D molecular generation
        self.coordinates_prediction_head = nn.Linear(config.final_condition_dim, config.output_atom_coords_dim)
        self.atom_type_prediction_head = nn.Linear(config.final_condition_dim, config.output_atom_type_dim)

        # New: Projection layer for alignment loss if dimensions don't match
        # Assuming selfies_context_embeds (LLM d_model) and mol_3d_embeds (mol_3d_encoder_output_dim)
        # need to be projected to a common dimension for alignment. Let's project to d_model.
        if config.mol_3d_encoder_output_dim != config.d_model:
            self.mol_embed_projection_for_alignment = nn.Linear(config.mol_3d_encoder_output_dim, config.d_model)
        else:
            self.mol_embed_projection_for_alignment = nn.Identity()

        # New: SELFIES prediction head (if LLM's lm_head is not directly used for SELFIES)
        # This is for the discrete diffusion loss on SELFIES tokens.
        # Assuming the LLM's final hidden state before lm_head is used for prediction
        # The vocab size should be for the SELFIES tokenizer.
        # Here we'll re-use the LLM's lm_head for simplicity, assuming SELFIES tokens are part of LLM vocab.
        # If SELFIES vocabulary is separate, you'd need a dedicated head.
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
        **kwargs,
    ):
        batch_size = selfies_input_ids.shape[0]
        
        # 1. Process SELFIES input using LLM backbone
        # The LLM takes input_ids and attention_mask
        # We need the hidden states to get the contextual embedding for SELFIES
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


        # 2. Process 3D molecular input
        mol_3d_embeds, per_atom_features = self.molecular_3d_encoder(atom_vec, coordinates, atoms_mask)
        # mol_3d_embeds shape: (batch_size, mol_3d_encoder_output_dim)
        # per_atom_features shape: (batch_size, max_atoms, mol_3d_encoder_output_dim)


        # 3. Handle optional text input (if not a pure 1D-to-3D task)
        # For '1d_to_3d' task, text_input_ids might be None or dummy.
        # If it's a general multimodal setup, you'd concatenate or fuse text_context_embeds too.
        # For simplicity in 1d_to_3d, we assume text is not the primary condition or is empty.
        # If text is relevant as condition, add similar processing for text_input_ids here.
        
        # 4. Multimodal fusion
        fused_features = torch.cat([selfies_context_embeds, mol_3d_embeds], dim=-1)
        final_condition_embeds = self.multimodal_fusion_mlp(fused_features)
        # Shape: (batch_size, final_condition_dim)

        # 5. Prediction heads for 3D generation
        predicted_coordinates = self.coordinates_prediction_head(final_condition_embeds)
        predicted_atom_type_logits = self.atom_type_prediction_head(final_condition_embeds)

        # For SELFIES prediction (if selfies_coeff > 0)
        # We need the raw LLM logits for the SELFIES tokens *before* any pooling
        # This assumes the LLM's lm_head can correctly predict SELFIES tokens
        # If SELFIES tokens are within the LLM's original vocabulary, we can reuse lm_head.
        # Take the output logits corresponding to the SELFIES input part
        # Note: llm_selfies_output.logits is raw logits from LLM.lm_head
        selfies_logits = llm_selfies_output.logits[:, :selfies_input_ids.shape[1], :] # Only relevant part for SELFIES
        
        # Return all necessary outputs for loss calculation
        return predicted_coordinates, predicted_atom_type_logits, selfies_logits, selfies_context_embeds, mol_3d_embeds, per_atom_features


    def forward_process(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        text_input_ids: Optional[torch.LongTensor],
        text_attention_mask: Optional[torch.LongTensor],
        atom_vec: torch.LongTensor,
        coordinates: torch.FloatTensor,
        atoms_mask: torch.BoolTensor,
        task_type: str,
        true_coordinates: torch.FloatTensor,
        true_atom_vec: torch.LongTensor,
        mask_schedule_coords: Callable, # For continuous diffusion noise
        # New: true_selfies_labels for SELFIES masked prediction loss
        true_selfies_labels: Optional[torch.LongTensor] = None,
        # New: inputs for hierarchical alignment if needed (e.g., intermediate features)
        # This will be more complex and depend on how you define "hierarchical alignment"
        edge_type: Optional[torch.LongTensor] = None, # Unused for now
        bond_type: Optional[torch.LongTensor] = None, # Unused for now
        dist: Optional[torch.FloatTensor] = None, # Unused for now
        rdmol2selfies: Optional[torch.FloatTensor] = None, # Unused for now
        timesteps: Optional[torch.LongTensor] = None, # Timestep for diffusion
    ):
        batch_size = coordinates.shape[0]
        losses = {}

        # Determine diffusion timestep for continuous coordinates
        if timesteps is None:
            timesteps = torch.randint(0, self.config.diffusion_timesteps, (batch_size,), device=coordinates.device).long()
        
        # Add noise to coordinates for continuous diffusion input (x_t)
        # This follows the paper's continuous diffusion for 3D coordinates.
        noise = torch.randn_like(coordinates) * atoms_mask.unsqueeze(-1).float() # Only noise valid atoms
        alphas_bar_sqrt = mask_schedule_coords(timesteps).unsqueeze(-1).unsqueeze(-1)
        one_minus_alphas_bar_sqrt = (1.0 - alphas_bar_sqrt**2).sqrt()

        # Noisy coordinates for model input
        noisy_coordinates = (alphas_bar_sqrt * true_coordinates + one_minus_alphas_bar_sqrt * noise) * atoms_mask.unsqueeze(-1).float()


        # Apply masking to selfies_input_ids for discrete diffusion (masked token prediction)
        # This is crucial for the L_Diff-disc loss on SELFIES.
        # We need a separate masking mechanism for SELFIES tokens.
        # Assume mask_token_id is properly set in config (e.g., to LLM tokenizer's mask_token_id or a special token)
        # For now, let's use a simple random masking
        
        # NOTE: The current implementation of SELFIES input in `train_mmada_stage2.py`
        # treats them as conditioning. For L_Diff-disc, they need to be masked and predicted.
        # This part requires changes in `prepare_molecular_inputs_and_labels` as well.
        
        # For now, let's assume `selfies_input_ids` here are already potentially masked for the LLM
        # and `true_selfies_labels` holds the original unmasked tokens.
        # If selfies_coeff > 0, we treat selfies as a prediction target with masking.
        if self.config.selfies_coeff > 0 and true_selfies_labels is not None:
            # Create a mask for SELFIES tokens based on a random ratio per token for discrete diffusion
            # This is a simplified masking, a proper schedule like in training/utils.py's `get_mask_schedule`
            # and `mask_or_random_replace_tokens` should be applied, but adapted for SELFIES vocabulary.
            # Here, we'll simulate a mask directly within forward_process for demonstration.
            # A more robust masking should happen in data preparation.
            
            # This is a placeholder for actual discrete diffusion masking
            # For `L_Diff-disc`, `selfies_input_ids` should be the masked version (x_t)
            # and `true_selfies_labels` should be the original tokens (x_0)
            
            # Let's create a dummy masked selfies_input_ids here for now if it's not pre-masked.
            # In actual use, `selfies_input_ids` passed to this method should be the already masked version.
            # Here, we assume `selfies_input_ids` passed in is the 'noisy' version (x_t)
            # and `true_selfies_labels` is the 'clean' version (x_0).
            pass # Masking should ideally be handled during data preparation / input creation for forward_process


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
        )

        # 1D (SELFIES/Text) to 3D Generation Task
        if task_type == '1d_to_3d':
            # 1. 坐标预测损失 (MSE Loss)
            # We predict the noise `epsilon`, so the target for prediction is `noise`.
            # The model actually predicts x_0, so we should compare predicted_coordinates directly to true_coordinates
            # or derive noise prediction from predicted_coordinates.
            # Given the existing code structure, it's predicting the denoised x_0 directly.
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

            # 3. 新增：SELFIES 预测损失 (Masked Cross-Entropy Loss)
            # This is L_Diff-disc from the paper for discrete tokens like SELFIES.
            # We need to ensure `true_selfies_labels` contains the unmasked original SELFIES tokens.
            # And `selfies_input_ids` fed into `forward` should be the masked version.
            if self.config.selfies_coeff > 0 and true_selfies_labels is not None:
                # `selfies_logits` are from the LLM's head, predicting the next token.
                # When using L_Diff-disc, we are predicting the *masked* tokens.
                # Assuming `true_selfies_labels` has -100 for unmasked tokens (standard for HF CE loss)
                # And `selfies_logits` covers the full sequence length for SELFIES.
                
                # Filter logits and labels to only masked positions if masking happens here
                # Or assume true_selfies_labels already contains -100 where unmasked.
                selfies_loss = F.cross_entropy(
                    selfies_logits.view(-1, selfies_logits.size(-1)), # (B*L, vocab_size)
                    true_selfies_labels.view(-1), # (B*L)
                    ignore_index=-100, # Ignore padding and unmasked tokens (if -100 used)
                    reduction='mean'
                )
                losses['selfies_loss'] = selfies_loss

            # 4. 新增：对齐损失 (Alignment Loss)
            # Aligning SELFIES context embedding with Molecular 3D embedding
            if self.config.alignment_coeff > 0:
                # Project mol_3d_embeds to LLM hidden dim if needed for comparison
                projected_mol_embeds = self.mol_embed_projection_for_alignment(mol_3d_embeds)
                
                # Using MSE as a simple alignment loss.
                # More sophisticated methods like InfoNCE (contrastive loss) could be used but are more complex.
                alignment_loss = F.mse_loss(selfies_context_embeds, projected_mol_embeds)
                losses['alignment_loss'] = alignment_loss

            # 5. 新增：分层对齐 (Hierarchical Alignment Loss)
            # This is a more complex concept. For now, it's a placeholder.
            # Example: Aligning per-atom features with a corresponding fine-grained SELFIES/text representation.
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