from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .common_modules import MLP, SinusoidalPositionalEmbedding
from .modeling_llada import LLaDAModelLM


def _pairwise_distances(coords: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return padded upper-triangular distance vectors per batch and a mask of valid entries.

    Args:
        coords: (B, N, 3) float – coordinates.
        mask:   (B, N) bool   – valid-atom mask.

    Returns:
        padded:     (B, M) float – distances padded to max M in batch.
        valid_mask: (B, M) bool  – which entries are valid.
    """
    B, N, _ = coords.shape
    dists: List[torch.Tensor] = []

    for b in range(B):
        diff = coords[b].unsqueeze(1) - coords[b].unsqueeze(0)       # (N, N, 3)
        dist = (diff.square().sum(-1) + 1e-6).sqrt()                         # (N, N)
        valid = mask[b].unsqueeze(1) & mask[b].unsqueeze(0)         # (N, N)
        keep = torch.triu(torch.ones_like(valid), 1).bool() & valid
        dists.append(dist[keep])                                    # (M_b,)

    max_M = max(d.size(0) for d in dists)
    device = coords.device

    padded = coords.new_zeros((B, max_M))
    valid_mask = torch.zeros((B, max_M), dtype=torch.bool, device=device)

    for b, vec in enumerate(dists):
        m = vec.size(0)
        padded[b, :m] = vec
        valid_mask[b, :m] = True

    return padded, valid_mask


def _kabsch_align(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Align `pred` to `gt` via Kabsch algorithm (per batch).

    Args:
        pred: (B, N, 3)
        gt:   (B, N, 3)
        mask: (B, N) bool

    Returns:
        aligned pred: (B, N, 3)
    """
    B, N, _ = pred.shape
    centroid_pred = (pred * mask.unsqueeze(-1)).sum(1, keepdim=True) / (mask.sum(1, keepdim=True) + 1e-8)
    centroid_gt = (gt * mask.unsqueeze(-1)).sum(1, keepdim=True) / (mask.sum(1, keepdim=True) + 1e-8)

    P = pred - centroid_pred
    G = gt - centroid_gt

    C = torch.matmul(P.transpose(2, 1), G)  # (B, 3, 3)
    V, S, W = torch.svd(C)
    d = torch.det(torch.matmul(W, V.transpose(2, 1))).unsqueeze(-1).unsqueeze(-1)
    D = torch.diag_embed(torch.ones(B, 3, device=pred.device))
    D[:, -1, -1] = d.squeeze()
    R = torch.matmul(torch.matmul(W, D), V.transpose(2, 1))

    return torch.matmul(P, R) + centroid_gt


class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(
        self,
        llm_config_path: str = "llada-8b-instruct",
        llm_model_name_or_path: str = None,
        mol_atom_embedding_dim: int = 128,
        mol_coord_embedding_dim: int = 128,
        mol_3d_encoder_output_dim: int = 768,
        num_atom_types: int = 120,
        max_atoms: int = 256,
        max_selfies_length: int = 256,
        output_atom_coords_dim: int = 3,
        output_atom_type_dim: int = 120,
        d_model: int = 768,
        fusion_hidden_dim: int = 2048,
        final_condition_dim: int = 768,
        diffusion_timesteps: int = 1000,
        noise_schedule_beta_start: float = 0.0001,
        noise_schedule_beta_end: float = 0.02,
        lm_coeff: float = 1.0,
        inv_coeff: float = 1.0,
        distance_coeff: float = 1.0,
        coords_coeff: float = 1.0,
        diff_coeff: float = 0.0,
        mae_coeff: float = 0.0,
        atom_type_coeff: float = 1.0,
        alignment_coeff: float = 1.0,
        hierarchical_coeff: float = 0.0,
        mask_token_id: int = 120,
        mask_replace_ratio: float = 0.1,
        mask_schedule_name: str = "linear",
        mask_schedule_start: float = 0.0001,
        mask_schedule_end: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        params = {k: v for k, v in locals().items() if k not in {"self", "__class__"}}
        for k, v in params.items():
            setattr(self, k, v)


class Molecular3DEncoder(nn.Module):
    def __init__(self, config: MMadaConfig):
        super().__init__()
        self.atom_embedding = nn.Embedding(config.num_atom_types, config.mol_atom_embedding_dim)
        self.coord_projection = nn.Linear(config.output_atom_coords_dim, config.mol_coord_embedding_dim)
        combined_dim = config.mol_atom_embedding_dim + config.mol_coord_embedding_dim
        self.per_atom_mlp = MLP(combined_dim, combined_dim * 2, config.mol_3d_encoder_output_dim, 2)
        self.position_embeddings = SinusoidalPositionalEmbedding(config.mol_3d_encoder_output_dim, init_range=config.max_atoms)

    def forward(
        self,
        atom_vec: torch.LongTensor,
        coordinates: torch.FloatTensor,
        atoms_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_embeds = self.atom_embedding(atom_vec)
        coord_embeds = self.coord_projection(coordinates.float()).to(atom_embeds.dtype)
        combined = torch.cat([atom_embeds, coord_embeds], dim=-1)
        per_atom = self.per_atom_mlp(combined)
        pos_ids = torch.arange(per_atom.size(1), device=per_atom.device).unsqueeze(0)
        per_atom = per_atom + self.position_embeddings(pos_ids)
        masked = per_atom * atoms_mask.unsqueeze(-1).float()
        mol_embed = masked.sum(1) / (atoms_mask.sum(1, keepdim=True).float() + 1e-5)
        return mol_embed, per_atom


class MMadaModelLM(PreTrainedModel):
    config_class = MMadaConfig
    base_model_prefix = "model"

    def __init__(self, config: MMadaConfig, tokenizer: Optional[AutoTokenizer] = None):
        super().__init__(config)
        self.llm_backbone = LLaDAModelLM.from_pretrained(config.llm_model_name_or_path)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name_or_path, use_fast=True)
        self.tokenizer = tokenizer
        if self.tokenizer.mask_token_id is None:
            self.tokenizer.add_special_tokens({"mask_token": "<mask>"})
        self.llm_backbone.resize_token_embeddings(len(self.tokenizer))

        for p in self.llm_backbone.parameters():
            p.requires_grad = False

        self.molecular_3d_encoder = Molecular3DEncoder(config)
        fusion_input_dim = config.d_model + config.mol_3d_encoder_output_dim
        self.multimodal_fusion_mlp = MLP(fusion_input_dim, config.fusion_hidden_dim, config.final_condition_dim, 2)
        self.coordinates_prediction_head = nn.Linear(config.final_condition_dim, config.max_atoms * config.output_atom_coords_dim)
        self.atom_type_prediction_head = nn.Linear(config.final_condition_dim, config.max_atoms * config.output_atom_type_dim)
        if config.mol_3d_encoder_output_dim != config.d_model:
            self.mol_embed_projection_for_alignment = nn.Linear(config.mol_3d_encoder_output_dim, config.d_model)
        else:
            self.mol_embed_projection_for_alignment = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.coordinates_prediction_head.weight)
        nn.init.zeros_(self.coordinates_prediction_head.bias)
        nn.init.zeros_(self.atom_type_prediction_head.weight)
        nn.init.zeros_(self.atom_type_prediction_head.bias)
        if isinstance(self.mol_embed_projection_for_alignment, nn.Linear):
            nn.init.xavier_uniform_(self.mol_embed_projection_for_alignment.weight)
            nn.init.zeros_(self.mol_embed_projection_for_alignment.bias)

    def forward(
        self,
        selfies_input_ids: torch.LongTensor,
        selfies_attention_mask: torch.LongTensor,
        atom_vec: torch.LongTensor,
        coordinates: torch.FloatTensor,
        atoms_mask: torch.BoolTensor,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if text_input_ids is not None:
            input_ids = torch.cat([selfies_input_ids, text_input_ids], dim=1)
            attention_mask = torch.cat([selfies_attention_mask, text_attention_mask], dim=1)
        else:
            input_ids = selfies_input_ids
            attention_mask = selfies_attention_mask

        vocab_size = self.llm_backbone.get_input_embeddings().weight.size(0)
        for name, tensor in [("selfies_input_ids", selfies_input_ids)] + ([("text_input_ids", text_input_ids)] if text_input_ids is not None else []):
            if tensor is not None and (tensor >= vocab_size).any():
                bad = tensor[tensor >= vocab_size].unique().tolist()[:5]
                raise ValueError(f"[{name}] contains invalid token IDs >= {vocab_size}: {bad}...")

        llm_out = self.llm_backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden = llm_out.hidden_states[-1]
        selfies_len = selfies_input_ids.size(1)
        selfies_hidden = hidden[:, :selfies_len, :]
        selfies_ctx = (selfies_hidden * selfies_attention_mask.unsqueeze(-1).float()).sum(1) / (selfies_attention_mask.sum(1, keepdim=True).float() + 1e-5)
        selfies_logits = llm_out.logits[:, :selfies_len, :]

        mol_embed, per_atom_feats = self.molecular_3d_encoder(atom_vec, coordinates, atoms_mask)
        fused = torch.cat([selfies_ctx, mol_embed], dim=-1)
        cond = self.multimodal_fusion_mlp(fused)
        B = cond.size(0)
        pred_coords = self.coordinates_prediction_head(cond).view(B, self.config.max_atoms, self.config.output_atom_coords_dim)
        pred_atom_logits = self.atom_type_prediction_head(cond).view(B, self.config.max_atoms, self.config.output_atom_type_dim)

        return {
            "selfies_logits": selfies_logits,
            "predicted_coordinates": pred_coords,
            "predicted_atom_type_logits": pred_atom_logits,
            "selfies_context_embeds": selfies_ctx,
            "mol_3d_embeds": mol_embed,
            "per_atom_features": per_atom_feats,
            "hidden_states": hidden if output_hidden_states else None,
            "llm_output_logits": llm_out.logits,
        }

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
        global_step: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = coordinates.size(0)
        device = coordinates.device

        if timesteps is None:
            timesteps = torch.randint(0, self.config.diffusion_timesteps, (B,), device=device).long()
        timesteps = timesteps.view(-1)

        out = self.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            atom_vec=atom_vec,
            coordinates=coordinates,
            atoms_mask=atoms_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        pred_coords: torch.Tensor = out["predicted_coordinates"]
        pred_atom_logits: torch.Tensor = out["predicted_atom_type_logits"]
        selfies_logits: torch.Tensor = out["selfies_logits"]

        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=device)

        if self.config.lm_coeff and true_selfies_labels is not None:
            lm_loss = F.cross_entropy(
                selfies_logits.reshape(-1, selfies_logits.size(-1)),
                true_selfies_labels.reshape(-1),
                ignore_index=-100
            )
            losses["lm_loss"] = lm_loss * self.config.lm_coeff

        if self.config.diff_coeff:
            sqrt_alpha_bar = mask_schedule_coords(timesteps.long()).clamp(0.0, 1.0).view(-1, 1, 1)
            alpha_bar = (sqrt_alpha_bar ** 2).clamp(0.0, 1.0)
            sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar).clamp(min=1e-8))

            noise_gt = torch.randn_like(true_coordinates)
            pos_0 = true_coordinates
            pos_t = sqrt_alpha_bar * pos_0 + sqrt_one_minus_alpha_bar * noise_gt

            noise_pred = pred_coords - pos_t  # keep your current head behavior
            mask3 = atoms_mask.unsqueeze(-1).float()
            diff_mse = F.mse_loss(noise_pred * mask3, noise_gt * mask3, reduction="sum") / (mask3.sum() + 1e-5)
            losses["diff_loss"] = diff_mse * self.config.diff_coeff


        if self.config.mae_coeff and ("pred_props" in out) and ("true_props" in kwargs):
            mae_loss = F.l1_loss(out["pred_props"], kwargs["true_props"], reduction="mean")
            losses["mae_loss"] = mae_loss * self.config.mae_coeff

        if self.config.atom_type_coeff:
            valid_idx = atoms_mask
            if valid_idx.any():
                flat_logits = pred_atom_logits[valid_idx].view(-1, pred_atom_logits.size(-1))
                flat_labels = true_atom_vec[valid_idx].view(-1)
                at_loss = F.cross_entropy(flat_logits, flat_labels)
            else:
                at_loss = torch.tensor(0.0, device=device)
            losses["atom_type_loss"] = at_loss * self.config.atom_type_coeff

        if not losses:
            return total_loss, {}
        total_loss = sum(losses.values())
        return total_loss, losses


    @torch.no_grad()
    def generate_molecule(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_selfies_len: int,
        max_atoms: int,
        diffusion_timesteps: int,
        mask_schedule: Callable[..., torch.Tensor],
        tokenizer: Any,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_beams: int = 1,
    ) -> Dict[str, Any]:
        out_ids = self.llm_backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_selfies_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
        )
        seq = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        smiles, mol = None, None
        try:
            import selfies as sf, rdkit.Chem as Chem
            smiles = sf.decoder(seq)
            mol = Chem.MolFromSmiles(smiles)
        except:
            pass
        return {"selfies": seq, "smiles": smiles, "mol": mol}


AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
