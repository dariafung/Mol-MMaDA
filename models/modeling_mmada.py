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
from training.training_utils import get_noise_schedule


def _pairwise_distances(coords: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = coords.shape
    dists: List[torch.Tensor] = []
    for b in range(B):
        diff = coords[b].unsqueeze(1) - coords[b].unsqueeze(0)
        dist = (diff.square().sum(-1) + 1e-6).sqrt()
        valid = mask[b].unsqueeze(1) & mask[b].unsqueeze(0)
        keep = torch.triu(torch.ones_like(valid), 1).bool() & valid
        dists.append(dist[keep])
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
    B, N, _ = pred.shape
    centroid_pred = (pred * mask.unsqueeze(-1)).sum(1,
                                                    keepdim=True) / (mask.sum(1, keepdim=True) + 1e-8)
    centroid_gt = (gt * mask.unsqueeze(-1)).sum(1, keepdim=True) / \
        (mask.sum(1, keepdim=True) + 1e-8)
    P = pred - centroid_pred
    G = gt - centroid_gt
    C = torch.matmul(P.transpose(2, 1), G)
    V, S, W = torch.svd(C)
    d = torch.det(torch.matmul(W, V.transpose(2, 1))
                  ).unsqueeze(-1).unsqueeze(-1)
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
        inv_coeff: float = 0.0,
        distance_coeff: float = 0.0,
        coords_coeff: float = 0.0,
        diff_coeff: float = 0.0,
        mae_coeff: float = 0.0,
        atom_type_coeff: float = 1.0,
        alignment_coeff: float = 0.0,
        hierarchical_coeff: float = 0.0,
        mask_token_id: int = 120,
        mask_replace_ratio: float = 0.1,
        mask_schedule_name: str = "linear",
        mask_schedule_start: float = 0.0001,
        mask_schedule_end: float = 0.02,
        num_scalar_props: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        params = {k: v for k, v in locals().items() if k not in {
            "self", "__class__"}}
        for k, v in params.items():
            setattr(self, k, v)


class Molecular3DEncoder(nn.Module):
    def __init__(self, config: MMadaConfig):
        super().__init__()
        self.atom_embedding = nn.Embedding(
            config.num_atom_types, config.mol_atom_embedding_dim)
        self.coord_projection = nn.Linear(
            config.output_atom_coords_dim, config.mol_coord_embedding_dim)
        combined_dim = config.mol_atom_embedding_dim + config.mol_coord_embedding_dim
        self.per_atom_mlp = MLP(
            combined_dim, combined_dim * 2, config.mol_3d_encoder_output_dim, 2)
        self.position_embeddings = SinusoidalPositionalEmbedding(
            config.mol_3d_encoder_output_dim, init_range=config.max_atoms
        )

    def forward(
        self,
        atom_vec: torch.LongTensor,
        coordinates: torch.FloatTensor,
        atoms_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_embeds = self.atom_embedding(atom_vec)
        coord_embeds = self.coord_projection(
            coordinates.float()).to(atom_embeds.dtype)
        combined = torch.cat([atom_embeds, coord_embeds], dim=-1)
        per_atom = self.per_atom_mlp(combined)
        pos_ids = torch.arange(per_atom.size(
            1), device=per_atom.device).unsqueeze(0)
        per_atom = per_atom + self.position_embeddings(pos_ids)
        masked = per_atom * atoms_mask.unsqueeze(-1).float()
        mol_embed = masked.sum(
            1) / (atoms_mask.sum(1, keepdim=True).float() + 1e-5)
        return mol_embed, per_atom


class MMadaModelLM(PreTrainedModel):
    config_class = MMadaConfig
    base_model_prefix = "model"

    def __init__(self, config: MMadaConfig, tokenizer: Optional[AutoTokenizer] = None):
        super().__init__(config)
        self.llm_backbone = LLaDAModelLM.from_pretrained(
            config.llm_model_name_or_path)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.llm_model_name_or_path, use_fast=True)
        self.tokenizer = tokenizer
        if self.tokenizer.mask_token_id is None:
            self.tokenizer.add_special_tokens({"mask_token": "<mask>"})
        self.llm_backbone.resize_token_embeddings(len(self.tokenizer))

        for p in self.llm_backbone.parameters():
            p.requires_grad = False

        self.molecular_3d_encoder = Molecular3DEncoder(config)
        fusion_input_dim = config.d_model + config.mol_3d_encoder_output_dim
        self.multimodal_fusion_mlp = MLP(
            fusion_input_dim, config.fusion_hidden_dim, config.final_condition_dim, 2)

        # timestep conditioning
        self.timestep_sinus = SinusoidalPositionalEmbedding(
            config.final_condition_dim, init_range=config.diffusion_timesteps
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(config.final_condition_dim, config.final_condition_dim),
            nn.SiLU(),
            nn.Linear(config.final_condition_dim, config.final_condition_dim),
        )

        self.coordinates_prediction_head = nn.Linear(
            config.final_condition_dim, config.max_atoms * config.output_atom_coords_dim
        )
        self.atom_type_prediction_head = nn.Linear(
            config.final_condition_dim, config.max_atoms * config.output_atom_type_dim
        )
        self.properties_head = nn.Linear(
            config.final_condition_dim, config.num_scalar_props)

        if config.mol_3d_encoder_output_dim != config.d_model:
            self.mol_embed_projection_for_alignment = nn.Linear(
                config.mol_3d_encoder_output_dim, config.d_model)
        else:
            self.mol_embed_projection_for_alignment = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.coordinates_prediction_head.weight)
        nn.init.zeros_(self.coordinates_prediction_head.bias)
        nn.init.zeros_(self.atom_type_prediction_head.weight)
        nn.init.zeros_(self.atom_type_prediction_head.bias)
        nn.init.xavier_uniform_(self.properties_head.weight)
        nn.init.zeros_(self.properties_head.bias)
        if isinstance(self.mol_embed_projection_for_alignment, nn.Linear):
            nn.init.xavier_uniform_(
                self.mol_embed_projection_for_alignment.weight)
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
        timesteps: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        selfies_input_ids = selfies_input_ids.long()
        selfies_attention_mask = selfies_attention_mask.long()
        if text_input_ids is not None:
            text_input_ids = text_input_ids.long()
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.long()

        if text_input_ids is not None:
            input_ids = torch.cat([selfies_input_ids, text_input_ids], dim=1)
            attention_mask = torch.cat(
                [selfies_attention_mask, text_attention_mask], dim=1)
        else:
            input_ids = selfies_input_ids
            attention_mask = selfies_attention_mask

        vocab_size = self.llm_backbone.get_input_embeddings().weight.size(0)
        for name, tensor in [("selfies_input_ids", selfies_input_ids)] + (
            [("text_input_ids", text_input_ids)
             ] if text_input_ids is not None else []
        ):
            if tensor is not None and (tensor >= vocab_size).any():
                bad = tensor[tensor >= vocab_size].unique().tolist()[:5]
                raise ValueError(
                    f"[{name}] contains invalid token IDs >= {vocab_size}: {bad}...")

        llm_out = self.llm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = llm_out.hidden_states[-1]
        selfies_len = selfies_input_ids.size(1)
        selfies_hidden = hidden[:, :selfies_len, :]
        selfies_ctx = (selfies_hidden * selfies_attention_mask.unsqueeze(-1).float()).sum(1) / (
            selfies_attention_mask.sum(1, keepdim=True).float() + 1e-5
        )
        selfies_logits = llm_out.logits[:, :selfies_len, :]

        mol_embed, per_atom_feats = self.molecular_3d_encoder(
            atom_vec, coordinates, atoms_mask)
        fused = torch.cat([selfies_ctx, mol_embed], dim=-1)
        cond = self.multimodal_fusion_mlp(fused)

        if timesteps is not None:
            t_ids = timesteps.view(-1, 1).to(cond.device)
            t_emb = self.timestep_sinus(t_ids).squeeze(1)
            t_emb = self.timestep_mlp(t_emb)
            cond = cond + t_emb

        B = cond.size(0)
        pred_coords = self.coordinates_prediction_head(cond).view(
            B, self.config.max_atoms, self.config.output_atom_coords_dim
        )
        pred_atom_logits = self.atom_type_prediction_head(cond).view(
            B, self.config.max_atoms, self.config.output_atom_type_dim
        )
        pred_props = self.properties_head(cond)

        return {
            "selfies_logits": selfies_logits,
            # interpreted as epsilon in diffusion
            "predicted_coordinates": pred_coords,
            "predicted_atom_type_logits": pred_atom_logits,
            "pred_props": pred_props,
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
            timesteps = torch.randint(
                0, self.config.diffusion_timesteps, (B,), device=device).long()
        timesteps = timesteps.view(-1)

        # Construct x_t and train epsilon prediction
        sqrt_alpha_bar = mask_schedule_coords(
            timesteps.long()).clamp(0.0, 1.0).view(-1, 1, 1)
        alpha_bar = (sqrt_alpha_bar ** 2).clamp(0.0, 1.0)
        sqrt_one_minus_alpha_bar = torch.sqrt(
            (1.0 - alpha_bar).clamp(min=1e-8))
        noise_gt = torch.randn_like(true_coordinates)
        pos_0 = true_coordinates
        pos_t = sqrt_alpha_bar * pos_0 + sqrt_one_minus_alpha_bar * noise_gt

        out = self.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            atom_vec=atom_vec,
            coordinates=pos_t,                # feed x_t
            atoms_mask=atoms_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            timesteps=timesteps,
            output_hidden_states=True,
            return_dict=True,
        )

        pred_eps: torch.Tensor = out["predicted_coordinates"]
        pred_atom_logits: torch.Tensor = out["predicted_atom_type_logits"]
        selfies_logits: torch.Tensor = out["selfies_logits"]
        pred_props: torch.Tensor = out.get("pred_props", None)

        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=device)

        if self.config.lm_coeff and true_selfies_labels is not None:
            lm_loss = F.cross_entropy(
                selfies_logits.reshape(-1, selfies_logits.size(-1)),
                true_selfies_labels.reshape(-1),
                ignore_index=-100,
            )
            losses["lm_loss"] = lm_loss * self.config.lm_coeff

        if self.config.diff_coeff:
            mask3 = atoms_mask.unsqueeze(-1).float()
            diff_mse = F.mse_loss(
                pred_eps * mask3, noise_gt * mask3, reduction="sum") / (mask3.sum() + 1e-5)
            losses["diff_loss"] = diff_mse * self.config.diff_coeff

        if self.config.mae_coeff and (pred_props is not None) and ("true_props" in kwargs):
            true_props: torch.Tensor = kwargs["true_props"]
            if true_props is not None and true_props.numel() > 0:
                if true_props.dim() == 1:
                    true_props = true_props.unsqueeze(0)
                mask = ~torch.isnan(true_props)
                if mask.any():
                    mae_loss = F.l1_loss(
                        pred_props[mask], true_props[mask], reduction="mean")
                    losses["mae_loss"] = mae_loss * self.config.mae_coeff

        if self.config.atom_type_coeff:
            valid_idx = atoms_mask
            if valid_idx.any():
                flat_logits = pred_atom_logits[valid_idx].view(
                    -1, pred_atom_logits.size(-1))
                flat_labels = true_atom_vec[valid_idx].view(-1)
                at_loss = F.cross_entropy(flat_logits, flat_labels)
            else:
                at_loss = torch.tensor(0.0, device=device)
            losses["atom_type_loss"] = at_loss * self.config.atom_type_coeff

        # Add simple chemical constraint loss
        if hasattr(self.config, 'chemical_coeff') and self.config.chemical_coeff > 0:
            chemical_loss = self._compute_chemical_constraint_loss(
                true_coordinates, true_atom_vec, atoms_mask)
            losses["chemical_loss"] = chemical_loss * \
                self.config.chemical_coeff

        if not losses:
            return total_loss, {}
        total_loss = sum(losses.values())
        return total_loss, losses

    def _compute_chemical_constraint_loss(self, coordinates: torch.Tensor, atom_types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Simple chemical constraint loss to encourage reasonable bond lengths"""
        batch_size, max_atoms = coordinates.shape[:2]
        total_loss = 0.0
        total_pairs = 0

        for b in range(batch_size):
            valid_mask = mask[b]
            if valid_mask.sum() < 2:
                continue

            valid_coords = coordinates[b][valid_mask]
            n_atoms = valid_coords.shape[0]

            # Check distances between all atom pairs
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dist = torch.norm(valid_coords[i] - valid_coords[j])

                    # Penalize distances that are too close (< 0.5) or too far (> 5.0)
                    if dist < 0.5:
                        total_loss += (0.5 - dist) ** 2
                    elif dist > 5.0:
                        total_loss += (dist - 5.0) ** 2

                    total_pairs += 1

        return total_loss / max(total_pairs, 1)

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
        seq = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        smiles, mol = None, None
        try:
            import selfies as sf
            import rdkit.Chem as Chem
            smiles = sf.decoder(seq)
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            pass
        return {"selfies": seq, "smiles": smiles, "mol": mol}

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        max_atoms: int,
        device: torch.device,
        num_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        B, K = batch_size, max_atoms

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        selfies_input_ids = torch.full(
            (B, 1), pad_id, dtype=torch.long, device=device)
        selfies_attention_mask = torch.ones(
            (B, 1), dtype=torch.long, device=device)

        text_input_ids = None
        text_attention_mask = None

        atom_vec = torch.zeros(B, K, dtype=torch.long, device=device)
        coordinates = torch.zeros(B, K, 3, dtype=torch.float32, device=device)
        atoms_mask = torch.ones(B, K, dtype=torch.bool, device=device)

        out = self.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            atom_vec=atom_vec,
            coordinates=coordinates,
            atoms_mask=atoms_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            timesteps=None,
            output_hidden_states=False,
            return_dict=True,
        )

        logits = out["predicted_atom_type_logits"]
        coords = out["predicted_coordinates"]

        K_conf = logits.size(1)
        if K > K_conf:
            raise ValueError(
                f"requested max_atoms={K} exceeds model config.max_atoms={K_conf}")
        if K < K_conf:
            logits = logits[:, :K, :]
            coords = coords[:, :K, :]

        atom_idx = logits.argmax(dim=-1)
        mask = atom_idx > 0

        return {"atom_idx": atom_idx, "coords": coords, "mask": mask}

    @torch.no_grad()
    def sample_diffusion(
        self,
        batch_size: int,
        max_atoms: int,
        device: torch.device,
        t_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        type_conf_frac: float = 0.15,
        type_topk: int = 3,
        temperature: float = 1.0,
        scheduler_name: Optional[str] = None,
        predicts_eps: bool = True,
        commit_class0: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        B, K = batch_size, max_atoms
        T = int(t_steps or self.config.diffusion_timesteps)

        atom_idx = torch.zeros(B, K, dtype=torch.long, device=device)
        x_t = torch.randn(
            B, K, self.config.output_atom_coords_dim, device=device)
        atoms_mask = torch.ones(B, K, dtype=torch.bool, device=device)

        sqrt_ab_fn = get_noise_schedule(
            name=(scheduler_name or getattr(
                self.config, "noise_schedule_name", "linear")),
            beta_start=self.config.noise_schedule_beta_start,
            beta_end=self.config.noise_schedule_beta_end,
            timesteps=T,
            device=device,
        )

        for step in range(T - 1, -1, -1):
            t = torch.full((B,), step, dtype=torch.long, device=device)

            pred = self._predict_step(
                atom_idx=atom_idx,
                x_t=x_t,
                atoms_mask=atoms_mask,
                t=t,
                guidance_scale=guidance_scale,
                temperature=temperature,
                predicts_eps=predicts_eps,
            )
            type_logits = pred["type_logits"]
            # epsilon if predicts_eps=True else x0
            coord_pred = pred["coord_pred"]

            sqrt_a_t = sqrt_ab_fn(t).clamp(0.0, 1.0).view(B, 1, 1)
            a_t = (sqrt_a_t ** 2).clamp(0.0, 1.0)
            sqrt_one_minus_a_t = torch.sqrt((1.0 - a_t).clamp(min=1e-8))

            if predicts_eps:
                x0 = (x_t - sqrt_one_minus_a_t * coord_pred) / \
                    sqrt_a_t.clamp_min(1e-8)
                eps_pred = coord_pred
            else:
                x0 = coord_pred
                eps_pred = (x_t - sqrt_a_t * x0) / \
                    sqrt_one_minus_a_t.clamp_min(1e-8)

            if step > 0:
                t_prev = torch.full(
                    (B,), step - 1, dtype=torch.long, device=device)
                sqrt_a_prev = sqrt_ab_fn(t_prev).clamp(0.0, 1.0).view(B, 1, 1)
                a_prev = (sqrt_a_prev ** 2).clamp(0.0, 1.0)
                x_t = torch.sqrt(
                    a_prev) * x0 + torch.sqrt((1.0 - a_prev).clamp(min=1e-8)) * eps_pred
            else:
                x_t = x0

            undecided = (atom_idx == 0) & atoms_mask
            if undecided.any():
                logits = type_logits
                if not commit_class0:
                    logits = logits.clone()
                    logits[..., 0] = logits[..., 0] - 1e9

                probs = F.softmax(
                    logits / max(1e-6, float(temperature)), dim=-1)
                conf, argmax = probs.max(dim=-1)

                U = int(undecided.sum().item())
                n_decode = max(1, int(type_conf_frac * U))
                conf_vals = conf[undecided]
                _, topos = torch.topk(conf_vals, k=n_decode)
                undec_idx = torch.nonzero(undecided, as_tuple=False)
                pick = undec_idx[topos]

                if type_topk and type_topk > 1:
                    tk = min(type_topk, probs.size(-1))
                    topv, topk_idx = torch.topk(
                        probs[pick[:, 0], pick[:, 1], :], k=tk, dim=-1)
                    topv = topv / \
                        topv.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    rel = torch.multinomial(topv, num_samples=1).squeeze(-1)
                    sampled = topk_idx[torch.arange(
                        topk_idx.size(0), device=device), rel]
                    atom_idx[pick[:, 0], pick[:, 1]] = sampled
                else:
                    atom_idx[pick[:, 0], pick[:, 1]
                             ] = argmax[pick[:, 0], pick[:, 1]]

        return {"atom_idx": atom_idx, "coords": x_t, "mask": atoms_mask}

    def _predict_step(
        self,
        atom_idx: torch.Tensor,
        x_t: torch.Tensor,
        atoms_mask: torch.Tensor,
        t: torch.Tensor,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        predicts_eps: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B, K = atom_idx.shape
        device = x_t.device

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        selfies_input_ids = torch.full(
            (B, 1), pad_id, dtype=torch.long, device=device)
        selfies_attention_mask = torch.ones(
            (B, 1), dtype=torch.long, device=device)

        out = self.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            atom_vec=atom_idx,
            coordinates=x_t,
            atoms_mask=atoms_mask,
            text_input_ids=None,
            text_attention_mask=None,
            timesteps=t,
            output_hidden_states=False,
            return_dict=True,
        )
        type_logits = out["predicted_atom_type_logits"][:, :K, :]
        pred_coords = out["predicted_coordinates"][:, :K, :]

        coord_pred = pred_coords.to(x_t.dtype)
        return {"type_logits": type_logits, "coord_pred": coord_pred}

    def _compact_to_onehot(self, atom_idx: torch.Tensor) -> torch.Tensor:
        C = int(self.config.output_atom_type_dim)
        B, K = atom_idx.shape
        onehot = torch.zeros(
            B, K, C, device=atom_idx.device, dtype=torch.float32)
        onehot.scatter_(
            dim=-1, index=atom_idx.unsqueeze(-1).clamp_(0, C - 1), value=1.0)
        return onehot


AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
