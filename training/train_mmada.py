import os
import sys

# Ensure the project root is in the path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.modeling_mmada import MMadaModelLM, MMadaConfig
from models.lr_schedulers import get_scheduler
from parquet.my_dataset import MolecularUnifiedDataset
from selfies import get_semantic_robust_alphabet
from training.utils import get_noise_schedule

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    # ----------------- load yaml -----------------
    cli = parse_args()
    with open(cli.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    args = OmegaConf.create(cfg_dict)

    # ----------------- Initialize Accelerator and WandB -----------------
    accelerator = Accelerator(
        mixed_precision=args.training.mixed_precision,
        log_with="wandb",
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        project_dir=Path(args.experiment.output_dir) / "logs",
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        flat_cfg = OmegaConf.to_container(args, resolve=True)
        accelerator.init_trackers(
            project_name=args.experiment.project,
            config=flat_cfg,
            init_kwargs={"wandb": {"name": args.experiment.name, "resume": "allow"}},
        )

    if args.training.seed is not None:
        set_seed(args.training.seed)

    torch.autograd.set_detect_anomaly(True)

    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # ----------------- tokenizer -----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.llm_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    selfies_alphabet = set(get_semantic_robust_alphabet())
    existing = set(tokenizer.get_vocab().keys())
    to_add = list(selfies_alphabet - existing)
    if to_add:
        tokenizer.add_tokens(to_add)
        accelerator.print(f"Added {len(to_add)} SELFIES tokens.")

    selfies_mask_token_id = tokenizer.mask_token_id
    if selfies_mask_token_id is None:
        if tokenizer.pad_token_id is not None:
            selfies_mask_token_id = tokenizer.pad_token_id
            accelerator.print("No mask token. Using pad token as mask.")
        else:
            raise ValueError("No mask or pad token available.")

    # ----------------- model config -----------------
    model_cfg = MMadaConfig(
        llm_config_path=args.model.llm_config_path,
        llm_model_name_or_path=args.model.llm_model_name_or_path,
        mol_atom_embedding_dim=args.model.mol_atom_embedding_dim,
        mol_coord_embedding_dim=args.model.mol_coord_embedding_dim,
        mol_3d_encoder_output_dim=args.model.mol_3d_encoder_output_dim,
        num_atom_types=args.model.num_atom_types,
        max_atoms=args.model.max_atoms,
        max_selfies_length=args.model.max_selfies_length,
        output_atom_coords_dim=args.model.output_atom_coords_dim,
        output_atom_type_dim=args.model.output_atom_type_dim,
        d_model=args.model.d_model,
        fusion_hidden_dim=args.model.fusion_hidden_dim,
        final_condition_dim=args.model.final_condition_dim,
        diffusion_timesteps=args.model.diffusion_timesteps,
        noise_schedule_beta_start=args.model.noise_schedule_beta_start,
        noise_schedule_beta_end=args.model.noise_schedule_beta_end,
        noise_schedule_name=args.model.noise_schedule_name,

        # Active loss coefficients
        lm_coeff=getattr(args.model, "lm_coeff", 1.0),
        diff_coeff=getattr(args.model, "diff_coeff", 0.0),
        mae_coeff=getattr(args.model, "mae_coeff", 0.0),
        atom_type_coeff=getattr(args.model, "atom_type_coeff", 1.0),

        # QM9 scalar properties (mu, alpha, homo, lumo, gap, cv)
        num_scalar_props=6,

        mask_token_id=args.model.mask_token_id,
        mask_replace_ratio=args.model.mask_replace_ratio,
        mask_schedule_name=args.model.mask_schedule_name,
        mask_schedule_start=args.model.mask_schedule_start,
        mask_schedule_end=args.model.mask_schedule_end,

        vocab_size=len(tokenizer),
        embedding_size=((len(tokenizer) + 127) // 128) * 128,
    )

    # For logging raw losses
    loss_coeff: Dict[str, float] = {
        "lm_loss": model_cfg.lm_coeff,
        "diff_loss": model_cfg.diff_coeff,
        "mae_loss": model_cfg.mae_coeff,
        "atom_type_loss": model_cfg.atom_type_coeff,
    }

    model = MMadaModelLM(model_cfg, tokenizer=tokenizer)

    for p in model.llm_backbone.parameters():
        p.requires_grad = False

    # Optionally unfreeze last k blocks of LLaDA
    k = 2
    inner = model.llm_backbone.model if hasattr(model.llm_backbone, "model") else model.llm_backbone
    tr = inner.transformer

    if hasattr(tr, "blocks"):
        transformer_layers = list(tr.blocks)
    else:
        transformer_layers = [layer for g in tr.block_groups for layer in g]

    for layer in transformer_layers[-k:]:
        for p in layer.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    accelerator.print(
        f"Trainable params: {trainable/1e6:.2f} M / {total/1e6:.2f} M "
        f"({trainable/total:.2%}) — Unfrozen last {k} LLaDA blocks"
    )

    # ----------------- dataset / dataloader -----------------
    dataset = MolecularUnifiedDataset(
        data_path=args.model.data_path,
        tokenizer=tokenizer,
        max_text_length=args.dataset.preprocessing.max_text_length,
        max_selfies_length=args.model.max_selfies_length,
        max_atoms=args.model.max_atoms,
        mask_token_id=selfies_mask_token_id,
        diffusion_timesteps=model_cfg.diffusion_timesteps,
        mask_schedule_name=model_cfg.mask_schedule_name,
        mask_schedule_start=model_cfg.mask_schedule_start,
        mask_schedule_end=model_cfg.mask_schedule_end,
        selfies_mask_ratio=args.model.selfies_mask_ratio,
        atom_type_mask_prob=args.model.atom_type_mask_prob,
        include_edge_bond_dist=args.model.include_edge_bond_dist,
        include_rdmol2selfies=args.model.include_rdmol2selfies,
        shuffle=getattr(args.training, "shuffle", True),
        repeat=getattr(args.training, "repeat", True),
        buffer_size=args.dataset.params.shuffle_buffer_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.training.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=args.dataset.params.num_workers,
        pin_memory=args.dataset.params.pin_memory,
        persistent_workers=args.dataset.params.persistent_workers,
    )

    base_lr = float(args.optimizer.params.learning_rate)

    backbone_trainable = [p for p in model.llm_backbone.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_trainable}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]

    optimizer = AdamW(
        [
            {"params": other_params, "lr": base_lr},
            {"params": backbone_trainable, "lr": base_lr * 2},
        ],
        weight_decay=float(args.optimizer.params.weight_decay),
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.params.warmup_steps,
        num_training_steps=args.training.max_train_steps,
        min_lr_scale=getattr(args.lr_scheduler.params, "min_lr_scale", 0.1),
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    mask_schedule_coords = get_noise_schedule(
        name=model_cfg.noise_schedule_name,
        beta_start=model_cfg.noise_schedule_beta_start,
        beta_end=model_cfg.noise_schedule_beta_end,
        timesteps=model_cfg.diffusion_timesteps,
        device=accelerator.device,
    )

    # ----------------- optional overfit -----------------
    if getattr(args.training, "overfit_one_batch", False):
        first_batch = next(iter(dataloader))
        import itertools as _itertools
        dataloader = _itertools.repeat(first_batch)
        args.training.max_train_steps = 200
        accelerator.print("One-batch overfit mode ON | steps: 200")

    global_step = 0

    if args.experiment.resume_from_checkpoint:
        outdir = Path(args.experiment.output_dir)
        ckpts = [d for d in outdir.iterdir() if d.name.startswith("checkpoint-")]
        if ckpts:
            latest = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))[-1]
            accelerator.print(f"Resuming from {latest}")
            accelerator.load_state(latest)
            meta_file = latest / "metadata.json"
            if meta_file.exists():
                with meta_file.open() as f:
                    meta = json.load(f)
                global_step = meta.get("global_step", 0)

    progress = tqdm(range(args.training.max_train_steps), disable=not accelerator.is_main_process, desc="Training")

    while global_step < args.training.max_train_steps:
        for batch in dataloader:
            if global_step >= args.training.max_train_steps:
                break

            model.train()

            if model_cfg.mae_coeff > 0 and ("true_props" not in batch):
                raise RuntimeError("mae_coeff > 0 but batch['true_props'] is missing from the dataloader output.")

            inputs = {
                "selfies_input_ids": batch["selfies_input_ids"],
                "selfies_attention_mask": batch["selfies_attention_mask"],
                "atom_vec": batch["atom_vec"],
                "coordinates": batch["coordinates"],
                "atoms_mask": batch["atoms_mask"],
                "timesteps": batch["timesteps"],
                "task_type": "pretraining",
                "true_coordinates": batch["true_coordinates"],
                "true_atom_vec": batch["true_atom_vec"],
                "true_selfies_labels": batch["true_selfies_labels"],
                "mask_schedule_coords": mask_schedule_coords,
                "text_input_ids": batch["text_input_ids"],
                "text_attention_mask": batch["text_attention_mask"],
                "global_step": global_step,
                "true_props": batch.get("true_props", None),
            }

            with accelerator.accumulate(model):
                total_loss, losses = model.forward_process(**inputs)
                accelerator.backward(total_loss)

                if args.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                progress.update(1)
                log_d = {
                    "train/total_loss": float(total_loss.item()),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                }
                log_d.update({f"train/{k}": float(v.item()) for k, v in losses.items()})
                for k, v in losses.items():
                    coeff = loss_coeff.get(k, 1.0)
                    log_d[f"train/{k}_raw"] = float((v / max(coeff, 1e-12)).item())
                accelerator.log(log_d, step=global_step)

                if global_step % args.experiment.save_every == 0:
                    ckpt_dir = Path(args.experiment.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(str(ckpt_dir))
                    with open(ckpt_dir / "metadata.json", "w") as f:
                        json.dump({"global_step": global_step}, f)
                    accelerator.print(f"Saved checkpoint {ckpt_dir}")

            global_step += 1

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = Path(args.experiment.output_dir) / "final_checkpoint"
        accelerator.save_state(str(final_dir))
        with open(final_dir / "metadata.json", "w") as f:
            json.dump({"global_step": global_step}, f)
        accelerator.end_training()


if __name__ == "__main__":
    main()
