import os
import json
import itertools
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import transformers
import yaml
import wandb

from models.modeling_mmada import MMadaConfig, MMadaModelLM
from models.lr_schedulers import get_scheduler
from parquet.my_dataset import MolecularUnifiedDataset
from training.utils import get_noise_schedule
from generate import generate_for_evaluation
from evaluation.eval_functions import get_3D_edm_metric

logger = get_logger(__name__)


def flatten_dict(d: dict, parent: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@torch.no_grad()
def run_periodic_evaluation(
    model,
    tokenizer,
    cfg,
    accelerator,
    step: int,
    pre_generated,
):
    model.eval()
    try:
        scores, _ = get_3D_edm_metric(
            pre_generated,
            dataset_name=getattr(cfg.model, "dataset_name", "QM9"),
        )
        wandb.log({f"eval/{k}": v for k, v in scores.items()}, step=step)
    except Exception as e:
        accelerator.print(f"[EVAL] metric calculation failed: {e}")
    model.train()


def main() -> None:
    # ----------------- load yaml -----------------
    with open("configs/mmada_pretraining_stage2_llada_instruct.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    class C:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, C(v) if isinstance(v, dict) else v)

    args = C(cfg_dict)

    # ----------------- accelerator / wandb -----------------
    accelerator = Accelerator(
        mixed_precision=args.training.mixed_precision,
        log_with="wandb",
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        wandb.init(
            project="mol-mmada-stage2",
            name=args.experiment.name,
            config=flatten_dict(cfg_dict),
            resume="allow",
        )

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    set_seed(args.training.seed)

    # ----------------- tokenizer first -----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model.llm_model_name_or_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added_mask = False
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        added_mask = True
    mask_token_id = tokenizer.mask_token_id
    accelerator.print(f"[DEBUG] mask_token_id = {mask_token_id}")

    # ----------------- build config with real mask_token_id -----------------
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
        coords_coeff=args.model.coords_coeff,
        atom_type_coeff=args.model.atom_type_coeff,
        selfies_coeff=args.model.selfies_coeff,
        alignment_coeff=args.model.alignment_coeff,
        hierarchical_coeff=args.model.hierarchical_coeff,
        mask_token_id=mask_token_id,
        mask_replace_ratio=args.model.mask_replace_ratio,
        mask_schedule_name=args.model.mask_schedule_name,
        mask_schedule_start=args.model.mask_schedule_start,
        mask_schedule_end=args.model.mask_schedule_end,
    )

    # ----------------- create model -----------------
    model = MMadaModelLM(model_cfg)

    if added_mask:
        model.llm_backbone.resize_token_embeddings(len(tokenizer))

    # ----------------- dataset & dataloader -----------------
    dataset = MolecularUnifiedDataset(
        data_path=args.model.data_path,
        tokenizer=tokenizer,
        max_text_length=args.dataset.preprocessing.max_text_length,
        max_selfies_length=args.model.max_selfies_length,
        max_atoms=args.model.max_atoms,
        mask_token_id=mask_token_id,
        diffusion_timesteps=model_cfg.diffusion_timesteps,
        mask_schedule_name=model_cfg.mask_schedule_name,
        mask_schedule_start=model_cfg.mask_schedule_start,
        mask_schedule_end=model_cfg.mask_schedule_end,
        selfies_mask_ratio=args.model.selfies_mask_ratio,
        atom_type_mask_prob=args.model.atom_type_mask_prob,
        include_edge_bond_dist=args.model.include_edge_bond_dist,
        include_rdmol2selfies=args.model.include_rdmol2selfies,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        shuffle=True,
        repeat=True,
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

    # ----------------- optimizer / scheduler -----------------
    optimizer = AdamW(
        model.parameters(),
        lr=float(args.optimizer.params.learning_rate),
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

    # ----------------- optional: overfit one batch -----------------
    if getattr(args.training, "overfit_one_batch", False):
        first_batch = next(iter(dataloader))
        dataloader = itertools.repeat(first_batch)
        args.training.max_train_steps = 200
        accelerator.print(
            "[DEBUG] one-batch overfit mode ON  |  steps:", args.training.max_train_steps
        )

    accelerator.print(">>> max_train_steps =", args.training.max_train_steps)
    progress = tqdm(
        range(args.training.max_train_steps),
        disable=not accelerator.is_main_process,
        desc="Training",
    )

    # ----------------- training loop -----------------
    global_step = 0
    while global_step < args.training.max_train_steps:
        for batch in dataloader:
            if global_step >= args.training.max_train_steps:
                break

            model.global_step = global_step
            model.train()

            inputs = {
                "selfies_input_ids": batch["selfies_input_ids"],
                "selfies_attention_mask": batch["selfies_attention_mask"],
                "atom_vec": batch["atom_vec"],
                "coordinates": batch["coordinates"],
                "atoms_mask": batch["atoms_mask"],
                "timesteps": batch["timesteps"],
                "task_type": "1d_to_3d",
                "true_coordinates": batch["true_coordinates"],
                "true_atom_vec": batch["true_atom_vec"],
                "true_selfies_labels": batch["true_selfies_labels"],
                "mask_schedule_coords": mask_schedule_coords,
            }

            with accelerator.accumulate(model):
                tot_loss, losses = model.forward_process(**inputs)
                accelerator.backward(tot_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                progress.update(1)
                global_step += 1

                log_d = {
                    "train/total_loss": tot_loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                }
                log_d.update({f"train/{k}": v.item() for k, v in losses.items()})
                wandb.log(log_d, step=global_step)

                if global_step % args.experiment.eval_every == 0:
                    decoded = tokenizer.batch_decode(
                        batch["selfies_input_ids"], skip_special_tokens=True
                    )
                    prompts = decoded[:8]

                    gen_mols = generate_for_evaluation(
                        accelerator.unwrap_model(model),
                        tokenizer,
                        prompts=prompts,
                        device=accelerator.device,
                    )
                    print(f"[DBG] step={global_step} generated {len(gen_mols)} molecules")

                    run_periodic_evaluation(
                        model, tokenizer, args, accelerator, global_step, gen_mols
                    )

                if global_step % args.experiment.save_every == 0:
                    ckpt_dir = os.path.join(
                        args.experiment.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(ckpt_dir)
                    accelerator.print(f"Saved checkpoint to {ckpt_dir}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.experiment.output_dir, "final_checkpoint")
        accelerator.save_state(final_dir)
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
