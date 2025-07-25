import os
import json
import itertools
from pathlib import Path
from typing import Any, Dict, List
from omegaconf import OmegaConf

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from tqdm.auto import tqdm
import transformers
import yaml
import wandb

from models.modeling_mmada import MMadaConfig, MMadaModelLM
from models.lr_schedulers import get_scheduler
from parquet.my_dataset import MolecularUnifiedDataset
from training.utils import get_noise_schedule, flatten_omega_conf

logger = get_logger(__name__)


@torch.no_grad()
def generate_for_evaluation(
    model: MMadaModelLM,
    tokenizer: Any,
    prompts: List[str],
    device: torch.device,
    generation_timesteps: int,
    mask_schedule_coords: Any,
    max_atoms: int,
    max_selfies_length: int,
) -> List[Dict[str, Any]]:
    """
    Minimal stub. Assumes model implements `generate_molecule`.
    """
    try:
        import selfies as sf
    except ImportError:
        sf = None

    model.eval()
    outputs = []
    from rdkit import Chem

    for prompt in prompts:
        enc = tokenizer([prompt], return_tensors="pt").to(device)
        sample = model.generate_molecule(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_selfies_len=max_selfies_length,
            max_atoms=max_atoms,
            diffusion_timesteps=generation_timesteps,
            mask_schedule=mask_schedule_coords,
        )

        selfies_str = sample.get("selfies", "")
        smiles = None
        mol = None
        if sf is not None and selfies_str:
            try:
                smiles = sf.decoder(selfies_str)
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                mol = None
        outputs.append({**sample, "prompt": prompt, "selfies": selfies_str, "smiles": smiles, "mol": mol})
    model.train()
    return outputs


@torch.no_grad()
def run_periodic_evaluation(
    model: MMadaModelLM,
    tokenizer: Any,
    cfg: Any,
    accelerator: Accelerator,
    step: int,
    generated: List[Dict[str, Any]],
) -> None:
    """
    Minimal placeholder. Replace with your real metric implementation.
    """
    model.eval()
    try:
        total = len(generated)
        valid = sum(1 for g in generated if g.get("mol") is not None)
        uniq_smiles = {g["smiles"] for g in generated if g.get("smiles")}
        scores = {
            "validity": valid / max(1, total),
            "uniqueness": len(uniq_smiles) / max(1, valid),
            "diversity": len(uniq_smiles) / max(1, total),
        }
        wandb.log({f"eval/{k}": v for k, v in scores.items()}, step=step)
    except Exception as e:
        accelerator.print(f"[EVAL] metric calculation failed: {e}")
    model.train()


def flatten_dict(d: Dict, parent: str = "", sep: str = ".") -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main() -> None:
    with open("configs/mmada_pretraining_stage2_llada_instruct.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    class C:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, C(v) if isinstance(v, dict) else v)

    args = C(cfg_dict)

    accelerator = Accelerator(
        mixed_precision=args.training.mixed_precision,
        log_with="wandb",
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        # You can set micro batch size here if needed
        pass

    if accelerator.is_main_process:
        flat_cfg = OmegaConf.to_container(OmegaConf.create(cfg_dict), resolve=True)
        wandb.init(
            project=args.experiment.project,
            name=args.experiment.name,
            config=flat_cfg,
            resume="allow",
        )

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.training.seed is not None:
        set_seed(args.training.seed)

    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.llm_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added_mask = False
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        added_mask = True
    mask_token_id = tokenizer.mask_token_id

    # model config
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

    model = MMadaModelLM(model_cfg)
    if added_mask:
        model.llm_backbone.resize_token_embeddings(len(tokenizer))

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

    if getattr(args.training, "overfit_one_batch", False):
        first_batch = next(iter(dataloader))
        dataloader = itertools.repeat(first_batch)
        args.training.max_train_steps = 200
        accelerator.print("[DEBUG] one-batch overfit mode ON | steps: 200")

    global_step = 0

    if args.experiment.resume_from_checkpoint:
        outdir = Path(args.experiment.output_dir)
        ckpts = [p for p in outdir.iterdir() if p.name.startswith("checkpoint-")]
        if ckpts:
            latest = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))[-1]
            logger.info(f"Resuming from {latest}")
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
                total_loss, losses = model.forward_process(**inputs)
                accelerator.backward(total_loss)
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
                wandb.log(log_d, step=global_step)

                if global_step % args.experiment.eval_every == 0:
                    decoded = tokenizer.batch_decode(batch["selfies_input_ids"], skip_special_tokens=True)
                    prompts = decoded[:8]
                    gen_list = generate_for_evaluation(
                        accelerator.unwrap_model(model),
                        tokenizer,
                        prompts=prompts,
                        device=accelerator.device,
                        generation_timesteps=args.model.diffusion_timesteps,
                        mask_schedule_coords=mask_schedule_coords,
                        max_atoms=args.model.max_atoms,
                        max_selfies_length=args.model.max_selfies_length,
                    )
                    run_periodic_evaluation(model, tokenizer, args, accelerator, global_step, gen_list)

                if global_step % args.experiment.save_every == 0:
                    ckpt_dir = Path(args.experiment.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(str(ckpt_dir))
                    with open(ckpt_dir / "metadata.json", "w") as f:
                        json.dump({"global_step": global_step}, f)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")

            global_step += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = Path(args.experiment.output_dir) / "final_checkpoint"
        accelerator.save_state(str(final_dir))
        with open(final_dir / "metadata.json", "w") as f:
            json.dump({"global_step": global_step}, f)
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
