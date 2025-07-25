import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, List
from omegaconf import DictConfig, ListConfig, OmegaConf

import yaml
import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from rdkit import Chem
import transformers

from models.modeling_mmada import MMadaModelLM, MMadaConfig
from models.lr_schedulers import get_scheduler

from parquet.my_dataset import MolecularUnifiedDataset
from selfies import get_semantic_robust_alphabet
from training.utils import flatten_omega_conf, get_noise_schedule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_periodic_evaluation(
    model: MMadaModelLM,
    tokenizer: Any,
    cfg: Any,
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
        logger.info(f"[EVAL] scores @ step {step}: {scores}")
    except Exception as e:
        logger.warning(f"[EVAL] metric calculation failed: {e}")
    model.train()


@torch.no_grad()
def generate_for_evaluation(
    model: MMadaModelLM,
    tokenizer: Any,
    prompts: List[str],
    device: torch.device,
    generation_timesteps: int,
    mask_schedule_coords: Callable,
    max_atoms: int,
    max_selfies_length: int,
) -> List[Dict[str, Any]]:
    """
    Minimal stub. Assumes model implements `generate_molecule`.
    """
    model.eval()
    results = []
    try:
        import selfies as sf
    except ImportError:
        sf = None

    for p in prompts:
        enc = tokenizer([p], return_tensors="pt").to(device)
        sample = model.generate_molecule(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_selfies_len=max_selfies_length,
            max_atoms=max_atoms,
            diffusion_timesteps=generation_timesteps,
            mask_schedule=mask_schedule_coords,
        )  # You must implement this method in MMadaModelLM

        selfies_str = sample.get("selfies", "")
        smiles = None
        mol = None
        if sf is not None and selfies_str:
            try:
                smiles = sf.decoder(selfies_str)
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                mol = None
        results.append({**sample, "prompt": p, "selfies": selfies_str, "smiles": smiles, "mol": mol})
    model.train()
    return results


def main() -> None:
    # ----------------- load yaml -----------------
    with open("configs/mmada_pretraining_stage1_llada_instruct.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    class C:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, C(v) if isinstance(v, dict) else v)

    args = C(cfg_dict)

    # -----------------  wandb -----------------
    
    flat_cfg = OmegaConf.to_container(OmegaConf.create(cfg_dict), resolve=True)
    wandb.init(
        project=args.experiment.project,
        name=args.experiment.name,
        config=flat_cfg,
        resume="allow",
    )

    transformers.utils.logging.set_verbosity_warning()
    
    if args.training.seed is not None:
        torch.manual_seed(args.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ----------------- tokenizer -----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.llm_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added_mask_token_to_tokenizer = False
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        added_mask_token_to_tokenizer = True

    selfies_alphabet = set(get_semantic_robust_alphabet())
    existing = set(tokenizer.get_vocab().keys())
    to_add = list(selfies_alphabet - existing)
    if to_add:
        tokenizer.add_tokens(to_add)
        logger.info(f"Added {len(to_add)} SELFIES tokens.")

    selfies_mask_token_id = tokenizer.mask_token_id
    if selfies_mask_token_id is None:
        if tokenizer.pad_token_id is not None:
            selfies_mask_token_id = tokenizer.pad_token_id
            logger.warning("No mask token. Using pad token as mask.")
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
        coords_coeff=args.model.coords_coeff,
        atom_type_coeff=args.model.atom_type_coeff,
        selfies_coeff=args.model.selfies_coeff,
        alignment_coeff=args.model.alignment_coeff,
        hierarchical_coeff=args.model.hierarchical_coeff,
        mask_token_id=args.model.mask_token_id,
        mask_replace_ratio=args.model.mask_replace_ratio,
        mask_schedule_name=args.model.mask_schedule_name,
        mask_schedule_start=args.model.mask_schedule_start,
        mask_schedule_end=args.model.mask_schedule_end,
        vocab_size=len(tokenizer),
        embedding_size=((len(tokenizer) + 127) // 128) * 128,
    )

    model = MMadaModelLM(model_cfg)
    model.to(device)

    if added_mask_token_to_tokenizer or to_add:
        model.llm_backbone.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}.")

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

    mask_schedule_coords = get_noise_schedule(
        name=model_cfg.noise_schedule_name,
        beta_start=model_cfg.noise_schedule_beta_start,
        beta_end=model_cfg.noise_schedule_beta_end,
        timesteps=model_cfg.diffusion_timesteps,
        device=device,
    )

    # ----------------- optional overfit -----------------
    if getattr(args.training, "overfit_one_batch", False):
        first_batch = next(iter(dataloader))
        dataloader = itertools.repeat(first_batch)
        args.training.max_train_steps = 200
        logger.info("One-batch overfit mode ON | steps: 200")

    global_step = 0

    if args.experiment.resume_from_checkpoint:
        outdir = Path(args.experiment.output_dir)
        ckpts = [d for d in outdir.iterdir() if d.name.startswith("checkpoint-")]
        if ckpts:
            latest = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))[-1]
            logger.info(f"Resuming from {latest}")
            model.load_state_dict(torch.load(latest / "pytorch_model.bin", map_location=device))
            if (latest / "optimizer.bin").exists():
                optimizer.load_state_dict(torch.load(latest / "optimizer.bin", map_location=device))
            
            meta_file = latest / "metadata.json"
            if meta_file.exists():
                with meta_file.open() as f:
                    meta = json.load(f)
                global_step = meta.get("global_step", 0)

    progress = tqdm(range(args.training.max_train_steps), disable=False, desc="Training") 

    scaler = None
    if args.training.mixed_precision == "fp16": 
        scaler = torch.cuda.amp.GradScaler()
    elif args.training.mixed_precision == "bf16":
        pass 

    while global_step < args.training.max_train_steps:
        for batch in dataloader:
            if global_step >= args.training.max_train_steps:
                break

            model.train()

            inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                      for k, v in {
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
                      }.items()}

            optimizer.zero_grad()

            with torch.autocast(
                device_type=device.type, 
                dtype=torch.bfloat16 if args.training.mixed_precision == "bf16" else torch.float16, 
                enabled=args.training.mixed_precision != "no" and device.type == "cuda"
            ):
                total_loss, losses = model.forward_process(**inputs)

            if scaler is not None: 
                scaler.scale(total_loss).backward()
                if args.training.max_grad_norm is not None:
                    scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else: 
                total_loss.backward()
                if args.training.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.training.max_grad_norm)
                optimizer.step()
            
            lr_scheduler.step()

            progress.update(1)
            log_d = {
                "train/total_loss": float(total_loss.item()),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
            }
            log_d.update({f"train/{k}": float(v.item()) for k, v in losses.items()})
            wandb.log(log_d, step=global_step)

            if global_step % args.experiment.eval_every == 0:
                prompts = [
                    "Design a molecule with antibacterial activity.",
                    "Generate a molecule containing a benzene ring.",
                ]
                
                gen_list = generate_for_evaluation(
                    model,
                    tokenizer,
                    prompts=prompts,
                    device=device, 
                    generation_timesteps=args.model.diffusion_timesteps,
                    mask_schedule_coords=mask_schedule_coords,
                    max_atoms=args.model.max_atoms,
                    max_selfies_length=args.model.max_selfies_length,
                )
                run_periodic_evaluation(
                    model, 
                    tokenizer,
                    args,
                    global_step,
                    gen_list,
                )

            if global_step % args.experiment.save_every == 0:
                ckpt_dir = Path(args.experiment.output_dir) / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
                torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.bin") 
                with open(ckpt_dir / "metadata.json", "w") as f:
                    json.dump({"global_step": global_step}, f)
                logger.info(f"Saved checkpoint {ckpt_dir}")

            global_step += 1


    final_dir = Path(args.experiment.output_dir) / "final_checkpoint"
    final_dir.mkdir(parents=True, exist_ok=True) 
    torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    torch.save(optimizer.state_dict(), final_dir / "optimizer.bin")
    with open(final_dir / "metadata.json", "w") as f:
        json.dump({"global_step": global_step}, f)
    wandb.finish()


if __name__ == "__main__":
    main()
