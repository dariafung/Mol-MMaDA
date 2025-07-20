import os
import sys
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
import yaml
import wandb  # <-- 1. 导入 wandb
import json

# Import your model and config
from models.modeling_mmada import MMadaConfig, MMadaModelLM
# Import your dataset
from parquet.my_dataset import MolecularUnifiedDataset
from models.lr_schedulers import get_scheduler
# from training.optimizer import get_optimizer # 你的项目中可能有这个文件，如果没有，需要创建或修改
from torch.optim import AdamW # 使用标准的 AdamW 作为备用
from training.utils import (
    get_noise_schedule,
)
# --- 2. 导入评估和生成逻辑 ---
from generate import generate_for_evaluation # 假设 generate.py 中有这个函数
from evaluation.eval_functions import get_3D_edm_metric # 确保这个路径正确
from rdkit import Chem


logger = get_logger(__name__)

# --- 3. 添加一个辅助函数来展平配置，方便 wandb 记录 ---
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# --- 4. 将评估逻辑封装成一个函数 ---
@torch.no_grad()
def run_periodic_evaluation(model, tokenizer, config, accelerator, global_step):
    """
    在训练过程中定期运行生成和评估，并将结果记录到 W&B。
    """
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    accelerator.print(f"\nRunning evaluation at step {global_step}...")

    # A. 生成分子 (调用你的生成逻辑)
    # 假设 generate_for_evaluation 返回一个 RDKit Mol 对象的列表
    try:
        # 你需要根据 generate.py 的逻辑来调整这里
        # 这里是一个示例，假设你有一些固定的 prompts 来生成
        eval_prompts = ["CCO", "C1=CC=CC=C1"] # 用一些简单的 SELFIES/SMILES 作为评估提示
        generated_rdmols = generate_for_evaluation(unwrapped_model, tokenizer, eval_prompts, accelerator.device)
    except Exception as e:
        accelerator.print(f"Evaluation failed during generation: {e}")
        model.train()
        return

    # B. 评估生成的分子
    try:
        scores, _ = get_3D_edm_metric(
            generated_rdmols,
            dataset_name=config.model.dataset_name if hasattr(config.model, 'dataset_name') else "QM9"
        )
    except Exception as e:
        accelerator.print(f"Evaluation failed during metrics calculation: {e}")
        model.train()
        return

    # C. 记录评估分数到 W&B
    eval_log_dict = {}
    for metric_name, value in scores.items():
        eval_log_dict[f"eval/{metric_name}"] = value
    
    accelerator.print(f"Evaluation scores at step {global_step}: {json.dumps(scores, indent=2)}")

    # D. 记录一些生成的分子作为3D可视化对象到 W&B
    molecules_to_log = []
    for i, mol in enumerate(generated_rdmols[:5]): # 只记录前5个
        if mol is not None:
            try:
                caption = f"Step {global_step} - Mol {i+1}"
                pdb_string = Chem.MolToPDBBlock(mol)
                wandb_mol = wandb.Molecule(pdb_string, caption=caption, file_type='pdb')
                molecules_to_log.append(wandb_mol)
            except Exception as e:
                accelerator.print(f"Could not convert generated mol {i} to PDB for logging: {e}")

    if molecules_to_log:
        eval_log_dict["eval/generated_molecules"] = molecules_to_log

    wandb.log(eval_log_dict, step=global_step)
    accelerator.print("Evaluation results and molecules logged to W&B.")

    model.train() # 评估结束后切回训练模式


def main():
    # --- 1. 加载配置 (保持不变) ---
    config_path = "configs/mmada_pretraining_stage2_llada_instruct.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    class Config:
        def __init__(self, d):
            for a, b in d.items():
                if isinstance(b, dict):
                    setattr(self, a, Config(b))
                else:
                    setattr(self, a, b)
    
    args = Config(config_dict)

    # --- 5. 修改 Accelerator 初始化，直接与 W&B 集成 ---
    accelerator = Accelerator(
        mixed_precision=args.training.mixed_precision,
        log_with="wandb", # <--- 修改为 "wandb"
        gradient_accumulation_steps=args.training.gradient_accumulation_steps
    )
    
    # --- 6. 初始化 W&B ---
    if accelerator.is_main_process:
        # 使用你的 wandb.yaml 配置（如果存在），或者直接在这里配置
        wandb_config = flatten_dict(config_dict)
        wandb.init(
            project="mol-mmada-stage2", # 建议的项目名
            name=args.experiment.name,
            config=wandb_config,
            resume="allow"
        )

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    set_seed(args.training.seed)
    
    # ...(模型和 tokenizer 的加载逻辑保持不变)...
    # --- MMadaConfig needs to be instantiated correctly with relevant parameters ---
    model_config = MMadaConfig(
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
        noise_schedule_name=args.model.noise_schedule_name if hasattr(args.model, 'noise_schedule_name') else 'cosine',
        coords_coeff=args.model.coords_coeff,
        atom_type_coeff=args.model.atom_type_coeff,
        selfies_coeff=args.model.selfies_coeff, # Ensure this is passed from config
        alignment_coeff=args.model.alignment_coeff, # Ensure this is passed from config
        hierarchical_coeff=args.model.hierarchical_coeff, # Ensure this is passed from config
        mask_token_id=args.model.mask_token_id, # Ensure this is passed from config
        mask_replace_ratio=args.model.mask_replace_ratio, # Ensure this is passed from config
        mask_schedule_name=args.model.mask_schedule_name, # Ensure this is passed from config
        mask_schedule_start=args.model.mask_schedule_start, # Ensure this is passed from config
        mask_schedule_end=args.model.mask_schedule_end, # Ensure this is passed from config
    )
    
    model = MMadaModelLM(model_config)

    from transformers import AutoTokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(args.model.llm_model_name_or_path)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    if llm_tokenizer.mask_token_id is not None:
        model_config.mask_token_id = llm_tokenizer.mask_token_id
    elif model_config.mask_token_id == -1:
        model_config.mask_token_id = llm_tokenizer.pad_token_id

    # ...(数据集和 DataLoader 的加载逻辑保持不变)...
    train_dataset = MolecularUnifiedDataset(
        data_path=args.model.data_path,
        tokenizer=llm_tokenizer, 
        max_text_length=args.dataset.preprocessing.max_text_length,
        max_selfies_length=args.model.max_selfies_length, 
        max_atoms=args.model.max_atoms,
        mask_token_id=model_config.mask_token_id,
        diffusion_timesteps=model_config.diffusion_timesteps,
        mask_schedule_name=model_config.mask_schedule_name,
        mask_schedule_start=model_config.mask_schedule_start,
        mask_schedule_end=model_config.mask_schedule_end,
        selfies_mask_ratio=args.model.selfies_mask_ratio,
        atom_type_mask_prob=args.model.atom_type_mask_prob,
        include_edge_bond_dist=args.model.include_edge_bond_dist, 
        include_rdmol2selfies=args.model.include_rdmol2selfies,
        rank=accelerator.process_index, 
        world_size=accelerator.num_processes,
        shuffle=True, 
        repeat=True, 
        buffer_size=args.dataset.params.shuffle_buffer_size 
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size, 
        collate_fn=train_dataset.collate_fn,
        num_workers=args.dataset.params.num_workers,
        pin_memory=args.dataset.params.pin_memory,
        persistent_workers=args.dataset.params.persistent_workers,
    )
    
    # ...(优化器和学习率调度器的加载逻辑保持不变)...
    optimizer = AdamW(model.parameters(), lr=float(args.optimizer.params.learning_rate), weight_decay=float(args.optimizer.params.weight_decay))
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler.scheduler, 
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.params.warmup_steps,
        num_training_steps=args.training.max_train_steps,
        min_lr_scale=args.lr_scheduler.params.get('min_lr_scale', 0.1)
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    mask_schedule_coords = get_noise_schedule(
        name=model_config.noise_schedule_name,
        beta_start=model_config.noise_schedule_beta_start,
        beta_end=model_config.noise_schedule_beta_end,
        timesteps=model_config.diffusion_timesteps,
    )

    progress_bar = tqdm(
        range(args.training.max_train_steps),
        desc="Training steps",
        disable=not accelerator.is_main_process,
    )

    global_step = 0
    for epoch in range(100): # 外循环可以是 epoch 或直接一个大循环
        for batch in train_dataloader:
            if global_step >= args.training.max_train_steps:
                break
            
            model.train() # 确保在训练模式
            
            model_inputs = {
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
                "mask_schedule_coords": mask_schedule_coords.to(accelerator.device), # 确保在正确设备
            }

            with accelerator.accumulate(model):
                total_loss, losses = model.forward_process(**model_inputs)

                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                progress_bar.update(1)
                global_step += 1
                
                # --- 7. 使用 W&B 记录训练指标 ---
                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                }
                for loss_name, loss_value in losses.items():
                    log_dict[f"train/{loss_name}"] = loss_value.item()
                
                wandb.log(log_dict, step=global_step)
            
            # --- 8. 周期性评估与保存 ---
            if global_step > 0 and global_step % args.experiment.eval_every == 0:
                if accelerator.is_main_process:
                    run_periodic_evaluation(model, llm_tokenizer, args, accelerator, global_step)

            if global_step > 0 and global_step % args.experiment.save_every == 0:
                if accelerator.is_main_process:
                    output_dir = os.path.join(args.experiment.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(output_dir)
                    accelerator.print(f"\nSaved checkpoint to {output_dir}")

        if global_step >= args.training.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 保存最终模型
        output_dir = os.path.join(args.experiment.output_dir, "final_checkpoint")
        accelerator.save_state(output_dir)
        accelerator.print(f"Final checkpoint saved to {output_dir}")
        wandb.finish() # 结束 W&B 运行

    accelerator.end_training()


if __name__ == "__main__":
    main()