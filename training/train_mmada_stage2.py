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

# Import your model and config
from models.modeling_mmada import MMadaConfig, MMadaModelLM
# Import your dataset
from parquet.my_dataset import MolecularUnifiedDataset # Corrected import path if different
from models.lr_schedulers import get_scheduler
from training.optimizer import get_optimizer
from training.utils import (
    get_noise_schedule, # For continuous diffusion
)

logger = get_logger(__name__)


def main():
    # --- 1. Load Configuration ---
    config_path = "configs/mmada_pretraining_stage2_llada_instruct.yaml" # Adjust if your config path is different
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert dictionary to a simple object for easier access
    class Config:
        def __init__(self, d):
            for a, b in d.items():
                if isinstance(b, dict):
                    setattr(self, a, Config(b))
                else:
                    setattr(self, a, b)
    
    args = Config(config_dict)

    accelerator = Accelerator(
        mixed_precision=args.training.mixed_precision,
        log_with="tensorboard",
        project_dir=args.experiment.output_dir,
    )

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
        accelerator.init_trackers("mmada_training", config=config_dict)
    else:
        transformers.utils.logging.set_verbosity_error()

    set_seed(args.training.seed)

    accelerator.print(f"Loading model config from {config_path}")
    # --- MMadaConfig needs to be instantiated correctly with relevant parameters ---
    model_config = MMadaConfig(
        llm_config_path=args.model.llm_config_path,
        llm_model_name_or_path=args.model.llm_model_name_or_path,

        mol_atom_embedding_dim=args.model.mol_atom_embedding_dim,
        mol_coord_embedding_dim=args.model.mol_coord_embedding_dim,
        mol_3d_encoder_output_dim=args.model.mol_3d_encoder_output_dim,
        num_atom_types=args.model.num_atom_types,
        max_atoms=args.model.max_atoms,
        output_atom_coords_dim=args.model.output_atom_coords_dim,
        output_atom_type_dim=args.model.output_atom_type_dim,
        d_model=args.model.d_model,
        fusion_hidden_dim=args.model.fusion_hidden_dim,
        final_condition_dim=args.model.final_condition_dim,
        diffusion_timesteps=args.model.diffusion_timesteps,
        noise_schedule_beta_start=args.model.noise_schedule_beta_start,
        noise_schedule_beta_end=args.model.noise_schedule_beta_end,
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

    # --- 2. Initialize Tokenizer for LLaDA ---
    from transformers import AutoTokenizer
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(args.model.llm_model_name_or_path)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token # Recommended for many LLMs
        
        # --- 修正 mask_token_id 的处理逻辑 ---
        if llm_tokenizer.mask_token_id is not None:
            # 如果 tokenizer 有 mask token，则优先使用 tokenizer 的 ID
            model_config.mask_token_id = llm_tokenizer.mask_token_id
        elif model_config.mask_token_id == -1:
            # 如果 tokenizer 没有 mask token 且 config 中也未指定有效 ID，则发出警告并尝试使用 pad_token_id
            # 这是一个临时处理，更健壮的方案是向 tokenizer 添加一个新 token 并调整模型嵌入层
            accelerator.print("Warning: LLM tokenizer does not have a mask_token_id. Attempting to use pad_token_id as mask_token_id.")
            model_config.mask_token_id = llm_tokenizer.pad_token_id 
            if model_config.mask_token_id is None:
                 raise ValueError("Neither mask_token_id nor pad_token_id found in tokenizer. Cannot proceed without a valid mask token.")
        # else: model_config.mask_token_id 已经从配置文件中加载了有效值

    except Exception as e:
        accelerator.print(f"Error loading LLM tokenizer from {args.model.llm_model_name_or_path}: {e}")
        accelerator.print("Falling back to a dummy tokenizer (This might cause issues if not handled properly).")
        # 从 parquet.my_dataset 导入 DummyTokenizer (假设存在)
        # from parquet.my_dataset import DummyTokenizer 
        # llm_tokenizer = DummyTokenizer()
        # model_config.mask_token_id = llm_tokenizer.mask_token_id 
        raise e # 如果加载 tokenizer 失败，最好直接报错，因为这会影响后续的 tokenization

    accelerator.print("Tokenizer loaded.")
    
    # --- 3. Prepare Dataset and DataLoader ---
    accelerator.print(f"Loading data from {args.model.data_path}")
    train_dataset = MolecularUnifiedDataset(
        data_path=args.model.data_path,
        tokenizer=llm_tokenizer, # 传递加载的 tokenizer 实例
        max_text_length=args.dataset.preprocessing.max_text_length,
        max_selfies_length=args.model.max_selfies_length, # 从 model config 获取 selfies max length
        max_atoms=args.model.max_atoms,
        mask_token_id=model_config.mask_token_id, # 使用已修正的 mask_token_id
        diffusion_timesteps=model_config.diffusion_timesteps, # 传递总扩散步长
        mask_schedule_name=model_config.mask_schedule_name,
        mask_schedule_start=model_config.mask_schedule_start,
        mask_schedule_end=model_config.mask_schedule_end,
        selfies_mask_ratio=args.model.selfies_mask_ratio, # 传递 selfies_mask_ratio
        include_edge_bond_dist=args.model.include_edge_bond_dist, # 确保传递了这些
        include_rdmol2selfies=args.model.include_rdmol2selfies, # 确保传递了这些
        # dataset.params 里的参数
        rank=accelerator.process_index, # 传递当前进程的 rank
        world_size=accelerator.num_processes, # 传递总进程数
        shuffle=True, # 假设训练时需要 shuffle
        repeat=True, # 假设训练时需要 repeat
        buffer_size=args.dataset.params.shuffle_buffer_size # 使用 config 中的 buffer_size
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size, # --- 修正为 args.training.batch_size ---
        collate_fn=train_dataset.collate_fn,
        num_workers=args.dataset.params.num_workers, # 使用 config 中的 num_workers
        pin_memory=args.dataset.params.pin_memory, # 使用 config 中的 pin_memory
        persistent_workers=args.dataset.params.persistent_workers, # 使用 config 中的 persistent_workers
    )
    
    # --- 4. Optimizer and Scheduler ---
    optimizer = get_optimizer(
        model, 
        learning_rate=float(args.optimizer.params.learning_rate), 
        weight_decay=args.optimizer.params.weight_decay, 
        name=args.optimizer.name # 确保也传递了优化器名称
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler.scheduler, # 修正为 args.lr_scheduler.scheduler
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.params.warmup_steps, # 修正为从 params 获取
        num_training_steps=args.training.max_train_steps,
        min_lr_scale=args.lr_scheduler.params.min_lr_scale # 添加 min_lr_scale
    )

    # --- 5. Accelerator Preparation ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Define noise schedule for continuous coordinates diffusion
    # --- 确保 noise_schedule_name 在 config.yaml 中有定义 ---
    mask_schedule_coords = get_noise_schedule(
        name=model_config.mask_schedule_name, # 使用 model_config 中的 mask_schedule_name，现在用于 noise schedule
        beta_start=model_config.noise_schedule_beta_start,
        beta_end=model_config.noise_schedule_beta_end,
        timesteps=model_config.diffusion_timesteps,
    )

    # --- 6. Training Loop ---
    progress_bar = tqdm(
        range(args.training.max_train_steps),
        desc="Training steps",
        disable=not accelerator.is_main_process,
    )

    for step, batch in enumerate(train_dataloader):
        if step >= args.training.max_train_steps:
            break

        # 准备传递给 model.forward_process 的所有输入
        model_inputs = {
            "selfies_input_ids": batch["selfies_input_ids"],
            "selfies_attention_mask": batch["selfies_attention_mask"],
            "text_input_ids": batch.get("text_input_ids"),
            "text_attention_mask": batch.get("text_attention_mask"),
            "atom_vec": batch["atom_vec"],
            "coordinates": batch["coordinates"], # 使用转换后的坐标
            "atoms_mask": batch["atoms_mask"],
            "task_type": "1d_to_3d",
            "true_coordinates": batch["coordinates"], # 使用转换后的真实坐标
            "true_atom_vec": batch["atom_vec"],
            "true_selfies_labels": batch["true_selfies_labels"],
            "mask_schedule_coords": mask_schedule_coords,
            "timesteps": batch["timesteps"],
        }

        with accelerator.accumulate(model):
            # The model's forward_process will handle adding noise to coordinates for input
            total_loss, losses = model.forward_process(**model_inputs)

            accelerator.backward(total_loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process:
            logs = {
                "total_loss": total_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": step,
            }
            # Log individual losses if they exist in the `losses` dictionary
            for loss_name, loss_value in losses.items():
                logs[loss_name] = loss_value.item()
            
            accelerator.log(logs, step=step)
            progress_bar.update(1)

        # Use args.experiment.save_every and output_dir
        if (step + 1) % args.experiment.save_every == 0: # 修正为 (step + 1)
            if accelerator.is_main_process:
                output_dir = os.path.join(args.experiment.output_dir, f"checkpoint-{step+1}") # 修正目录名
                accelerator.save_state(output_dir)
                accelerator.print(f"Saved checkpoint to {output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()