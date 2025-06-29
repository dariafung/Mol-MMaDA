# Copyright 2025 MMaDA Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any, Callable

import numpy as np
# from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
# from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_scheduler as transformers_get_scheduler, SchedulerType
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

# from training.data import Text2ImageDataset
from training.utils import get_config, flatten_omega_conf, AverageMeter
# from training.imagenet_dataset import ImageNetDataset
# from parquet import RefinedWebDataset

# from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from models import get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler as get_lr_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import selfies
import torch.nn.functional as F

from parquet.my_dataset import MolecularUnifiedDataset

SYSTEM_PROMPT_LEN = 28

# from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from training.utils import get_config, flatten_omega_conf, AverageMeter

logger = get_logger(__name__, log_level="INFO")

# 将 prepare_molecular_inputs_and_labels 函数定义在 main 函数外部，以便它能访问到 model_ref
# 并且能够接收 mask_schedule_coords 参数
@torch.no_grad()
def prepare_molecular_inputs_and_labels(
    batch: Dict[str, torch.Tensor],
    accelerator_device: torch.device,
    config: Any, # Pass the full config object
    mask_schedule_coords: Callable, # Pass the coordinate noise scheduler
    task_type: str, # '1d_to_3d', '3d_to_1d', 'multimodal_joint'
    model_ref: Any # Pass the model instance to access get_alpha_bar
) -> Dict[str, Optional[torch.Tensor]]:
    """
    根据任务类型准备分子数据的输入和标签。

    Args:
        batch: 从 DataLoader 获取的原始批次数据。
        accelerator_device: 加速器所在的设备（CPU/GPU）。
        config: 配置文件对象。
        mask_schedule_coords: 用于坐标噪声的调度器函数。
        task_type: 当前批次要处理的任务类型（例如 '1d_to_3d'）。
        model_ref: 模型实例，用于访问 get_alpha_bar 等方法。

    Returns:
        一个字典，包含准备好的模型输入和损失计算所需的真实标签。
    """
    # 将所有批次张量移动到适当的设备
    selfies_input_ids = batch["selfies_input_ids"].to(accelerator_device)
    selfies_attention_mask = batch["selfies_attention_mask"].to(accelerator_device)
    text_input_ids = batch["text_input_ids"].to(accelerator_device)
    text_attention_mask = batch["text_attention_mask"].to(accelerator_device)
    atom_vec = batch["atom_vec"].to(accelerator_device)
    coordinates = batch["coordinates"].to(accelerator_device) # 这是原始的干净坐标
    atoms_mask = batch["atoms_mask"].to(accelerator_device)

    # 这些将是 model.forward() 的实际输入
    # Initialize with original values, then modify based on task_type
    model_inputs = {
        "selfies_input_ids": selfies_input_ids,
        "selfies_attention_mask": selfies_attention_mask,
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "atom_vec": atom_vec,
        "coordinates": coordinates, # This will be the *input* coordinates to the model
        "atoms_mask": atoms_mask,
        "llm_generate_input_ids": selfies_input_ids, # For 1D->3D, LLM processes selfies sequence
        "llm_generate_attention_mask": selfies_attention_mask,
    }

    # 这些将是损失计算的真实目标
    target_labels = {}

    batch_size = selfies_input_ids.shape[0] # Assuming consistent batch size

    if task_type == '1d_to_3d':
        # 1D (SELFIES/Text) to 3D Generation Task
        # Inputs: SELFIES, Text. Targets: 3D (denoised coordinates, atom types)
        
        # Apply noise to coordinates for diffusion input
        timesteps_coords = torch.randint(0, config.model.mmada.diffusion_timesteps, (batch_size,), device=coordinates.device).long()
        noise_coords = torch.randn_like(coordinates)
        
        # 从模型实例中获取 alpha_bar，因为 get_alpha_bar 是 MMadaModelLM 的方法
        alpha_bar_t = model_ref.get_alpha_bar(timesteps_coords) 
        sqrt_alpha_bar_t = alpha_bar_t.sqrt().unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt().unsqueeze(-1).unsqueeze(-1)
        
        # 模型的输入坐标是带有噪声的
        model_inputs["coordinates"] = sqrt_alpha_bar_t * coordinates + sqrt_one_minus_alpha_bar_t * noise_coords
        
        # 真实的原始坐标和原子向量是损失计算的目标
        target_labels["true_coordinates"] = coordinates.clone()
        target_labels["true_atom_vec"] = atom_vec.clone()

    elif task_type == '3d_to_1d':
        # 3D 到 1D (SELFIES) 生成任务 (暂时不实现，仅为结构预留)
        pass 
    elif task_type == 'multimodal_joint':
        # 多模态联合任务 (暂时不实现，仅为结构预留)
        pass

    # Combine model inputs and targets
    prepared_batch = {**model_inputs, **target_labels}
    
    # Add task type to the batch for forward_process
    prepared_batch['task_type'] = task_type
    
    # 传递噪声调度器到 forward_process (如果 forward_process 需要直接访问它们)
    prepared_batch['mask_schedule_coords'] = mask_schedule_coords

    return prepared_batch

def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = config.training.batch_size 
    total_batch_size = (
        config.training.batch_size
        * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.tokenizer_path, padding_side="left")

    new_selfies_tokens = list(selfies.get_semantic_robust_alphabet())
    num_added_toks = tokenizer.add_tokens(new_selfies_tokens)
    logger.info(f"Added {num_added_toks} new SELFIES tokens to tokenizer vocabulary.")

    # 1. 首先加载基础的 LLM 模型 (LLaDA-8B-Instruct)
    # 这一步会确保正确的 LLaDAConfig 和 LLaDAModelLM 类被加载到内存中
    base_llm_model = AutoModelForCausalLM.from_pretrained(
        config.model.mmada.pretrained_model_path,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True # 信任远程代码，以便正确加载 LLaDA 的自定义模块
    )
    
    # 2. 从加载的基础 LLM 模型中获取其配置
    base_llm_config = base_llm_model.config # 这是 transformers_modules 提供的 LLaDAConfig 实例
    
    # 3. 将原始 LLM 配置的参数转换为字典
    mmada_model_config_kwargs = base_llm_config.to_dict() # 以 base_llm_config 的参数为基础
    
    # 4. 合并我们自定义的分子参数 (来自 config.yaml 的 model.mmada 部分)
    mmada_model_config_kwargs.update({
        "llm_vocab_size": config.model.mmada.llm_vocab_size,
        "llm_model_path": config.model.mmada.tokenizer_path, # 
        "num_new_special_tokens": config.model.mmada.num_new_special_tokens,
        "gradient_checkpointing": config.model.mmada.gradient_checkpointing, 
       
        "mol_atom_embedding_dim": config.model.mmada.mol_atom_embedding_dim,
        "mol_coord_embedding_dim": config.model.mmada.mol_coord_embedding_dim,
        "mol_3d_encoder_output_dim": config.model.mmada.mol_3d_encoder_output_dim,
        "fusion_hidden_dim": config.model.mmada.fusion_hidden_dim,
        "final_condition_dim": config.model.mmada.final_condition_dim,
        "num_atom_types": config.model.mmada.num_atom_types,
        "max_atoms": config.max_atoms, 
        "output_atom_coords_dim": config.model.mmada.output_atom_coords_dim,
        "output_atom_type_dim": config.model.mmada.output_atom_type_dim,
        "diffusion_timesteps": config.model.mmada.diffusion_timesteps,
        "noise_schedule_beta_start": config.model.mmada.noise_schedule_beta_start,
        "noise_schedule_beta_end": config.model.mmada.noise_schedule_beta_end,

        "coords_coeff": config.training.coords_coeff,
        "atom_type_coeff": config.training.atom_type_coeff,
        "alignment_coeff": config.training.get("alignment_coeff", 0.0), 
        "selfies_coeff": config.training.get("selfies_coeff", 0.0), 
        "hierarchical_coeff": config.training.get("hierarchical_coeff", 0.0), 
    })
    
    # 5. 实例化一个完整的 MMadaConfig 对象
    # 这个 mmada_config 现在包含了所有基础LLM参数和我们的自定义参数
    mmada_config = MMadaConfig(**mmada_model_config_kwargs)

    # 6. 使用这个完整的 mmada_config 实例化 MMadaModelLM
    # MMadaModelLM 的 __init__ 方法会使用这个 config 来设置其所有层
    model = MMadaModelLM(mmada_config, base_llm_model)

    del base_llm_model
    torch.cuda.empty_cache()

    logger.info(f"Loaded pretrained LLM weights into MMadaModelLM from {config.model.mmada.pretrained_model_path}")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    print('special tokens : \n', uni_prompting.sptids_dict)

    mask_id = model.config.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        llm_mask_schedule = get_mask_schedule(schedule, **args) 
    else:
        llm_mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine")) 

    mask_schedule_coords = get_mask_schedule(config.training.get("coords_mask_schedule", "cosine"))

    lr_scheduler = get_lr_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders")

     # --- 实例化你的 MolecularUnifiedDataset ---
    train_dataset = MolecularUnifiedDataset(
        data_path=config.data_path, # 从 config 中获取的新的数据路径
        tokenizer=tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        shuffle=config.dataset.params.shuffle_buffer_size > 0, # 如果 buffer_size > 0 则 shuffle
        repeat=True, # 通常训练集是重复的
        buffer_size=config.dataset.params.shuffle_buffer_size,
        max_text_length=config.dataset.preprocessing.max_seq_length,
        max_selfies_length=config.max_selfies_length, # 从 config 中获取
        max_atoms=config.max_atoms,                   # 从 config 中获取
        include_edge_bond_dist=config.include_edge_bond_dist, # 从 config 中获取
        include_rdmol2selfies=config.include_rdmol2selfies    #从 config 中获取
    )

    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size, # 使用你新的 batch_size 参数
        collate_fn=train_dataset.collate_fn, # 使用你自定义的 collate_fn
        num_workers=config.dataset.params.num_workers,
        pin_memory=config.dataset.params.pin_memory,
        persistent_workers=config.dataset.params.persistent_workers,
    )

    # --- 计算 num_update_steps_per_epoch 和 num_train_epochs ---
    # 这里需要一个预估的训练样本总数 (max_train_examples_molecular)
    total_num_train_examples = config.experiment.get("max_train_examples_molecular", 20_000_000) # 假设 2000万分子

    num_update_steps_per_epoch = math.ceil(total_num_train_examples / total_batch_size)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    logger.info(f"Total number of epochs: {num_train_epochs}")

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        logger.info(f"dirs: {dirs}")
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        logger.info(f"path: {path}")
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            if os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin'):
                state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            elif os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin.index.json'):
                from safetensors.torch import load_file
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, f'{path}/unwrapped_model/')
            # if safetensors sharded checkpoint exists
            elif os.path.exists(f'{path}/unwrapped_model/model.safetensors.index.json'):
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(
                    model, 
                    f'{path}/unwrapped_model/',
                    # weight_map=None, 
                    # load_state_dict_fn="safetensors"
                )
            else:
                raise FileNotFoundError(f"Checkpoint {path}/unwrapped_model/pytorch_model.bin not found")
    else:
        logger.info("Not resuming from checkpoint")

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    mask_dtype = model.llm_backbone.get_input_embeddings().weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader): 
            data_time_m.update(time.time() - end)

            # --- 准备新的输入 ---
            # 明确指定 task_type 为 '1d_to_3d'
            # 传递 model 实例给 prepare_molecular_inputs_and_labels
            prepared_batch = prepare_molecular_inputs_and_labels(
                batch,
                accelerator.device,
                config,
                mask_schedule_coords, # 传递 coords 调度器
                '1d_to_3d', # 指定任务类型
                model # 传递模型实例
            )

            with accelerator.accumulate(model):
                # --- 将新输入传递给模型 ---
                # model.forward_process 现在接受更多参数
                total_loss, individual_losses = model.forward_process(
                    selfies_input_ids=prepared_batch["selfies_input_ids"],
                    selfies_attention_mask=prepared_batch["selfies_attention_mask"],
                    text_input_ids=prepared_batch["text_input_ids"],
                    text_attention_mask=prepared_batch["text_attention_mask"],
                    atom_vec=prepared_batch["atom_vec"],
                    coordinates=prepared_batch["coordinates"], # 这是带噪声的输入坐标
                    atoms_mask=prepared_batch["atoms_mask"],
                    task_type=prepared_batch["task_type"], # '1d_to_3d'
                    # 传递真实标签
                    true_coordinates=prepared_batch.get("true_coordinates"),
                    true_atom_vec=prepared_batch.get("true_atom_vec"),
                    # 暂时不传 selfies 相关，因为是 1D->3D 任务
                    # true_selfies_labels=prepared_batch.get("true_selfies_labels"), # No longer passed for 1D->3D
                    mask_schedule_coords=mask_schedule_coords, # 传递 coords 调度器
                )

                # Gather the loss across all processes
                # total_loss 和 individual_losses 是从 model.forward_process 返回的
                avg_loss = accelerator.gather(total_loss.repeat(config.training.batch_size)).mean()
                
                accelerator.backward(total_loss) # 使用 total_loss 进行反向传播

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * config.training.batch_size / batch_time_m.val
                    )
                    logs = {
                        "total_loss": avg_loss.item(), # 总损失
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    # 记录 individual_losses
                    for loss_name, loss_value in individual_losses.items():
                        logs[f"loss/{loss_name}"] = accelerator.gather(loss_value.repeat(config.training.batch_size)).mean().item()
                    
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1, uni_prompting)

                # --- 移除图像生成和理解相关的调用 ---
                # config.experiment.generate_every 和 config.experiment.val_every (如果用于图像生成) 也要在 config 中调整或移除
                # if ((global_step + 1) % config.experiment.generate_every == 0 or global_step == 0) and accelerator.is_main_process:
                #    generate_images(...)
                #    understanding_images(...) # 这两个函数和它们的调用都已移除

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step, uni_prompting)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()

def save_checkpoint(model, config, accelerator, global_step, uni_prompting):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

        # save tokenizer
        uni_prompting.text_tokenizer.save_pretrained(save_path/ "unwrapped_model")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
