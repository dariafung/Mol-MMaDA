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

from parquet.my_dataset import MolecularUnifiedDataset

SYSTEM_PROMPT_LEN = 28

# from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from training.utils import get_config, flatten_omega_conf, AverageMeter

logger = get_logger(__name__, log_level="INFO")

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
        # 你的自定义分子参数
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
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

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

    @torch.no_grad()
    def prepare_molecular_inputs_and_labels(batch, accelerator_device):
        """
        准备分子数据集的输入和标签。
        batch: 从 MolecularUnifiedDataset 的 DataLoader 中获取的批次数据。
        accelerator_device: 模型所在的设备。
        """
        selfies_input_ids = batch["selfies_input_ids"].to(accelerator_device)
        selfies_attention_mask = batch["selfies_attention_mask"].to(accelerator_device)
        text_input_ids = batch["text_input_ids"].to(accelerator_device)
        text_attention_mask = batch["text_attention_mask"].to(accelerator_device)
        atom_vec = batch["atom_vec"].to(accelerator_device)
        coordinates = batch["coordinates"].to(accelerator_device)
        atoms_mask = batch["atoms_mask"].to(accelerator_device)

        # 3D 辅助特征 (如果包含)
        edge_type = batch.get("edge_type", None)
        if edge_type is not None: edge_type = edge_type.to(accelerator_device)
        bond_type = batch.get("bond_type", None)
        if bond_type is not None: bond_type = bond_type.to(accelerator_device)
        dist = batch.get("dist", None)
        if dist is not None: dist = dist.to(accelerator_device)
        rdmol2selfies = batch.get("rdmol2selfies", None)
        if rdmol2selfies is not None: rdmol2selfies = rdmol2selfies.to(accelerator_device)

        labels = coordinates.clone()

        return (
            selfies_input_ids, selfies_attention_mask,
            text_input_ids, text_attention_mask,
            atom_vec, coordinates, atoms_mask,
            edge_type, bond_type, dist, rdmol2selfies,
            labels # 返回 labels
        )

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        # 直接迭代 train_dataloader，因为不再是 CombinedLoader
        for step, batch in enumerate(train_dataloader): 
            data_time_m.update(time.time() - end)

            # --- 准备新的输入 ---
            (
                selfies_input_ids, selfies_attention_mask,
                text_input_ids, text_attention_mask,
                atom_vec, coordinates, atoms_mask,
                edge_type, bond_type, dist, rdmol2selfies,
                labels # 接收 labels
            ) = prepare_molecular_inputs_and_labels(batch, accelerator.device)

            with accelerator.accumulate(model):
                # --- 将新输入传递给模型 ---
                # 这部分需要你修改 MMadaModelLM 的 forward_process 函数来接受这些参数
                # 假设 forward_process 返回一个标量损失
                loss = model.forward_process(
                    selfies_input_ids=selfies_input_ids,
                    selfies_attention_mask=selfies_attention_mask,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask, # 假设模型需要 attention mask
                    atom_vec=atom_vec,
                    coordinates=coordinates,
                    atoms_mask=atoms_mask,
                    edge_type=edge_type, 
                    bond_type=bond_type, 
                    dist=dist,           
                    rdmol2selfies=rdmol2selfies,
                    labels=labels # 传递 labels
                )

                # Gather the loss across all processes
                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                
                accelerator.backward(loss)

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
                        "step_loss": avg_loss.item(), # 只有一个总损失
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
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
