# coding=utf-8
# Copyright 2025 MMaDA Team.
#
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
import pandas
import logging
import math
import shutil
import time
import html
from pathlib import Path
from typing import Union

import numpy as np
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.utils import get_config, flatten_omega_conf
from parquet import ChatDataset
from parquet.my_dataset import MolecularUnifiedDataset 

from models import get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter



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

    total_batch_size_per_gpu = (config.training.batch_size_lm + config.training.batch_size_mol)
    total_batch_size = (
                (config.training.batch_size_lm + config.training.batch_size_mol)
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

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    print('special tokens : \n', uni_prompting.sptids_dict)   

    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16).to(accelerator.device)

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

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset_lm = ChatDataset(data_path=dataset_config.train_lm_shards_path_or_url,
                                   rank=accelerator.process_index,
                                   world_size=accelerator.num_processes,
                                   num_workers=dataset_config.num_workers,
                                   max_length=preproc_config.max_lm_text_length,
                                   tokenizer=uni_prompting.text_tokenizer,
                                   )

    train_dataloader_lm = torch.utils.data.DataLoader(dataset_lm, batch_size=config.training.batch_size_lm,
                                                      sampler=None, collate_fn=dataset_lm.collate_fn,
                                                      num_workers=dataset_config.num_workers)
    
    train_dataset_mol = MolecularUnifiedDataset(
        data_path=dataset_config.train_mol_data_path, # 你需要在config中定义这个路径
        tokenizer=tokenizer, # 沿用你现有的tokenizer
        mask_token_id=model.config.mask_token_id, # 从模型config获取
        diffusion_timesteps=model.config.diffusion_timesteps,
        mask_schedule_name=model.config.mask_schedule_name,
        mask_schedule_start=model.config.mask_schedule_start,
        mask_schedule_end=model.config.mask_schedule_end,
        max_text_length=preproc_config.max_lm_text_length, # 文本描述长度
        max_selfies_length=config.dataset.preprocessing.max_selfies_length, # SELFIES长度，需要在config中定义
        max_atoms=config.model.mmada.max_atoms, # 最大原子数，需要在config中定义
    )

    train_dataloader_mol = torch.utils.data.DataLoader(
        train_dataset_mol,
        batch_size=config.training.batch_size_mol, # 你需要在config中定义batch_size_mol
        collate_fn=train_dataset_mol.collate_fn,
        num_workers=dataset_config.num_workers,
        pin_memory=True, # 如果需要
    )

    # Combine these dataloaders into a single iterable model
    iterables = {
        "lm_flow": train_dataloader_lm,
        "mol_flow": train_dataloader_mol,
    }

    # 
    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)
    
    num_batches_per_epoch_lm = len(train_dataloader_lm)
    num_batches_per_epoch_mol = len(train_dataloader_mol)

    # 根据你的 config.dataset.combined_loader_mode 来决定如何合并批次总数
    if config.dataset.combined_loader_mode == "max_size_cycle": # 根据你的实际模式调整
        total_batches_per_epoch = max(num_batches_per_epoch_lm, num_batches_per_epoch_mol)
    elif config.dataset.combined_loader_mode == "min_size":
        total_batches_per_epoch = min(num_batches_per_epoch_lm, num_batches_per_epoch_mol)
    elif config.dataset.combined_loader_mode == "sequential":
        total_batches_per_epoch = num_batches_per_epoch_lm + num_batches_per_epoch_mol
    else:
         # 如果有其他模式，需要相应处理，或者设置一个合理的默认值
         total_batches_per_epoch = num_batches_per_epoch_lm + num_batches_per_epoch_mol 

    num_update_steps_per_epoch = math.ceil(total_batches_per_epoch / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    start_step = 0

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
            global_step = start_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            if os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin'):
                state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            elif os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin.index.json'):
                from safetensors.torch import load_file
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, f'{path}/unwrapped_model/')
            elif os.path.exists(f'{path}/unwrapped_model/model.safetensors.index.json'):
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(
                    model, 
                    f'{path}/unwrapped_model/',
                )
            else:
                raise FileNotFoundError(f"Checkpoint {path}/unwrapped_model/pytorch_model.bin not found")
    else:
        logger.info("Not resuming from checkpoint")

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    mask_dtype = model.get_input_embeddings().weight.dtype

    from training.utils import get_noise_schedule # 确保这个导入在文件顶部
    mask_schedule_coords = get_noise_schedule(
        name=config.model.mmada.noise_schedule_name, # 假设在config中定义
        beta_start=config.model.mmada.noise_schedule_beta_start,
        beta_end=config.model.mmada.noise_schedule_beta_end,
        timesteps=config.model.mmada.diffusion_timesteps,
    )

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        texts: Union[str, list[str]], max_seq_len, eps=1e-3
    ):
        # create MLM mask and labels
        
        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts, max_seq_len), 'lm')
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id 
        
        return noisy_batch, labels_lm, p_mask

    @torch.no_grad()
    def prepare_inputs_and_labels_for_chat_text(
        texts: Union[str, list[str]], max_seq_len, eps=1e-3
    ):
        # create MLM mask and labels
        
        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts, max_seq_len), 'lm_chat')
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id 
        noisy_batch[prompt_mask.bool()] = input_ids_lm[prompt_mask.bool()]
        masked_indices = noisy_batch == mask_id 
        answer_lengths_lm = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths_lm = answer_lengths_lm.repeat(1, noisy_batch.shape[1])
        
        return noisy_batch, labels_lm, p_mask, answer_lengths_lm

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:
            data_time_m.update(time.time() - end)

            lm_batch = batch["lm_flow"]
            mol_batch = batch["mol_flow"]

            batch_size_lm = len(lm_batch["input_ids"])
            batch_size_mol = len(mol_batch["selfies_input_ids"]) # 或者其他适合你数据的批处理大小获取方式

            data_time_m.update(time.time() - end)

            # Prepare LM inputs:
            # 确保 prepare_inputs_and_labels_for_chat_text 已经更新，处理 lm_batch["input_ids"]
            # 它的返回值应为 (noisy_batch, labels_lm, p_mask, answer_lengths_lm)
            lm_input_ids, lm_labels, lm_p_mask, lm_answer_lengths = \
                prepare_inputs_and_labels_for_chat_text(lm_batch["input_ids"], config.dataset.preprocessing.max_lm_text_length)

            # 准备传递给 MMadaModelLM.forward_process 的所有输入
            model_inputs = {
                "selfies_input_ids": mol_batch["selfies_input_ids"],
                "selfies_attention_mask": mol_batch["selfies_attention_mask"],
                "atom_vec": mol_batch["atom_vec"],
                "coordinates": mol_batch["coordinates"], # 这是原始的 x_0
                "atoms_mask": mol_batch["atoms_mask"],
                "text_input_ids": lm_batch["text_input_ids"], # 假设 ChatDataset 也能提供 text_input_ids
                "text_attention_mask": lm_batch["text_attention_mask"], # 假设 ChatDataset 也能提供 text_attention_mask
                "timesteps": mol_batch["timesteps"], # MolecularUnifiedDataset 中生成的 timesteps

                "true_coordinates": mol_batch["coordinates"], # 干净坐标，用于损失计算
                "true_atom_vec": mol_batch["atom_vec"], # 真实原子类型，用于损失计算
                "mask_schedule_coords": mask_schedule_coords, # 用于连续扩散的 mask schedule
                "true_selfies_labels": mol_batch["true_selfies_labels"], # 真实SELFIES标签，用于损失计算
                "task_type": "1d_to_3d", # 或者根据你的任务类型（可能动态设置）

                # 额外的 LM 任务相关参数，如果 MMadaModelLM.forward_process 内部支持同时计算
                # 注意：你需要调整 modeling_mmada.py 中的 forward_process 来接收这些参数
                "lm_input_ids": lm_input_ids, # 经过 masking 的 lm 输入
                "lm_labels": lm_labels, # 原始 lm 标签
                "lm_p_mask": lm_p_mask, # lm 的 masking 概率
                "lm_answer_lengths": lm_answer_lengths, # lm 的答案长度
            }

            with accelerator.accumulate(model):
                total_loss, losses = model.forward_process(**model_inputs)
                accelerator.backward(total_loss)

            # 6. 梯度同步后的操作 (只有在梯度累积完成时才执行)
            if accelerator.sync_gradients:
                # 梯度裁剪
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                # 优化器步进和学习率调度
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # 记录梯度范数
                if (global_step + 1) % config.experiment.log_grad_norm_every == 0 and accelerator.is_main_process:
                    log_grad_norm(model, accelerator, global_step + 1)

                # 更新时间统计
                batch_time_m.update(time.time() - end)
                end = time.time()

                # 日志记录 (与之前修改一致)
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "total_loss": total_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step + 1,
                    }
                    for loss_name, loss_value in losses.items():
                        logs[loss_name] = loss_value.item()
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(f"Step: {global_step + 1} Total Loss: {total_loss.item():0.4f} LR: {lr_scheduler.get_last_lr()[0]:0.6f}")
                    for loss_name, loss_value in losses.items():
                        logger.info(f"  {loss_name}: {loss_value.item():0.4f}")

                    batch_time_m.reset()
                    data_time_m.reset()

                # 保存检查点
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1, uni_prompting)

                # 生成聊天文本 (或其他任务的评估/生成)
                if ((global_step + 1) % config.experiment.generate_every == 0 or global_step == start_step) and accelerator.is_main_process:
                    generate_chat_text(
                        model, uni_prompting, accelerator, config, global_step + 1
                    )

                # 增加全局步数
                global_step += 1

            # 7. 检查是否达到最大训练步数，如果是则退出内层循环
            if global_step >= config.training.max_train_steps:
                break # 退出 for batch, ... 循环

        # 如果达到最大训练步数，也退出外层循环
        if global_step >= config.training.max_train_steps:
            break #

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step, uni_prompting)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()

@torch.no_grad()
def generate_chat_text(
        model,
        uni_prompting,
        accelerator,
        config,
        global_step,
):
    logger.info("Generating chat text...")
    model.eval()

    df = pandas.read_json(config.dataset.params.lm_chat_validation_jsonl, lines=True)
    prompts = df['question'].tolist()
    responses = [''] * len(prompts)

    device = accelerator.device

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    html_content = "<div style='font-family:Arial, sans-serif;'>"
    html_content += f"<h2 style='color:navy;'>Step {global_step}</h2>"

    for i, prompt in enumerate(prompts):
        original_prompt = prompt

        prompt_with_tags = "<|start_header_id|>user<|end_header_id|>\n" + f"{prompt}" + "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
        token_ids = uni_prompting.text_tokenizer([prompt_with_tags])['input_ids'][0]
        token_ids = [uni_prompting.text_tokenizer.bos_token_id] + token_ids
        input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            # 这一行是缺失的，用于生成 output_ids
            output_ids = accelerator.unwrap_model(model).mmu_generate(
                input_ids,
                max_new_tokens=config.dataset.preprocessing.max_seq_length,
                steps=config.dataset.preprocessing.max_lm_text_length // 2,
                block_length=config.dataset.preprocessing.max_seq_length // 4
            )
        text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        responses[i] += text[0]

        escaped_prompt = html.escape(original_prompt)
        escaped_response = html.escape(responses[i])
        html_content += f"""
        <div style='border: 1px solid #ddd; margin:10px 0; padding:10px;'>
          <h4 style='margin: 0;'>Prompt</h4>
          <p style='margin: 0;'>{escaped_prompt}</p>
          <h4 style='margin: 0; margin-top:5px;'>Response</h4>
          <p style='margin: 0;'>{escaped_response}</p>
        </div>
        """

    html_content += "</div>" 

    model.train()

    wandb.log({"chat_text_generation": wandb.Html(html_content)}, step=global_step)

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
