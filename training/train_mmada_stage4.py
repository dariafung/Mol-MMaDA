
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
import random
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
from parquet import RefinedWebDataset, ChatDataset, VQADataset

from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from torchmetrics.functional.multimodal import clip_score
from functools import partial

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

    total_batch_size_per_gpu = (config.training.batch_size_t2i
                                + config.training.batch_size_lm
                                + config.training.batch_size_mmu)
    total_batch_size = (
            (config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu)
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

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
            config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # LLM pure text dataset: RefinedWeb
    dataset_lm = RefinedWebDataset(data_path=dataset_config.train_lm_shards_path_or_url,
                                   rank=accelerator.process_index,
                                   world_size=accelerator.num_processes)
    train_dataloader_lm = torch.utils.data.DataLoader(dataset_lm, batch_size=config.training.batch_size_lm,
                                                      sampler=None, collate_fn=dataset_lm.collate_fn,
                                                      num_workers=dataset_config.num_workers)

    dataset_instruct = ChatDataset(data_path=dataset_config.train_instruct_shards_path_or_url,
                                   rank=accelerator.process_index,
                                   world_size=accelerator.num_processes,
                                   max_length=preproc_config.max_lm_text_length,
                                   tokenizer=uni_prompting.text_tokenizer,
                                   )

    train_dataloader_instruct = torch.utils.data.DataLoader(dataset_instruct, batch_size=config.training.batch_size_lm,
                                                      sampler=None, collate_fn=dataset_instruct.collate_fn,
                                                      num_workers=dataset_config.num_workers)

    dataset_vqa = VQADataset(
        json_path=dataset_config.external_vqa_caption_path,
        tokenizer=uni_prompting.text_tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resolution=preproc_config.resolution,
        max_length=preproc_config.max_seq_length
    )
    train_dataloader_vqa = torch.utils.data.DataLoader(dataset_vqa, batch_size=config.training.batch_size_mmu,
                                                       sampler=None, collate_fn=dataset_vqa.collate_fn,
                                                       num_workers=dataset_config.num_workers)

    dataset_clevr2 = VQADataset(
        json_path=dataset_config.external_clevr2_caption_path,
        tokenizer=uni_prompting.text_tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resolution=preproc_config.resolution,
        max_length=preproc_config.max_seq_length
    )
    train_dataloader_clevr2 = torch.utils.data.DataLoader(dataset_clevr2, batch_size=config.training.batch_size_mmu,
                                                       sampler=None, collate_fn=dataset_clevr2.collate_fn,
                                                       num_workers=dataset_config.num_workers)

    dataset_geo170k = VQADataset(
        json_path=dataset_config.external_geo170k_caption_path,
        tokenizer=uni_prompting.text_tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resolution=preproc_config.resolution,
        max_length=preproc_config.max_seq_length,
    )
    train_dataloader_geo170k = torch.utils.data.DataLoader(dataset_geo170k, batch_size=config.training.batch_size_mmu,
                                                       sampler=None, collate_fn=dataset_geo170k.collate_fn,
                                                       num_workers=dataset_config.num_workers)

    # Combine these dataloaders into a single iterable model
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "instruct_flow": train_dataloader_instruct,
        "mmu_flow": train_dataloader_mmu,
        "vqa_flow": train_dataloader_vqa,
        "clevr2_flow": train_dataloader_clevr2,
        "geo170k_flow": train_dataloader_geo170k,
    }

    # 
    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

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
            # if safetensors sharded checkpoint exists
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

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            texts: Union[str, list[str]],
            min_masking_rate: float = 0.0,
            is_train: bool = True,
            seed: int = None
    ):


        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
            seed=seed
        )
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')
        return input_ids, labels, mask_prob, masks

    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        texts: Union[str, list[str]], max_seq_len, eps=1e-3
    ):
        # create MLM mask and labels
        
        input_ids_lm, attention_mask, labels_lm = uni_prompting((texts, max_seq_len), 'lm')
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id 
        answer_lengths_lm = torch.sum(attention_mask, dim=-1, keepdim=True)
        answer_lengths_lm = answer_lengths_lm.clamp(min=1)
        answer_lengths_lm = answer_lengths_lm.repeat(1, noisy_batch.shape[1])
        
        return noisy_batch, labels_lm, p_mask, answer_lengths_lm

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
        answer_lengths_lm = answer_lengths_lm.clamp(min=1)
        answer_lengths_lm = answer_lengths_lm.repeat(1, noisy_batch.shape[1])
        
        return noisy_batch, labels_lm, p_mask, answer_lengths_lm

    @torch.no_grad()
    def prepare_inputs_and_labels_for_mmu(
        input_ids_mmu, prompt_masks, labels_mmu, eps=1e-3
    ):
        b, l = input_ids_mmu.shape
        t = torch.rand(b, device=input_ids_mmu.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_mmu.device) < p_mask
        # 126336 is used for [MASK] token 
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_mmu)
        masked_indices = noisy_batch == mask_id 
        noisy_batch[prompt_masks.bool()] = input_ids_mmu[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id 

        prompt_masks = prompt_masks.to(torch.int64)    
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])    

        return noisy_batch, labels_mmu, p_mask, answer_lengths



    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:

            # for loss calculation
            batch_size_lm = len(batch["lm_flow"]["input_ids"])

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for language modeling
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            max_seq_len = input_ids.shape[-1]

            probs = [config.training.base_in_lm_coeff, config.training.instruct_in_lm_coeff]
            probs_total = sum(probs)
            probs = [p / probs_total for p in probs]
            cum_probs = [sum(probs[:i+1]) for i in range(len(probs))]
            rand_val = random.random()
            if rand_val < cum_probs[0]:
                texts_lm = batch["lm_flow"]["input_ids"]
                (
                    input_ids_lm,  
                    labels_lm,
                    p_mask_lm,
                    answer_lengths_lm
                ) = prepare_inputs_and_labels_for_text(texts_lm, max_seq_len)  
                input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
                labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)
            else:
                texts_lm = batch["instruct_flow"]["input_ids"]
                (
                    input_ids_lm,  
                    labels_lm,
                    p_mask_lm,
                    answer_lengths_lm
                ) = prepare_inputs_and_labels_for_chat_text(texts_lm, max_seq_len)  
                input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
                labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)

            

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for captioning/multimodal understanding
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            if "llava" in config.dataset.und_type:
                input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)

                input_ids_mmu = torch.cat([
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(
                        accelerator.device),
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(
                        accelerator.device),
                    ,
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(
                        accelerator.device),
                    input_ids_mmu,
                ], dim=1).long()

                labels_mmu = torch.cat([
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                    labels_mmu.to(accelerator.device)
                ], dim=1).long()

            else:
                probs = [config.training.cot_in_mmu_coeff, config.training.vqa_in_mmu_coeff, config.training.clevr2_in_mmu_coeff, config.training.geo170k_in_mmu_coeff]
                probs_total = sum(probs)
                probs = [p / probs_total for p in probs]
                cum_probs = [sum(probs[:i+1]) for i in range(len(probs))]
                rand_val = random.random()

            input_ids = torch.cat((input_ids, input_ids_mmu.to(input_ids.device)), dim=0)
            labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)
            
            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            with accelerator.accumulate(model):
                logits, loss_t2i, loss_lm, loss_mmu = model.forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_lm=batch_size_lm,
                    batch_size_mmu=batch_size_mmu,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    p_mask_lm=p_mask_lm,
                    p_mask_mmu=p_mask_mmu,  
                    answer_lengths=answer_lengths,
                    t2i_masks=t2i_masks,
                    answer_lengths_lm=answer_lengths_lm
                )
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_t2i = accelerator.gather(loss_t2i.repeat(config.training.batch_size_t2i)).mean()
                avg_loss_lm = accelerator.gather(loss_lm.repeat(config.training.batch_size_lm)).mean()
                avg_loss_mmu = accelerator.gather(loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                loss = config.training.t2i_coeff * loss_t2i + \
                       config.training.lm_coeff * loss_lm + \
                       config.training.mmu_coeff * loss_mmu

                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_t2i)).mean()

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
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_mmu": avg_loss_mmu.item(),
                        "step_loss_lm": avg_loss_lm.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()


                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1, uni_prompting)

                if ((global_step + 1) % config.experiment.generate_every == 0 or global_step == start_step) and accelerator.is_main_process:

                    
                    generate_chat_text(
                        model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                    )

                global_step += 1
            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

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
