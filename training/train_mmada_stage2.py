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
from typing import Union

import numpy as np
# from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
# from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler as transformers_get_scheduler, SchedulerType
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

# try:
#     import apex

#     is_apex_available = True
# except ImportError:
#     is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


# def get_vq_model_class(model_type):
#     if model_type == "magvitv2":
#         return MAGVITv2
#     elif model_type == "vq16":
#         return VQ_16
#     else:
#         raise ValueError(f"model_type {model_type} not supported.")


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

    # --- 扩展 Tokenizer 词汇表以包含 SELFIES 符号 ---
    # get_semantic_robust_alphabet 提供了语义健壮的 SELFIES 符号
    new_selfies_tokens = list(selfies.get_semantic_robust_alphabet())
    num_added_toks = tokenizer.add_tokens(new_selfies_tokens)
    logger.info(f"Added {num_added_toks} new SELFIES tokens to tokenizer vocabulary.")

    # ... (加载模型和处理模型并行化的代码)
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16)

    # --- resize_token_embeddings (如果需要) ---
    # 如果LLM的embedding层大小与tokenizer的词汇表大小不匹配，需要调整
    if len(tokenizer) > model.config.vocab_size: # 检查新词汇表是否比模型原有的大
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to new vocabulary size: {len(tokenizer)}")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    # vq_model = get_vq_model_class(config.model.vq_model.type)
    # if config.model.vq_model.get("pretrained_model_path", None):
    #     vq_model = vq_model().to(accelerator.device)
    #     state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
    #     vq_model.load_state_dict(state_dict)
    # else:
    #     vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    # vq_model.eval()
    # vq_model.requires_grad_(False)

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

    # total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    # total_batch_size_t2i = (
    #         config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    # )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    # preproc_config = config.dataset.preprocessing
    # dataset_config = config.dataset.params

    # Data for generation
    # if config.dataset.gen_type == "t2i":
    #     dataset = Text2ImageDataset(
    #         train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
    #         tokenizer=None,  # we want to get raw texts
    #         max_seq_length=preproc_config.max_seq_length,
    #         num_train_examples=config.experiment.max_train_examples_t2i,
    #         per_gpu_batch_size=config.training.batch_size_t2i,
    #         global_batch_size=total_batch_size_t2i_without_accum,
    #         num_workers=dataset_config.num_workers,
    #         resolution=preproc_config.resolution,
    #         shuffle_buffer_size=dataset_config.shuffle_buffer_size,
    #         pin_memory=dataset_config.pin_memory,
    #         persistent_workers=dataset_config.persistent_workers,
    #         external_caption_path=dataset_config.external_caption_path,
    #         external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
    #         external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
    #         external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
    #     )
    #     train_dataloader_t2i = dataset.train_dataloader
    #     num_update_steps_per_epoch = math.ceil(
    #         train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps)
    #     num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # elif config.dataset.gen_type == "t2i_parquet":
    #     # this part relies on the internal packages, which will not be released
    #     num_update_steps_per_epoch = math.ceil(config.experiment.max_train_examples_t2i / total_batch_size_t2i)
    #     num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    #     train_dataloader_t2i = create_imagetext_dataloader(
    #         train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
    #         batch_size=config.training.batch_size_t2i,
    #         image_size=preproc_config.resolution,
    #         num_workers=dataset_config.num_workers,
    #         num_readers=32,
    #         predefined_steps=num_update_steps_per_epoch,
    #         drop_last=True,
    #         shuffle=True,
    #         shuffle_buffer_size=dataset_config.shuffle_buffer_size
    #     )

    # elif config.dataset.gen_type == "imagenet1k":
    #     dataset_imagenet = ImageNetDataset(
    #         dataset_config.train_t2i_shards_path_or_url,
    #         image_size=preproc_config.resolution,
    #     )

    #     print('process index : ',
    #           accelerator.process_index, ', ', accelerator.num_processes,
    #           "Length: ", len(dataset_imagenet))

    #     if accelerator.num_processes > 1:
    #         sampler = DistributedSampler(dataset_imagenet,
    #                                      num_replicas=accelerator.num_processes,
    #                                      rank=accelerator.process_index,
    #                                      shuffle=True,
    #                                      )
    #         shuffle = False
    #     else:
    #         sampler = None
    #         shuffle = True

    #     train_dataloader_t2i = DataLoader(dataset_imagenet, batch_size=config.training.batch_size_t2i,
    #                                       sampler=sampler, collate_fn=dataset_imagenet.collate_fn,
    #                                       shuffle=shuffle, num_workers=dataset_config.num_workers)
    #     num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
    #     num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # else:
    #     raise ValueError(f"Unsupported dataset type {config.dataset.type}")


    # total_batch_size_mmu_without_accum = config.training.batch_size_mmu * accelerator.num_processes
    # # Data for image captioning
    # if config.dataset.und_type == "captioning":
    #     dataset_mmu = Text2ImageDataset(
    #         train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
    #         tokenizer=None,  # we want to get raw texts
    #         max_seq_length=preproc_config.max_seq_length,
    #         num_train_examples=config.experiment.max_train_examples_mmu,
    #         per_gpu_batch_size=config.training.batch_size_mmu,
    #         global_batch_size=total_batch_size_mmu_without_accum,
    #         num_workers=dataset_config.num_workers,
    #         resolution=preproc_config.resolution,
    #         shuffle_buffer_size=dataset_config.shuffle_buffer_size,
    #         pin_memory=dataset_config.pin_memory,
    #         persistent_workers=dataset_config.persistent_workers,
    #         external_caption_path=dataset_config.external_caption_path,
    #         external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
    #         external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
    #         external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
    #         is_captioning=True,
    #         add_caption_prompt=dataset_config.add_caption_prompt,
    #     )
    #     train_dataloader_mmu = dataset_mmu.train_dataloader

    # elif config.dataset.und_type == "captioning_parquet":
    #     train_dataloader_mmu = create_imagetext_dataloader(
    #         train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
    #         batch_size=config.training.batch_size_mmu,
    #         image_size=preproc_config.resolution,
    #         num_workers=dataset_config.num_workers,
    #         num_readers=32,
    #         predefined_steps=num_update_steps_per_epoch,
    #         drop_last=True,
    #         shuffle=True,
    #         shuffle_buffer_size=dataset_config.shuffle_buffer_size,
    #         is_captioning=True
    #     )


    # else:
    #     raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    # # LLM pure text dataset: RefinedWeb
    # dataset_lm = RefinedWebDataset(data_path=dataset_config.train_lm_shards_path_or_url,
    #                                rank=accelerator.process_index,
    #                                world_size=accelerator.num_processes,
    #                                num_workers=dataset_config.num_workers)

    # train_dataloader_lm = torch.utils.data.DataLoader(dataset_lm, batch_size=config.training.batch_size_lm,
    #                                                   sampler=None, collate_fn=dataset_lm.collate_fn,
    #                                                   num_workers=dataset_config.num_workers)

    # # Combine these dataloaders into a single iterable model
    # iterables = {
    #     "t2i_flow": train_dataloader_t2i,
    #     "lm_flow": train_dataloader_lm,
    #     "mmu_flow": train_dataloader_mmu,
    # }

    # combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

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

        # 对于 Labels，这里只是一个占位符。
        # 实际的 labels 需要根据你的任务来定义。
        # 如果是自回归生成 3D，labels 可能是目标 3D token 序列。
        # 如果是类似 denoising 的任务，labels 可能与 input_ids 相似。
        # 暂时用 input_ids 作为一个默认的 placeholder label，但你需要根据 MMadaModelLM.forward_process 的期望来调整
        labels = text_input_ids.clone() # 这是一个简单的占位符，你需要根据你的损失函数和模型输出来定义真实的 labels

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


# @torch.no_grad()
# def visualize_predictions(
#         model,
#         vq_model,
#         uni_prompting,
#         config,
#         global_step,
#         input_ids,
#         image_tokens_ori,
#         ori_images,
#         texts,
#         logits,
#         accelerator
# ):
#     logger.info("Visualizing predictions...")
#     model.eval()

#     recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
#     recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
#     recons_images *= 255.0
#     recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

#     images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
#     images *= 255.0
#     images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#     predictions = logits[:config.training.batch_size_t2i, -(config.model.mmada.num_vq_tokens + 1):-1:, len(uni_prompting.text_tokenizer) + config.model.mmada.num_new_special_tokens: len(uni_prompting.text_tokenizer) + config.model.mmada.num_new_special_tokens + config.model.mmada.codebook_size]
#     predictions = predictions.argmax(axis=-1)
#     # mask_token_id = config.model.mmada.vocab_size - 1 - len(uni_prompting.text_tokenizer)
#     mask_token_id = accelerator.unwrap_model(model).config.mask_token_id - len(uni_prompting.text_tokenizer)
#     input_ids = input_ids[:config.training.batch_size_t2i, -(config.model.mmada.num_vq_tokens + 1):-1:] - len(uni_prompting.text_tokenizer)
#     mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
#         dim=-1) / config.model.mmada.num_vq_tokens).cpu().numpy())
#     predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)
#     predicted_images = vq_model.decode_code(predicted_images)
#     predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
#     predicted_images *= 255.0
#     predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#     predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
#     pil_images = [Image.fromarray(image) for image in predicted_images]

#     # Log images
#     wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} \n caption: {texts[i]}') for i, (image, r) in
#                     enumerate(zip(pil_images, mask_ratio))]
#     wandb.log({"Original images v.s. Reconstructed images v.s. Predicted images": wandb_images}, step=global_step)

#     model.train()


# @torch.no_grad()
# def generate_images(
#         model,
#         vq_model,
#         uni_prompting,
#         accelerator,
#         config,
#         global_step,
#         mask_schedule,
#         force_no_cfg = False
# ):
#     logger.info("Generating images...")
#     model.eval()

#     # read validation prompts from file
#     with open(config.dataset.params.validation_prompts_file, "r") as f:
#         validation_prompts = f.read().splitlines()

#     mask_dtype = model.get_input_embeddings().weight.dtype
#     mask_token_id = accelerator.unwrap_model(model).config.mask_token_id
#     image_tokens = torch.ones((len(validation_prompts), config.model.mmada.num_vq_tokens), dtype=torch.long,
#                               device=accelerator.device) * mask_token_id
#     input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
#     if not force_no_cfg and config.training.guidance_scale > 0:
#         uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
#         cfg_scale = config.training.guidance_scale
#     else:
#         uncond_input_ids = None
#         uncond_attention_mask = None
#         cfg_scale = 0
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         weight_dtype = torch.bfloat16
#     else:
#         weight_dtype = torch.float32

#     with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
#         # Generate images
#         gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
#             input_ids=input_ids,
#             uncond_input_ids=uncond_input_ids,
#             attention_mask=attention_mask,
#             uncond_attention_mask=uncond_attention_mask,
#             guidance_scale=cfg_scale,
#             temperature=config.training.get("generation_temperature", 1.0),
#             timesteps=config.training.generation_timesteps,
#             noise_schedule=mask_schedule,
#             noise_type=config.training.get("noise_type", "mask"),
#             predict_all_tokens=config.training.get("predict_all_tokens", False),
#             seq_len=config.model.mmada.num_vq_tokens,
#             uni_prompting=uni_prompting,
#             config=config,
#         )
#     # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
#     # so we clamp them to the correct range.
#     gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
#     images = vq_model.decode_code(gen_token_ids)

#     model.train()

#     if config.training.get("pre_encode", False):
#         del vq_model

#     # Convert to PIL images
#     images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
#     images *= 255.0
#     images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#     pil_images = [Image.fromarray(image) for image in images]

#     # Log images
#     wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
#     wandb.log({f"Generated images with cfg {cfg_scale}": wandb_images}, step=global_step)
    
    

# @torch.no_grad()
# def understanding_images(
#         model,
#         vq_model,
#         uni_prompting,
#         accelerator,
#         config,
#         global_step,
# ):
#     logger.info("Understanding images...")
#     model.eval()
        
#     file_list = os.listdir(config.dataset.params.mmu_image_root)
#     file_list = [f for f in file_list if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#     responses = ['' for i in range(len(file_list))]
#     images = []
    
#     device = accelerator.device
    
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         weight_dtype = torch.bfloat16
#     else:
#         weight_dtype = torch.float32
    
#     for i, file_name in enumerate(file_list):
#         image_path = os.path.join(config.dataset.params.mmu_image_root, file_name)
#         image_ori = Image.open(image_path).convert("RGB")
#         image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
#         image = image.unsqueeze(0)
#         images.append(image)
#         image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
#         batch_size = 1
        
#         input_ids = uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + "Please describe this image in detail."  +'<eot_id><|start_header_id|>assistant<|end_header_id|>\n'])['input_ids']
#         input_ids = torch.tensor(input_ids).to(device)

#         input_ids = torch.cat([
#             (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
#             (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
#             image_tokens,
#             (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
#             (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
#             input_ids
#         ], dim=1).long()
#         with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
#             output_ids = accelerator.unwrap_model(model).mmu_generate(input_ids)
#         # output_ids = torch.stack(output_ids).squeeze()[None]

#         text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
#         responses[i] += text[0]
#     model.train()
#     images = torch.cat(images, dim=0)
#     images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
#     images *= 255.0
#     images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#     pil_images = [Image.fromarray(image) for image in images]

#     # Log images
#     wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
#     wandb.log({"Understanding images": wandb_images}, step=global_step)


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
