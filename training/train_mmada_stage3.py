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
from typing import List, Union

import pandas as pd
import torch
import wandb
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.utils import get_config, flatten_omega_conf, AverageMeter
from models import get_mask_schedule, MMadaModelLM
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from training.prompting_utils import UniversalPrompting

logger = get_logger(__name__, log_level="INFO")


def save_checkpoint(model, config, accelerator, global_step, uni_prompting):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
            for rm in removing_checkpoints:
                shutil.rmtree(os.path.join(output_dir, rm))

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        uni_prompting.text_tokenizer.save_pretrained(save_path / "unwrapped_model")
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


@torch.no_grad()
def generate_chat_text(model, uni_prompting, accelerator, config, global_step):
    logger.info("Generating chat text...")
    model.eval()

    df = pd.read_json(config.dataset.params.lm_chat_validation_jsonl, lines=True)
    prompts = df["question"].tolist()
    responses = [""] * len(prompts)

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
        enc = uni_prompting.text_tokenizer([prompt_with_tags])["input_ids"]
        input_ids = torch.tensor(enc).to(device)

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            out_ids = accelerator.unwrap_model(model).generate(
                input_ids,
                max_new_tokens=config.dataset.preprocessing.max_seq_length,
                do_sample=False,
            )

        text = uni_prompting.text_tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        responses[i] += text[0]

        html_content += f"""
        <div style='border: 1px solid #ddd; margin:10px 0; padding:10px;'>
          <h4 style='margin: 0;'>Prompt</h4>
          <p style='margin: 0;'>{original_prompt}</p>
          <h4 style='margin: 0; margin-top:5px;'>Response</h4>
          <p style='margin: 0;'>{responses[i]}</p>
        </div>
        """

    html_content += "</div>"
    model.train()
    wandb.log({"chat_text_generation": wandb.Html(html_content)}, step=global_step)


def main():
    config = get_config()

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

    total_batch_size_per_gpu = config.training.batch_size_lm
    total_batch_size = total_batch_size_per_gpu * accelerator.num_processes * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

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
        torch.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.tokenizer_path, padding_side="left")

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|mmu|>",),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    logger.info(f"Special tokens:\n{uni_prompting.sptids_dict}")

    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16).to(
        accelerator.device
    )

    mask_id = model.config.mask_token_id

    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if config.optimizer.name == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer.name} not supported")

    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        ms_args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **ms_args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    from parquet import ChatDataset
    dataset_lm = ChatDataset(
        data_path=dataset_config.train_lm_shards_path_or_url,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        num_workers=dataset_config.num_workers,
        max_length=preproc_config.max_seq_length,
        tokenizer=uni_prompting.text_tokenizer,
    )
    train_dataloader_lm = torch.utils.data.DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=dataset_lm.collate_fn,
        num_workers=dataset_config.num_workers,
    )

    iterables = {"lm_flow": train_dataloader_lm}
    combined_mode = getattr(config.dataset, "combined_loader_mode", "max_size_cycle")
    combined_dataloader = CombinedLoader(iterables, mode=combined_mode)

    global_step = 0
    first_epoch = 0
    start_step = 0
    num_update_steps_per_epoch = len(combined_dataloader)

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        if dirs:
            path = os.path.join(config.experiment.output_dir, dirs[-1])
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = start_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            state_file = f"{path}/unwrapped_model/pytorch_model.bin"
            if os.path.exists(state_file):
                state_dict = torch.load(state_file, map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            else:
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, f'{path}/unwrapped_model/')

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    mask_dtype = model.get_input_embeddings().weight.dtype

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels_for_chat_text(
        texts: Union[str, List[str]],
        max_seq_len: int,
        eps: float = 1e-3,
    ):
        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts, max_seq_len), "lm_chat")
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        noisy_batch[prompt_mask.bool()] = input_ids_lm[prompt_mask.bool()]
        answer_lengths_lm = torch.sum((1 - prompt_mask), dim=-1, keepdim=True).repeat(1, l)
        return noisy_batch, labels_lm, p_mask, answer_lengths_lm

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    num_train_epochs = math.ceil(config.training.max_train_steps / len(combined_dataloader))

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in combined_dataloader:
            if global_step >= config.training.max_train_steps:
                break

            data_time_m.update(time.time() - end)
            end = time.time()

            batch_lm = batch["lm_flow"]
            texts_lm = batch_lm.get("text", None)
            if texts_lm is None:
                raise ValueError("ChatDataset must return 'text' in each sample.")

            max_seq_len = config.dataset.preprocessing.max_seq_length
            input_ids, labels, p_mask_lm, answer_lengths_lm = prepare_inputs_and_labels_for_chat_text(
                texts_lm, max_seq_len
            )

            with accelerator.accumulate(model):
                logits, loss_lm = model.forward_process_text_only(
                    input_ids=input_ids,
                    labels=labels,
                    p_mask_lm=p_mask_lm,
                    answer_lengths_lm=answer_lengths_lm,
                )

                avg_loss_lm = accelerator.gather(loss_lm.repeat(config.training.batch_size_lm)).mean()
                loss = config.training.lm_coeff * loss_lm

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                if accelerator.sync_gradients and (global_step + 1) % config.experiment.log_grad_norm_every == 0 \
                        and accelerator.is_main_process:
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    avg_masking_rate = p_mask_lm.mean()
                    logs = {
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
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    batch_time_m.reset()
                    data_time_m.reset()

                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1, uni_prompting)

                if ((global_step + 1) % config.experiment.generate_every == 0 or global_step == start_step) \
                        and accelerator.is_main_process:
                    generate_chat_text(model, uni_prompting, accelerator, config, global_step + 1)

                global_step += 1

    accelerator.wait_for_everyone()

    save_checkpoint(model, config, accelerator, global_step, uni_prompting)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
