import os
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
    # You might need to adjust or create a specific function for logging images/metrics
    # based on the new 1D+3D task
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
    # This is crucial. Use the actual tokenizer path for LLaDA.
    # The tokenizer object will be passed to MolecularUnifiedDataset.
    from transformers import AutoTokenizer
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(args.model.llm_model_name_or_path)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token # Or specify a pad token
        # Ensure mask_token_id is correctly set in model_config based on tokenizer's vocab
        if llm_tokenizer.mask_token_id is None and model_config.mask_token_id == -1:
            accelerator.print("Warning: LLM tokenizer does not have a mask_token_id. Please ensure model_config.mask_token_id is set to a valid ID or add a mask token to the tokenizer.")
            # You might need to add a new token and resize token embeddings in the model
            # For simplicity, if mask_token_id in config is still -1, let's use a dummy or a special token.
            # A more robust solution involves adding a special token to the tokenizer vocab and resizing embeddings.
            # For now, let's just make sure it's not -1 when passed to dataset, if llm_tokenizer doesn't have one
            # if no mask token in tokenizer, you could designate an unused token as mask or add one.
            # Example if you add it: llm_tokenizer.add_special_tokens({'mask_token': '<mask>'})
            # model.llm_backbone.resize_token_embeddings(len(llm_tokenizer))
            pass # Handle mask token ID carefully
        else:
            if model_config.mask_token_id == -1: # If config didn't specify, use tokenizer's
                model_config.mask_token_id = llm_tokenizer.mask_token_id
            
    except Exception as e:
        accelerator.print(f"Error loading LLM tokenizer from {args.model.llm_model_name_or_path}: {e}")
        accelerator.print("Falling back to a dummy tokenizer. This might cause issues.")
        from parquet.my_dataset import DummyTokenizer # Using the dummy one if real fails
        llm_tokenizer = DummyTokenizer()
        model_config.mask_token_id = llm_tokenizer.mask_token_id # Ensure dummy's mask token is used

    accelerator.print("Tokenizer loaded.")
    
    # --- 3. Prepare Dataset and DataLoader ---
    accelerator.print(f"Loading data from {args.model.data_path}")
    train_dataset = MolecularUnifiedDataset(
        data_path=args.model.data_path,
        max_text_length=args.dataset.preprocessing.max_text_length,
        max_atoms=args.model.max_atoms,
        mask_token_id=model_config.mask_token_id,
        mask_schedule_name=model_config.mask_schedule_name,
        mask_schedule_start=model_config.mask_schedule_start,
        mask_schedule_end=model_config.mask_schedule_end,
        selfies_mask_ratio=args.model.selfies_mask_ratio, # New parameter for selfies masking in dataset
        # atom_type_mapping is no longer needed here as atom_to_id handles it
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.per_device_train_batch_size,
        collate_fn=train_dataset.collate_fn, # Use the static collate_fn
        num_workers=args.data.num_workers,
        pin_memory=True,
    )
    
    # --- 4. Optimizer and Scheduler ---
    optimizer = get_optimizer(model, args.optimizer.learning_rate, args.optimizer.weight_decay)
    lr_scheduler = get_scheduler(
        name=args.optimizer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.optimizer.num_warmup_steps,
        num_training_steps=args.training.max_train_steps,
    )

    # --- 5. Accelerator Preparation ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Define noise schedule for continuous coordinates diffusion
    mask_schedule_coords = get_noise_schedule(
        name=args.model.noise_schedule_name, # Assuming this is also defined in config
        beta_start=args.model.noise_schedule_beta_start,
        beta_end=args.model.noise_schedule_beta_end,
        timesteps=args.model.diffusion_timesteps,
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

        # Move batch to device
        # All inputs are now in batch dict, including new ones
        
        # Prepare model inputs and true labels
        model_inputs = {
            "selfies_input_ids": batch["selfies_input_ids"],
            "selfies_attention_mask": batch["selfies_attention_mask"],
            "text_input_ids": batch["text_input_ids"], # Can be None if not applicable
            "text_attention_mask": batch["text_attention_mask"], # Can be None if not applicable
            "atom_vec": batch["atom_vec"],
            "coordinates": batch["coordinates"], # This is the clean coordinates (x_0)
            "atoms_mask": batch["atoms_mask"],
            "task_type": batch["task_type"],
            "true_coordinates": batch["coordinates"], # True coordinates for loss (same as coordinates initially)
            "true_atom_vec": batch["atom_vec"], # True atom types for loss (same as atom_vec initially)
            "true_selfies_labels": batch["true_selfies_labels"], # NEW: True SELFIES for loss
            "mask_schedule_coords": mask_schedule_coords, # For continuous diffusion
            "timesteps": None, # Will be sampled inside forward_process if None
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

        if step % args.training.save_steps == 0 and step > 0:
            if accelerator.is_main_process:
                output_dir = os.path.join(args.training.output_dir, f"checkpoint-{step}")
                accelerator.save_state(output_dir)
                accelerator.print(f"Saved checkpoint to {output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()