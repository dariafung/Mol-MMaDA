import torch
import os
import yaml
import safetensors.torch
import numpy as np
import pandas as pd  
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
from tqdm.auto import tqdm 

from models import MMadaModelLM
from models.modeling_mmada import MMadaConfig 
from training.utils import get_noise_schedule, get_mask_schedule, apply_selfies_masking 

import gc
from torch.cuda.amp import autocast

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, attention_mask=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L), where B is batch size.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        print(f"attention_bias: {attention_bias}")
    else:
        attention_bias = None
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_bias=attention_bias).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            # print(confidence.shape)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

@torch.no_grad()
def generate_molecular_3d(
    model: MMadaModelLM,
    tokenizer: AutoTokenizer,
    selfies_string: str,
    max_selfies_length: int,
    max_atoms: int, # From MMadaConfig.max_atoms
    num_atom_types: int, # From MMadaConfig.num_atom_types
    output_atom_coords_dim: int, # From MMadaConfig.output_atom_coords_dim (e.g., 3 for x,y,z)
    diffusion_timesteps: int, # From MMadaConfig.diffusion_timesteps
    noise_schedule_beta_start: float, # From MMadaConfig.noise_schedule_beta_start
    noise_schedule_beta_end: float, # From MMadaConfig.noise_schedule_beta_end
    sampling_steps: int = 30, # Number of inference steps, can be less than diffusion_timesteps
    temperature_atom_type: float = 1.0, # Temperature for sampling atom types (discrete)
    device: torch.device = 'cuda',
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """
    Generates 3D molecular coordinates and atom types from a SELFIES string using a reverse diffusion process.

    Args:
        model: The MMadaModelLM instance.
        tokenizer: The tokenizer for SELFIES.
        selfies_string: The input SELFIES string (e.g., '[C][C]O').
        max_atoms: Maximum number of atoms the model can handle.
        num_atom_types: Total number of possible atom types.
        output_atom_coords_dim: Dimension of atom coordinates (e.g., 3 for x, y, z).
        diffusion_timesteps: Total timesteps used during diffusion training.
        noise_schedule_beta_start: Beta start for noise schedule.
        noise_schedule_beta_end: Beta end for noise schedule.
        sampling_steps: Number of steps for the reverse diffusion sampling process.
                        Can be less than diffusion_timesteps for faster inference.
        temperature_atom_type: Temperature for Gumbel sampling of atom types.
        device: The device to run inference on ('cuda' or 'cpu').

    Returns:
        Tuple[torch.FloatTensor, torch.LongTensor]:
            - Final predicted 3D coordinates (batch_size=1, max_atoms, output_atom_coords_dim)
            - Final predicted atom types (batch_size=1, max_atoms)
    """
    model.eval()

    # 1. Prepare SELFIES input
    selfies_input_ids = tokenizer(selfies_string, return_tensors="pt", padding="max_length",
                                  truncation=True, max_length=model.config.max_selfies_length )['input_ids'].to(device)
    selfies_attention_mask = (selfies_input_ids != tokenizer.pad_token_id).long().to(device)
    
    batch_size = selfies_input_ids.shape[0]

    # 2. Initialize noisy 3D data (random coordinates, unknown/padding atom types)
    # Start with random noise for coordinates (x_T)
    current_coordinates = torch.randn(batch_size, max_atoms, output_atom_coords_dim, dtype=torch.float16, device=device)
    
    # Initialize atom types. Assuming 0 is a padding/unknown atom type.
    # We could also use a specific mask_id for atom types if defined in config/tokenizer vocab
    current_atom_vec = torch.zeros(batch_size, max_atoms, dtype=torch.long, device=device)
    
    # All atoms are initially 'active' or 'fillable', mask for valid atoms will be based on generated structure
    atoms_mask = torch.ones(batch_size, max_atoms, dtype=torch.bool, device=device)

    # 3. Get Diffusion Parameters
    # Recalculate betas, alphas, etc. as get_noise_schedule only returns the schedule_fn
    betas = torch.linspace(noise_schedule_beta_start, noise_schedule_beta_end, diffusion_timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Simplified steps for inference
    # We will sample `sampling_steps` times
    # Choose `t` values evenly spaced from `diffusion_timesteps` down to 1
    inference_timesteps = torch.linspace(diffusion_timesteps - 1, 0, sampling_steps, dtype=torch.long).to(device)

    # 4. Reverse Diffusion (Denoising) Loop
    for i in range(sampling_steps):
        t = inference_timesteps[i] # Current timestep (integer)
        
        # Prepare timestep tensor for model input (needs to be batch_size)
        timesteps_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)

        # Forward pass through MMadaModelLM to get predictions (x_0_pred, atom_type_logits_pred)
        # current_coordinates here acts as x_t
        with autocast(dtype=torch.float16):              # AMP 开启
            predicted_coordinates_x0, predicted_atom_type_logits, *_ = model.forward(
                selfies_input_ids=selfies_input_ids,
                selfies_attention_mask=selfies_attention_mask,
                atom_vec=current_atom_vec,
                coordinates=current_coordinates,
                atoms_mask=atoms_mask,
                timesteps=timesteps_tensor,
                text_input_ids=None,
                text_attention_mask=None,
            )
        # Apply mask to predictions
        predicted_coordinates_x0 = predicted_coordinates_x0 * atoms_mask.unsqueeze(-1).float()
        predicted_atom_type_logits = predicted_atom_type_logits * atoms_mask.unsqueeze(-1).float() # Mask logits

        # DDPM sampling step to get x_{t-1} from x_t and x_0_pred
        # Calculate alpha_t, alpha_bar_t for current t
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        
        # Calculate predicted noise from x_t and x_0_pred
        noise_pred = (current_coordinates - sqrt_alphas_cumprod[t] * predicted_coordinates_x0) / sqrt_one_minus_alphas_cumprod[t]
        
        if t > 0:
            # Calculate alpha_t_minus_1 and alpha_bar_t_minus_1
            alpha_t_minus_1 = alphas[t-1]
            alpha_bar_t_minus_1 = alphas_cumprod[t-1]
            
            # Mean for x_{t-1}
            mean = (current_coordinates - betas[t] * noise_pred / sqrt_one_minus_alphas_cumprod[t]) / torch.sqrt(alpha_t)
            
            # Variance for x_{t-1}
            # This is the original DDPM posterior variance
            variance = betas[t] * (1.0 - alphas_cumprod_prev[t]) / (1.0 - alpha_bar_t)
            
            # Sample from N(mean, variance * I)
            z = torch.randn_like(current_coordinates) # Standard normal noise
            current_coordinates = (mean + torch.sqrt(variance) * z) * atoms_mask.unsqueeze(-1).float()
        else: # t = 0, final step
            current_coordinates = predicted_coordinates_x0 # Directly use the predicted x_0

        # Sample atom types using Gumbel-softmax or argmax
        if temperature_atom_type > 0:
            # Apply Gumbel noise and then argmax for sampling
            sampled_atom_types = add_gumbel_noise(predicted_atom_type_logits, temperature=temperature_atom_type).argmax(dim=-1)
        else:
            # Deterministic argmax
            sampled_atom_types = torch.argmax(predicted_atom_type_logits, dim=-1)
        
        # Ensure sampled atom types are within valid range (0 to num_atom_types-1)
        # and apply atom mask
        current_atom_vec = (sampled_atom_types * atoms_mask).long()

    del predicted_coordinates_x0, predicted_atom_type_logits, sampled_atom_types, noise_pred
    torch.cuda.empty_cache()
    gc.collect()
    
    return current_coordinates, current_atom_vec

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. 从原始 YAML 配置文件加载完整的模型配置 ---
    original_config_path = "configs/mmada_pretraining_stage2_llada_instruct.yaml" 
    
    try:
        with open(original_config_path, "r") as f:
            original_config_dict = yaml.safe_load(f)
        
        # 这是一个辅助类，用于方便地通过属性访问 YAML 参数
        class Config: 
            def __init__(self, d):
                for a, b in d.items():
                    if isinstance(b, dict):
                        setattr(self, a, Config(b))
                    else:
                        setattr(self, a, b)
        
        args = Config(original_config_dict)

        # 从 YAML 参数直接构建 MMadaConfig 实例
        # 这样可以确保模型结构与原始训练时完全一致
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
            selfies_coeff=args.model.selfies_coeff,
            alignment_coeff=args.model.alignment_coeff,
            hierarchical_coeff=args.model.hierarchical_coeff,
            mask_token_id=args.model.mask_token_id,
            mask_replace_ratio=args.model.mask_replace_ratio,
            mask_schedule_name=args.model.mask_schedule_name,
            mask_schedule_start=args.model.mask_schedule_start,
            mask_schedule_end=args.model.mask_schedule_end,
            # 请确保这里包含了 MMadaConfig 构造函数所需的**所有**参数
            # 这些参数通常在 configs/mmada_pretraining_stage2_llada_instruct.yaml 的 'model' 部分定义
        )
        
        print(f"Loaded MMadaConfig directly from original YAML.")

    except Exception as e:
        print(f"Error loading original config file from {original_config_path} or creating MMadaConfig: {e}")
        print("请确保原始训练配置文件存在且路径正确，且包含所有必要的模型参数。")
        return


    # --- 2. 配置模型检查点路径 ---
    checkpoint_dir = "/media/volume/MMaDA/outputs/mmada-training-stage2-llada-instruct/checkpoint-10000" 
    model_state_dict_path = os.path.join(checkpoint_dir, "model.safetensors") 

    print(f"Loading model state dict from: {model_state_dict_path}")

    # 2.1 实例化模型（现在使用从 YAML 创建的 model_config）
    model = MMadaModelLM(model_config).half()

    # 2.2 加载模型权重（state_dict）
    try:
        state_dict = safetensors.torch.load_file(model_state_dict_path, device="cpu")
        
        new_state_dict = {
            (k[len("module."): ] if k.startswith("module.") else k): v.half()  # ☆ 转 FP16
            for k, v in state_dict.items()
        }
        model.load_state_dict(new_state_dict)

        model.to(device).eval()                        # 搬到 GPU
        torch.set_float32_matmul_precision("high")
    except Exception as e:
        print(f"Error loading model state dict from {model_state_dict_path}: {e}")
        print("请确保 model.safetensors 文件存在且是一个有效的模型 state_dict。")
        return

    # 2.3 加载 tokenizer 
    try:
        root_dir  = os.path.dirname(checkpoint_dir)        # ← ★ 新增
        tokenizer = AutoTokenizer.from_pretrained(
            root_dir,                                      # ← ★ 用父目录
            trust_remote_code=True                         # ← ★ 保险起见
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer from {checkpoint_dir}: {e}")
        print("请确保此目录包含 tokenizer 相关文件（如 tokenizer_config.json, vocab.json 等）。")
        return

    print("Model and Tokenizer loaded successfully.")

    # ... (以下是加载数据集和生成循环的代码，与之前提供的版本相同) ...
    data_to_generate_path = "/media/volume/MMaDA/data/m3_molecular_data.parquet" 
    print(f"Loading data from: {data_to_generate_path}")

    try:
        mol_df = pd.read_parquet(data_to_generate_path)
    except Exception as e:
        print(f"Error reading parquet file from {data_to_generate_path}: {e}")
        print("Please ensure the data path is correct and the file exists.")
        return
    
    if 'selfies_string' not in mol_df.columns:
        print("Error: 'selfies_string' column not found in the Parquet file.")
        print("Please check your Parquet file's column names and update the script accordingly.")
        return
    
    selfies_list = mol_df['selfies_string'].tolist()

    all_generated_molecules_data = []

    print(f"\nStarting 3D molecule generation for {len(selfies_list)} molecules...")
    for idx, selfies_to_generate in enumerate(tqdm(selfies_list, desc="Generating 3D Molecules")):
        if not isinstance(selfies_to_generate, str):
            print(f"Skipping index {idx}: SELFIES is not a string. Value: {selfies_to_generate}")
            continue

        try:
            generated_coords, generated_atom_types = generate_molecular_3d(
                model=model,
                tokenizer=tokenizer,
                selfies_string=selfies_to_generate,
                max_selfies_length=model_config.max_selfies_length,
                max_atoms=model_config.max_atoms, 
                num_atom_types=model_config.num_atom_types, 
                output_atom_coords_dim=model_config.output_atom_coords_dim, 
                diffusion_timesteps=model_config.diffusion_timesteps, 
                noise_schedule_beta_start=model_config.noise_schedule_beta_start, 
                noise_schedule_beta_end=model_config.noise_schedule_beta_end, 
                sampling_steps=30,
                temperature_atom_type=0.5,
                device=device,
            )
            
            all_generated_molecules_data.append({
                'mol_id': mol_df.loc[idx, 'id'] if 'id' in mol_df.columns else idx
                'original_selfies': selfies_to_generate,

                "generated_coords": generated_coords.squeeze(0).cpu().numpy().tolist(),
                "generated_atom_types": generated_atom_types.squeeze(0).cpu().numpy().tolist(),
                
            })
        except Exception as e:
            print(f"Error generating for SELFIES '{selfies_to_generate}' (index {idx}): {e}")
            continue

    output_parquet_path = "generated_3d_molecules_for_evaluation.parquet"
    pd.DataFrame(all_generated_molecules_data).to_parquet(output_parquet_path, index=False)
    print(f"\nAll generated 3D molecules saved to: {output_parquet_path}")
    print("您现在可以使用此 Parquet 文件中的生成数据进行评估。")

if __name__ == '__main__':
    main()
