import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Tuple

from models import MMadaModelLM
from models.modeling_mmada import MMadaConfig 
from training.utils import get_noise_schedule, get_mask_schedule, apply_selfies_masking 

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
    # Optional: text_prompt: str = None, # If you want to condition on text too
    max_atoms: int, # From MMadaConfig.max_atoms
    num_atom_types: int, # From MMadaConfig.num_atom_types
    output_atom_coords_dim: int, # From MMadaConfig.output_atom_coords_dim (e.g., 3 for x,y,z)
    diffusion_timesteps: int, # From MMadaConfig.diffusion_timesteps
    noise_schedule_beta_start: float, # From MMadaConfig.noise_schedule_beta_start
    noise_schedule_beta_end: float, # From MMadaConfig.noise_schedule_beta_end
    sampling_steps: int = 100, # Number of inference steps, can be less than diffusion_timesteps
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
                                  truncation=True, max_length=model.config.max_position_embeddings)['input_ids'].to(device)
    selfies_attention_mask = (selfies_input_ids != tokenizer.pad_token_id).long().to(device)
    
    batch_size = selfies_input_ids.shape[0]

    # 2. Initialize noisy 3D data (random coordinates, unknown/padding atom types)
    # Start with random noise for coordinates (x_T)
    current_coordinates = torch.randn(batch_size, max_atoms, output_atom_coords_dim, device=device)
    
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
        predicted_coordinates_x0, predicted_atom_type_logits, _, _, _, _ = model.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            atom_vec=current_atom_vec, # Pass current (noisy/refined) atom types
            coordinates=current_coordinates, # Pass current (noisy) coordinates
            atoms_mask=atoms_mask, # Pass current atom mask
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

    return current_coordinates, current_atom_vec

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load model and tokenizer from a checkpoint
    # You will need to replace this with your actual model path
    model_path = "/data_storage/ty/MMaDA/mmada-training-stage4-llada-instruct/checkpoint-170000/unwrapped_model"
    
    # Load MMadaConfig from the model's pretrained config
    config = MMadaConfig.from_pretrained(model_path)

    model = MMadaModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Model and Tokenizer loaded successfully.")

    # --- Example usage of generate_molecular_3d ---
    # Replace with a real SELFIES string for a molecule you want to generate
    # Example SELFIES for water: [H][O][H]
    # Example SELFIES for methane: [C]([H])([H])([H])[H]
    selfies_to_generate = "[C][C](O)[N]" # Example: Alanine backbone or similar

    print(f"\nGenerating 3D structure for SELFIES: '{selfies_to_generate}'")

    generated_coords, generated_atom_types = generate_molecular_3d(
        model=model,
        tokenizer=tokenizer,
        selfies_string=selfies_to_generate,
        max_atoms=config.max_atoms,
        num_atom_types=config.num_atom_types,
        output_atom_coords_dim=config.output_atom_coords_dim,
        diffusion_timesteps=config.diffusion_timesteps,
        noise_schedule_beta_start=config.noise_schedule_beta_start,
        noise_schedule_beta_end=config.noise_schedule_beta_end,
        sampling_steps=100, # Use fewer steps for faster inference
        temperature_atom_type=0.5, # Adjust temperature for atom type sampling
        device=device,
    )

    print("\nGenerated 3D Coordinates (first 5 atoms):")
    print(generated_coords[0, :5]) # Print for the first molecule, first 5 atoms

    print("\nGenerated Atom Types (first 5 atoms):")
    print(generated_atom_types[0, :5]) # Print for the first molecule, first 5 atoms

    # You might want to save these to a file (e.g., .xyz, .pdb) for visualization
    # This part is commented out as it requires external libraries like openbabel or a custom atom_type_map
    # import openbabel # You would need to install openbabel bindings for Python
    # from openbabel import OBMol, OBAtom, OBResidue

    # def save_xyz(coords, atom_types, filename, atom_type_map):
    #     with open(filename, 'w') as f:
    #         num_atoms = (atom_types != 0).sum().item() # Assuming 0 is padding
    #         f.write(f"{num_atoms}\n")
    #         f.write("Generated by MMaDA\n")
    #         for i in range(max_atoms):
    #             if atom_types[0, i] != 0: # Check if it's a valid atom
    #                 atom_symbol = atom_type_map.get(atom_types[0, i].item(), 'X') # Map ID to symbol
    #                 f.write(f"{atom_symbol} {coords[0, i, 0]:.4f} {coords[0, i, 1]:.4f} {coords[0, i, 2]:.4f}\n")

    # # Example atom type mapping (you need to define your actual mapping)
    # # This mapping should correspond to how your atom types are indexed during training
    # atom_type_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'} # Partial example
    # save_xyz(generated_coords, generated_atom_types, "generated_molecule.xyz", atom_type_map)
    # print("\nSaved generated molecule to generated_molecule.xyz (requires accurate atom_type_map).")


if __name__ == '__main__':
    main()
