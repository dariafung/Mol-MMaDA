import torch
import os
import yaml
import safetensors.torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Tuple, List
from tqdm.auto import tqdm
import gc

# RDKit 导入
from rdkit import Chem
from rdkit.Geometry import Point3D

# 本地模型和工具导入
from models import MMadaModelLM
from models.modeling_mmada import MMadaConfig

# --- 辅助函数 ---

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """向 logits 添加 Gumbel 噪声以进行随机采样。"""
    if temperature == 0:
        return logits
    # 使用高精度以保证生成质量
    logits_float64 = logits.to(torch.float64)
    noise = torch.rand_like(logits_float64)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-10))
    return logits_float64 + gumbel_noise * temperature


@torch.no_grad()
def generate_molecular_3d(
    model: MMadaModelLM,
    tokenizer: AutoTokenizer,
    selfies_string: str,
    device: torch.device = 'cuda',
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """
    使用反向扩散过程从 SELFIES 字符串生成 3D 分子坐标和原子类型。
    所有必要的超参数都从 model.config 中读取。
    """
    model.eval()
    config = model.config

    # 1. 准备 SELFIES 输入
    selfies_tokenized = tokenizer(
        selfies_string,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.max_selfies_length
    )
    selfies_input_ids = selfies_tokenized['input_ids'].to(device)
    selfies_attention_mask = selfies_tokenized['attention_mask'].to(device)
    batch_size = selfies_input_ids.shape[0]

    # 2. 初始化噪声数据 (x_T)
    current_coordinates = torch.randn(
        batch_size, config.max_atoms, config.output_atom_coords_dim,
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        device=device
    )
    current_atom_vec = torch.zeros(batch_size, config.max_atoms, dtype=torch.long, device=device)
    atoms_mask = torch.ones(batch_size, config.max_atoms, dtype=torch.bool, device=device)

    # 3. 获取扩散参数
    betas = torch.linspace(
        config.noise_schedule_beta_start, config.noise_schedule_beta_end,
        config.diffusion_timesteps, dtype=torch.float32, device=device
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    sampling_steps = getattr(config, 'generation_timesteps', 30)
    inference_timesteps = torch.linspace(config.diffusion_timesteps - 1, 0, sampling_steps, dtype=torch.long, device=device)

    # 4. 反向扩散（去噪）循环
    for t in inference_timesteps:
        timesteps_tensor = torch.full((batch_size,), t.item(), dtype=torch.long, device=device)

        # 模型前向传播，预测 x_0
        predicted_coordinates_x0, predicted_atom_type_logits, *_ = model.forward(
            selfies_input_ids=selfies_input_ids,
            selfies_attention_mask=selfies_attention_mask,
            atom_vec=current_atom_vec,
            coordinates=current_coordinates,
            atoms_mask=atoms_mask,
            timesteps=timesteps_tensor,
        )

        # 应用原子掩码
        predicted_coordinates_x0 = predicted_coordinates_x0 * atoms_mask.unsqueeze(-1).float()

        # DDPM 采样步骤，从 x_t 和预测的 x_0 计算 x_{t-1}
        if t > 0:
            noise_pred = (current_coordinates - sqrt_alphas_cumprod[t] * predicted_coordinates_x0) / sqrt_one_minus_alphas_cumprod[t]
            alpha_t = alphas[t]
            alpha_bar_t_prev = alphas_cumprod[t-1]
            
            mean_pred = (1 / torch.sqrt(alpha_t)) * (current_coordinates - ((1 - alpha_t) / sqrt_one_minus_alphas_cumprod[t]) * noise_pred)
            variance = (1 - alpha_bar_t_prev) / (1 - alphas_cumprod[t]) * betas[t]
            
            z = torch.randn_like(current_coordinates)
            current_coordinates = mean_pred + torch.sqrt(variance) * z
        else:
            current_coordinates = predicted_coordinates_x0

        # 采样原子类型
        temperature = getattr(config, 'generation_temperature_atom_type', 1.0)
        if temperature > 0:
            sampled_atom_types = add_gumbel_noise(predicted_atom_type_logits, temperature=temperature).argmax(dim=-1)
        else:
            sampled_atom_types = torch.argmax(predicted_atom_type_logits, dim=-1)
        
        current_atom_vec = (sampled_atom_types * atoms_mask).long()

    # 清理 GPU 内存
    gc.collect()
    torch.cuda.empty_cache()
    
    return current_coordinates, current_atom_vec


# --- RDKit 转换辅助函数 ---

_PERIODIC_TABLE = Chem.GetPeriodicTable()

def _type_to_symbol(atomic_number: int) -> str:
    """将原子序数转换为元素符号。"""
    return _PERIODIC_TABLE.GetElementSymbol(int(atomic_number))

def tensors_to_rdmol(types_tensor: torch.Tensor, coords_tensor: torch.Tensor) -> Chem.Mol:
    """从原子类型和坐标张量创建 RDKit Mol 对象。"""
    mol = Chem.RWMol()
    conf = Chem.Conformer(0)
    
    types = types_tensor.cpu().numpy()
    coords = coords_tensor.cpu().numpy()
    
    atom_idx_map = {}
    for i, atomic_num in enumerate(types):
        # 跳过填充原子（原子序数为0）或无效原子
        if not (1 <= atomic_num <= 118):
            continue
        
        rdkit_idx = mol.AddAtom(Chem.Atom(_type_to_symbol(atomic_num)))
        atom_idx_map[i] = rdkit_idx
        
        pos = Point3D(float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
        conf.SetAtomPosition(rdkit_idx, pos)
    
    conf.SetId(0)
    mol.AddConformer(conf, assignId=True)
    
    return mol.GetMol()


# --- 外部调用接口 ---

@torch.no_grad()
def generate_for_evaluation(model: MMadaModelLM, tokenizer: AutoTokenizer, prompts: List[str], device: torch.device) -> List[Chem.Mol]:
    """
    为评估流程设计的生成函数。
    接收一个 prompts 列表，返回一个 RDKit Mol 对象列表。
    """
    all_generated_rdmols = []
    model.eval()

    for selfies_prompt in tqdm(prompts, desc="Generating molecules for evaluation", leave=False, unit="mol"):
        try:
            generated_coords, generated_atom_types = generate_molecular_3d(
                model=model,
                tokenizer=tokenizer,
                selfies_string=selfies_prompt,
                device=device,
            )
            
            # .squeeze(0) 移除批次维度
            rd_mol = tensors_to_rdmol(generated_atom_types.squeeze(0), generated_coords.squeeze(0))
            all_generated_rdmols.append(rd_mol)

        except Exception as e:
            print(f"Error during generation for SELFIES '{selfies_prompt}': {e}", flush=True)
            all_generated_rdmols.append(None) # 如果失败，添加一个 None 占位

    return all_generated_rdmols


# --- 独立脚本运行逻辑 ---

def main():
    """当脚本被直接执行时运行此函数。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 加载配置
    config_path = "configs/mmada_pretraining_stage2_llada_instruct.yaml"
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    class Config: 
        def __init__(self, d):
            for a, b in d.items(): setattr(self, a, Config(b) if isinstance(b, dict) else b)
    args = Config(config_dict)
    
    # 2. 实例化模型配置
    # 使用 `args.model.__dict__` 可以自动传递所有 model 下的参数
    model_config = MMadaConfig(**args.model.__dict__)
    
    # 3. 加载模型权重和 Tokenizer
    checkpoint_dir = getattr(args.experiment, 'resume_from_checkpoint', "/work/hdd/bezp/yfeng7/outputs/mmada-training-stage2-llada-instruct/checkpoint-10000")
    model_state_dict_path = os.path.join(checkpoint_dir, "model.safetensors") 
    print(f"Loading model state dict from: {model_state_dict_path}")

    model = MMadaModelLM(model_config)
    state_dict = safetensors.torch.load_file(model_state_dict_path, device="cpu")
    # 自动处理 accelerate 保存的 'module.' 前缀
    new_state_dict = { (k.replace("module.", "", 1)): v.float() for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model.llm_model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("Model and Tokenizer loaded successfully.")

    # 4. 加载用于生成的输入数据
    data_path = args.model.data_path
    print(f"Loading data to generate from: {data_path}")
    mol_df = pd.read_parquet(data_path)
    selfies_list = mol_df['selfies_string'].tolist()

    all_generated_data = []

    # 5. 执行生成循环
    print(f"Starting 3D molecule generation for {len(selfies_list)} molecules...")
    for idx, selfies in enumerate(tqdm(selfies_list, desc="Generating 3D Molecules", unit="mol")):
        if not isinstance(selfies, str): continue

        try:
            coords_tensor, types_tensor = generate_molecular_3d(
                model=model,
                tokenizer=tokenizer,
                selfies_string=selfies,
                device=device,
            )
            
            coords_np = coords_tensor.squeeze(0).cpu().numpy()
            types_np = types_tensor.squeeze(0).cpu().numpy()

            # 过滤掉填充/无效原子
            valid_mask = (types_np > 0) & (types_np <= 118)
            valid_coords = coords_np[valid_mask].tolist()
            valid_types = types_np[valid_mask].astype(int).tolist()
                
            all_generated_data.append({
                "mol_id": int(mol_df.loc[idx, "id"]) if "id" in mol_df.columns else idx,
                "original_selfies": selfies,
                "generated_coords": valid_coords, 
                "generated_types": valid_types,
            })
        except Exception as e:
            print(f"Error generating for SELFIES (idx {idx}): {e}", flush=True)
            continue

    # 6. 保存结果
    output_path = "generated_3d_molecules_for_evaluation.parquet"
    pd.DataFrame(all_generated_data).to_parquet(output_path, index=False)
    print(f"\nGeneration complete. Saved {len(all_generated_data)} molecules to: {output_path}")

if __name__ == '__main__':
    main()