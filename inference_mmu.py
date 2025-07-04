# coding=utf-8
# Copyright 2025 MMaDA Team
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
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from tqdm import tqdm
import torch
import wandb
import pandas
import html
from models import MMadaConfig, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, flatten_omega_conf
from transformers import AutoTokenizer, AutoConfig
from parquet.my_dataset import parse_molecular_3d_data

if __name__ == '__main__':

    config = get_config()
    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name + '_mmu',
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|mmu|>", 
                                            "<|som|>", 
                                            "<|eom|>", 
                                            "<|sot|>", 
                                            "<|sov|>",
                                            "<|eov|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)
    
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device)

    mask_token_id = model.config.mask_token_id

    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability
    mol_data_df = pandas.read_parquet(config.mol_inference_data_path)

    prompts_for_each_molecule = config.question

    all_molecules_responses = [[] for _ in range(len(mol_data_df))] 

    for i, mol_data_row in enumerate(tqdm(mol_data_df.to_dict(orient='records'), desc="Processing Molecules")):
        # 解析原始分子数据，确保 parse_molecular_3d_data 函数可用
        processed_mol_data = parse_molecular_3d_data(mol_data_row)
        if not processed_mol_data:
            print(f"Skipping row {i}: Failed to parse molecular data.")
            continue

        # 将分子数据张量移动到设备
        atom_vec = processed_mol_data["atom_vec"].to(device).unsqueeze(0) # 添加 batch 维度
        coordinates = processed_mol_data["coordinates"].to(device).unsqueeze(0) # 添加 batch 维度
        atoms_mask = processed_mol_data["atoms_mask"].to(device).unsqueeze(0) # 添加 batch 维度
        selfies_input_ids_mol = processed_mol_data["selfies_input_ids"].to(device).unsqueeze(0) # 添加 batch 维度
        selfies_attention_mask_mol = processed_mol_data["selfies_attention_mask"].to(device).unsqueeze(0) # 添加 batch 维度
        # 注意：如果你的 parse_molecular_3d_data 返回的是没有批次维度的张量，这里需要 unsqueeze(0)

        # 为每个分子生成回答
        for question_text in prompts_for_each_molecule: # 遍历你为每个分子准备的问题
            # 1. 准备文本提示 token
            question_tokenized = uni_prompting.text_tokenizer([question_text], return_tensors="pt")
            question_input_ids = question_tokenized.input_ids.to(device)
            question_attention_mask = question_tokenized.attention_mask.to(device) # 如果需要

            # 2. 构建输入 token 序列 (文本问题 + 分子 token)
            # 格式：<|mmu|> <|som|> [SELFIES tokens] <|eom|> <|sot|> [问题 token]

            # 你的模型 MMadaModelLM.mmu_generate 方法可能需要调整以直接处理这些，或者它内部会转换
            # 这里假设 mmu_generate 可以处理这种连接方式
            input_ids_for_model = torch.cat([
                (torch.ones(1, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device), # MMU 任务 token
                (torch.ones(1, 1) * uni_prompting.sptids_dict['<|som|>']).to(device), # 分子开始 token
                selfies_input_ids_mol, # SELFIES token
                (torch.ones(1, 1) * uni_prompting.sptids_dict['<|eom|>']).to(device), # 分子结束 token
                (torch.ones(1, 1) * uni_prompting.sptids_dict['<|sot|>']).to(device), # 文本开始 token
                question_input_ids, # 文本问题 token
            ], dim=1).long()

            # 3. 调用模型进行生成
            with torch.no_grad():
                output_ids = model.mmu_generate(
                    input_ids_for_model, # 使用新的输入序列
                    max_new_tokens=config.dataset.preprocessing.max_seq_length,
                    steps=config.dataset.preprocessing.max_lm_text_length // 2, # 这个参数名称可能需要根据模型调整
                    block_length=config.dataset.preprocessing.max_seq_length // 4 # 这个参数名称可能需要根据模型调整
                )

            # 4. 解码生成的文本
            generated_text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids_for_model.shape[1]:], skip_special_tokens=True)[0]

            # 收集回答
            all_molecules_responses[i].append(f'User: {question_text}\n Answer : {generated_text}')

html_content = "<div style='font-family:Arial, sans-serif;'>"
html_content += f"<h2 style='color:navy;'>Molecular Understanding Inference Results</h2>"
for i, mol_responses in enumerate(all_molecules_responses):
    if not mol_responses: # 跳过没有生成回答的分子
        continue
    html_content += f"<h3>Molecule {i+1}</h3>"
    for resp in mol_responses:
        escaped_resp = html.escape(resp)
        html_content += f"<p style='border: 1px solid #ddd; margin:5px 0; padding:5px;'>{escaped_resp}</p>"
html_content += "</div>"

wandb.log({"Molecular Understanding Results": wandb.Html(html_content)}, step=0)