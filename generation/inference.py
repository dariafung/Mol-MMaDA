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
from data.my_dataset import parse_molecular_3d_data

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
        processed_mol_data = parse_molecular_3d_data(mol_data_row)
        if not processed_mol_data:
            print(f"Skipping row {i}: Failed to parse molecular data.")
            continue

        atom_vec = processed_mol_data["atom_vec"].to(device).unsqueeze(0) 
        coordinates = processed_mol_data["coordinates"].to(device).unsqueeze(0) 
        atoms_mask = processed_mol_data["atoms_mask"].to(device).unsqueeze(0) 
        selfies_input_ids_mol = processed_mol_data["selfies_input_ids"].to(device).unsqueeze(0)
        selfies_attention_mask_mol = processed_mol_data["selfies_attention_mask"].to(device).unsqueeze(0) 

        for question_text in prompts_for_each_molecule: 
            question_tokenized = uni_prompting.text_tokenizer([question_text], return_tensors="pt")
            question_input_ids = question_tokenized.input_ids.to(device)
            question_attention_mask = question_tokenized.attention_mask.to(device)

            input_ids_for_model = torch.cat([
                (torch.ones(1, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device), 
                (torch.ones(1, 1) * uni_prompting.sptids_dict['<|som|>']).to(device), 
                selfies_input_ids_mol, 
                (torch.ones(1, 1) * uni_prompting.sptids_dict['<|eom|>']).to(device), 
                (torch.ones(1, 1) * uni_prompting.sptids_dict['<|sot|>']).to(device), 
                question_input_ids, 
            ], dim=1).long()

            with torch.no_grad():
                output_ids = model.mmu_generate(
                    input_ids_for_model, 
                    max_new_tokens=config.dataset.preprocessing.max_seq_length,
                    steps=config.dataset.preprocessing.max_lm_text_length // 2, 
                    block_length=config.dataset.preprocessing.max_seq_length // 4 
                )

            generated_text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids_for_model.shape[1]:], skip_special_tokens=True)[0]

            all_molecules_responses[i].append(f'User: {question_text}\n Answer : {generated_text}')

html_content = "<div style='font-family:Arial, sans-serif;'>"
html_content += f"<h2 style='color:navy;'>Molecular Understanding Inference Results</h2>"
for i, mol_responses in enumerate(all_molecules_responses):
    if not mol_responses: 
        continue
    html_content += f"<h3>Molecule {i+1}</h3>"
    for resp in mol_responses:
        escaped_resp = html.escape(resp)
        html_content += f"<p style='border: 1px solid #ddd; margin:5px 0; padding:5px;'>{escaped_resp}</p>"
html_content += "</div>"

wandb.log({"Molecular Understanding Results": wandb.Html(html_content)}, step=0)