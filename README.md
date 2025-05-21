<div align="center">
<br>
<img src="docs/showo_title.png" width="166">
<h3>Multimodal Large Diffusion Language Models</h3></div>

<p align="center">
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/MMaDA-Paper-red?logo=arxiv&logoColor=red"
      alt="MMaDA Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/MMaDA-7B-MoT">
    <img 
        src="https://img.shields.io/badge/MMaDA%20Space-Hugging%20Face%20Space-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/MMaDA-7B-MoT">
    <img 
        src="https://img.shields.io/badge/MMaDA%20Model-Hugging%20Face%20Model-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
</p>


## Introduction
MMaDA is a novel class of **multimodal diffusion foundation models** designed to achieve superior performance across diverse domains such as textual reasoning, multimodal understanding, and text-to-image generation. MMaDA is distinguished by three key innovations:
1. MMaDA adopts a **unified diffusion architecture** with a shared probabilistic formulation and a modality-agnostic design, eliminating the need for modality-specific components.
2. MMaDA introduces a **mixed long chain-of-thought (CoT) fine-tuning** strategy that curates a unified CoT format across modalities.
3. MMaDA adopts a unified policy-gradient-based RL algorithm, which we call **UniGRPO**, tailored for diffusion foundation models. Utilizing diversified reward modeling, **UniGRPO** unifies post-training across both reasoning and generation tasks, ensuring consistent performance improvements.

## Model Overview
MMaDA is a series of multimodal diffusion models. We report three training stages in our paper, and each checkpoint after the stage are:
1. MMaDA-8B-Base: After pretraining and instruction tuning. Capable of basic text generation, image generation, image captioning and **thinking ablities**.
2. MMaDA-8B-MixCoT: After mixed long chain-of-thought (CoT) fine-tuning. Capable of complex textual and multimodal reasoning.
3. MMaDA-8B-Max: After UniGRPO reinforment learning. Excels at complex reasoning and awesome visual generation.

## News


* **[2025-05-22]** We release the inference and training code of MMaDA for text generation, multimodal generation and image generation. 
* **[2025-05-22]** We open source our MMaDA-8B-Base at Huggingface. MMaDA-8B-MixCoT and  MMaDA-8B-Max will be released in the future.
* **[2025-05-22]** We release our research paper for the first unified multimodal diffusion model: MMaDA. 


## TODO
- [ ] Release  MMaDA-8B-MixCoT and MMaDA-8B-Max
- [ ] Release OpenRLHF-based UniGRPO training code.

## Quick Start
First, set up the enviroment:
```
pip install -r requirements.txt
```
Lanuch the local gradio for three tasks:
```
python app.py
```
Or you can just experience it with our Huggingface Demo.

## Train