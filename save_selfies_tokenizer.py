from transformers import AutoTokenizer
from selfies import get_semantic_robust_alphabet
import os

checkpoint_dir = "/media/volume/MMaDA/outputs/mmada-training-stage2-llada-instruct"

# 1) 基座 LLaDA tokenizer
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

# 2) 加入训练时用的 SELFIES token
new_tokens = sorted(set(get_semantic_robust_alphabet()) - set(tok.get_vocab().keys()))
tok.add_tokens(new_tokens, special_tokens=False)

print("final vocab size =", len(tok))          # 应该打印 134656

# 3) 设置 pad_token（训练里也这样做）
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 4) 保存进 checkpoint 目录
tok.save_pretrained(checkpoint_dir)
print("Tokenizer files saved to", checkpoint_dir)
