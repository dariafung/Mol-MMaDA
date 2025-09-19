from transformers import AutoTokenizer
from selfies import get_semantic_robust_alphabet
import os

checkpoint_dir = "/work/hdd/bezp/yfeng7/outputs/mmada-training-stage2-llada-instruct"


tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

new_tokens = sorted(set(get_semantic_robust_alphabet()) - set(tok.get_vocab().keys()))
tok.add_tokens(new_tokens, special_tokens=False)

print("final vocab size =", len(tok))        

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

tok.save_pretrained(checkpoint_dir)
print("Tokenizer files saved to", checkpoint_dir)
