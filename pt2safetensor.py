import json
import time

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file
from trinity.common.models.utils import load_state_dict_from_verl_checkpoint

tokenizer = AutoTokenizer.from_pretrained('/mnt/checkpoint/qwen25/Qwen2.5-7B-Instruct')
model = AutoModelForCausalLM.from_pretrained('/mnt/checkpoint/qwen25/Qwen2.5-7B-Instruct')

test_model = "20250519-2"
test_step = "step_561"

ckp_path = f"/mnt/feiwei/checkpoints/qwen-7b-grpo-{test_model}/global_{test_step}/actor/"
model.load_state_dict(load_state_dict_from_verl_checkpoint(ckp_path))
output_dir = f"/mnt/feiwei/checkpoints/qwen-7b-grpo-{test_model}/global_{test_step}_converted/"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
model.config.save_pretrained(output_dir)
