import json
import time

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file
from trinity.common.models.utils import load_state_dict_from_verl_checkpoint

tokenizer = AutoTokenizer.from_pretrained('/mnt/checkpoint/qwen25/Qwen2.5-7B-Instruct')
model = AutoModelForCausalLM.from_pretrained('/mnt/checkpoint/qwen25/Qwen2.5-7B-Instruct')

test_model = "20250516-1"
ckp_path = f"/mnt/feiwei/checkpoints/qwen-7b-grpo-{test_model}/global_step_120/actor/"
model.load_state_dict(load_state_dict_from_verl_checkpoint(ckp_path))

output_dir = "/mnt/feiwei/checkpoints/qwen-7b-grpo-20250516-1/global_step_120/huggingface2"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
model.config.save_pretrained(output_dir)


# input_file_path = "/mnt/feiwei/data_processing/med_grpo_datasets/test.jsonl"
# print(f"loading data from {input_file_path}...")
# with open(input_file_path, "r") as lines:
#     sample_list = [json.loads(line.strip()) for line in lines]
#
# output_file_path = f"/mnt/feiwei/data_processing/med_grpo_test_results/{test_model}.jsonl"
# print(f"result will be saved to {output_file_path}...")

# rollout_repeat = 10
#
# from trinity.common.elem_prompts import test_prompt_v2e1 as sys_prompt
# print(sys_prompt)
# for sample in sample_list[:50]:
#     user_input = sample["prompt"]
#     messages = [{"role": "system", "content": sys_prompt},
#                 {"role": "user", "content": user_input}]
#     prompt = tokenizer.apply_chat_template(messages,
#                                            tokenize=False,
#                                            add_generation_prompt=True)
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
#     rollout_list = []
#     with torch.no_grad():
#         time_start = time.perf_counter()
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_new_tokens=512,         # 控制最大生成长度
#             temperature=1.0,            # 温度参数，控制输出多样性
#             top_p=0.95,                 # nucleus sampling，限制采样范围
#             do_sample=True,              # 启用采样（避免总是输出相同内容）
#             num_return_sequences=5
#         )
#         responses = [
#             tokenizer.decode(
#             outputs[0][inputs["input_ids"].shape[-1]:],  # 只取生成的部分
#             skip_special_tokens=True                     # 跳过特殊 token（如 <eos>）
#             )]
#     sample["rollouts"] = responses
#
#     with open(output_file_path, "a") as f:
#         f.write(json.dumps(sample, ensure_ascii=False) + "\n")
#     print(json.dumps(sample, ensure_ascii=False, indent=2))