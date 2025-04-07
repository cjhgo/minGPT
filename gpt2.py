import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2" # 举个栗子
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
import ipdb;ipdb.set_trace()
# 首次调用 (Prefill + 生成第一个 Token)
# use_cache=True 会让模型返回计算好的 past_key_values
outputs = model(**inputs, use_cache=True, return_dict=True)
logits = outputs.logits
past_key_values = outputs.past_key_values # 这就是 KV Cache！

# 获取预测的下一个 Token ID (简单用 argmax 示例)
next_token_id = logits[:, -1:, :].argmax(dim=-1)

# 解码循环 (生成后续 Token)
for _ in range(10): # 生成 10 个 Token
    # 注意！下一次输入的 inputs 只有 *最新* 的 token_id
    # 并且把上一轮得到的 past_key_values 传回去！
    inputs = {"input_ids": next_token_id,
              "attention_mask": torch.ones_like(next_token_id), # 注意力掩码也只需关注当前输入
              "past_key_values": past_key_values, # 把缓存传回去！
              "use_cache": True} # 继续使用并更新缓存

    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values # 获取 *更新后* 的 KV Cache

    # 获取下一个 Token ID
    next_token_id = logits[:, -1:, :].argmax(dim=-1)

    # 打印生成的 Token (需要解码)
    print(tokenizer.decode(next_token_id[0]), end="")