"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-31 13:26:15
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-06 03:45:44
FilePath: /Open-Llama/server.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import torch
import gradio as gr
import sentencepiece as spm
from dataset.tokenizer import Tokenizer
from transformers import LlamaForCausalLM, LlamaConfig


sp_model = spm.SentencePieceProcessor(
    model_file="configs/10w_vocab_guyu_test.model"
)
tokenizer = Tokenizer(sp_model)

raw_model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=1600,
        intermediate_size=6400,
        num_hidden_layers=48,
        num_attention_heads=25,
        initializer_range=0.01,
        pad_token_id=tokenizer.pad_id,
        rms_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        use_stable_embedding=True,
        shared_input_output_embedding=True,
    )
)
ckpt = torch.load(
    "data/saved_ckpt/4800.pt", map_location="cpu"
)
raw_model.load_state_dict(ckpt)
raw_model.eval()
model = raw_model.cuda()
print("ready")


def question_answer(prompt):
    print(prompt)
    raw_inputs = "user:{}\nsystem:".format(prompt)
    inputs_len = len(raw_inputs)
    inputs = tokenizer(raw_inputs, return_tensors=True, add_special_tokens=False)
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    pred = model.generate(**inputs, max_new_tokens=512, do_sample=True)
    pred = tokenizer.decode(pred.cpu())[0]
    pred = pred[inputs_len:]
    print(pred)
    return pred


examples=[
        "帮我写一封邮件，内容是咨询教授本学期量子力学课程的时间表？并且希望教授推荐一些相关书籍",
        "情人节送女朋友什么礼物，预算500",
        "我今天肚子有点不舒服，晚饭有什么建议么",
        "可以总结一下小说三体的核心内容么？",
        "Can you explain to me what quantum mechanics is and how it relates to quantum computing?",
        "请帮我写一个AI驱动的幼儿教育APP的商业计划书",
        "用python实现一个快速排序",
]

for e in examples:
    question_answer(e)
