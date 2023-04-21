"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-24 20:49:03
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-05 22:40:29
FilePath: /Open-Llama/dataset/train_tokenizer.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import random
from data_iter import DataIter, create_shard_kwargs

wudao_patterns = [
    "data/pretrain_data/part-wudao-*.jsonl.zst",
]
wudao_paths = create_shard_kwargs(wudao_patterns)
random.shuffle(wudao_paths)

pile_patterns = [
    "data/pretrain_data/part-pile-*.jsonl.zst",
]
pile_paths = create_shard_kwargs(pile_patterns)
random.shuffle(pile_paths)

pnews_patterns = [
    "data/pretrain_data/part-pnews-*.jsonl.zst",
]
pnews_paths = create_shard_kwargs(pnews_patterns)
random.shuffle(pnews_paths)

pbaike_patterns = [
    "data/pretrain_data/part-pbaike-*.jsonl.zst",
]
pbaike_paths = create_shard_kwargs(pbaike_patterns)
random.shuffle(pbaike_paths)

pshici_patterns = [
    "data/pretrain_data/part-pshici-*.jsonl.zst",
]
pshici_paths = create_shard_kwargs(pshici_patterns)
random.shuffle(pshici_paths)

plyrics_patterns = [
    "data/pretrain_data/part-plyrics-*.jsonl.zst",
]
plyrics_paths = create_shard_kwargs(plyrics_patterns)
random.shuffle(plyrics_paths)


pcouplets_patterns = [
    "data/pretrain_data/part-pcouplets-*.jsonl.zst",
]
pcouplets_paths = create_shard_kwargs(pcouplets_patterns)
random.shuffle(pcouplets_paths)


paths = pile_paths[:250] + pbaike_paths[:30] + pnews_paths[:30] + wudao_paths[:10] \
        + pcouplets_paths + plyrics_paths[:10] + pshici_paths

transform_dict = {
    "wudao": lambda line: line["title"] + "\n" + line["content"],
    "pile": lambda line: line["text"],
    "pnews": lambda line: line["title"] + "\n" + line["text"],
    "pbaike":lambda line: line["title"] + "\n" + line["main_content"],
    "pshici": lambda line: line["title"] + "\n" + line["author"] +"\n" + line["text"],
    "plyrics": lambda line: line["title"] + "\n" + line["singer"] +"\n" + line["text"],
    "pcouplets": lambda line: line["text"],
}
data_iter = iter(DataIter(paths, transform_dict))

import io
import sentencepiece as spm

# Loads model from URL as iterator and stores the model to BytesIO.
model = io.BytesIO()
spm.SentencePieceTrainer.train(
    sentence_iterator=data_iter,
    model_writer=model,
    shuffle_input_sentence=False,
    train_extremely_large_corpus=True,
    # hyperparameters of tokenizer
    max_sentence_length=16384,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    model_type="BPE",
    vocab_size=100000,
    num_threads=10,
    # split digits and fallback to byte same as Llama.
    # set split_by_unicode_script to True to avoid grouping punctuation and characters together.
    split_digits=True,
    split_by_unicode_script=True,
    byte_fallback=True,
    # reserve whitespace and \n and \t etc. for code generation
    allow_whitespace_only_pieces=True,
    remove_extra_whitespaces=False,
    normalization_rule_name="nfkc",
)

# Serialize the model as file.
with open("configs/10w_vocab_guyu.model", "wb") as f:
    f.write(model.getvalue())

# Directly load the model from serialized model.
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
print(sp.decode(sp.encode("只因你太美🤗▃     \n  1")))
