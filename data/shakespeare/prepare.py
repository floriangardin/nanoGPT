import os
import requests
import tiktoken
import numpy as np
from transformers import GPT2TokenizerFast
# download the tiny shakespeare dataset

input_file_path = '/content/drive/MyDrive/hfdata/train.txt'
with open(input_file_path, 'r') as f:
    data = f.read().replace('\n', '')
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

tokenizer = GPT2TokenizerFast.from_pretrained("floriangardin/musiclang_optimized")

train_ids = tokenizer(train_data)['input_ids']
val_ids = tokenizer(val_data)['input_ids']
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
