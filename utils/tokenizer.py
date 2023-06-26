#!/usr/bin/env python
import re, toml
from time import perf_counter
from transformers import BertTokenizer

config_dict = toml.load("config.toml")

class Tokenizer:
    def __init__(self, ):
        self.tokenizer = BertTokenizer.from_pretrained(config_dict['model']['BERT'], do_lower_case=False)
    def tokenize(self, sequences):
        _time_start = perf_counter()
        print("Tokenizing...",end = '\r')
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        sequences = self.tokenizer(sequences, max_length = config_dict['dataset']['MAX_SEQ_LEN'],
                          padding = "max_length", truncation = True, return_tensors = 'pt',)
        print(f"Tokenization took {perf_counter() - _time_start:.2f}s")
        return sequences['input_ids'], sequences['attention_mask']
