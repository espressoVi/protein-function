#!/usr/bin/env python
import torch
from transformers import AutoTokenizer
import toml, re, pickle

config_dict = toml.load("config.toml")

def read_sequences(filename):
    """ Reads fasta file and loads sequences. """
    with open(filename,'r') as f:
        raw = f.readlines()
    idx = [i for i,s in enumerate(raw) if '>' in s]
    idx.append(None)
    sequences = {}
    for start,end in zip(idx[:-1],idx[1:]):
        name = raw[start].split()[0][1:]
        sequence = ''.join([i.strip() for i in raw[start+1:end]])
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        sequences[name] = sequence
    return sequences

def main(train = True):
    filename = config_dict['files']['TRAIN_SEQ'] if train else config_dict['files']['TEST_SEQ']
    sequences = read_sequences(filename)
    names = list(sequences.keys())
    proteins = [sequences[name] for name in names]
    pretrained_model = config_dict['model']['PRETRAINED']
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenized = tokenizer(proteins, max_length = config_dict['dataset']['MAX_SEQ_LEN'], padding = "max_length", truncation = True)
    input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
    res = {}
    for name, input_id, atten in zip(names, input_ids, attention_mask):
        res[name] = {'input_ids':input_id, 'attention_mask':atten}
    with open(f"tokenized_{'train' if train else 'test'}.pkl", "wb") as f:
        pickle.dump(res, f)

if __name__ == "__main__":
    main(train=True)
    main(train=False)
