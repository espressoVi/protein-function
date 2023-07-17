#!/usr/bin/env python
import os
import torch,toml,re
import numpy as np
from utils.tokenizer import Tokenizer
from transformers import BertTokenizer
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, TensorDataset

config_dict = toml.load("config.toml")

def save_embeddings(model, device, test = False):
    embed_file = config_dict['files']['EMBEDS'] if not test else config_dict['files']['EMBEDS_TEST']
    file = config_dict['files']['TRAIN_SEQ'] if not test else config_dict['files']['TEST_SEQ']
    if os.path.exists(embed_file):
        raise ValueError("Embeddings already exist")
    sequences = {}
    with open(file,'r') as f:
        raw = f.readlines()
    idx = [i for i,s in enumerate(raw) if '>' in s]
    idx.append(None)
    for start,end in zip(idx[:-1],idx[1:]):
        name = raw[start].split()[0][1:]
        sequence = ''.join([i.strip() for i in raw[start+1:end]])
        sequences[name] = sequence
    names = sequences.keys()
    proteins = [sequences[name] for name in names]
    input_ids, attention_masks = Tokenizer().tokenize(proteins)
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size = config_dict['train']['TEST_BATCH_SIZE'], shuffle = False, )
    model.eval()
    embeds = []
    for batch in tqdm(dataloader, desc = "Generating embeddings"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = model(batch[0],batch[1])
        pred = pred.detach().cpu().numpy()
        embeds.extend(pred)
    result = {name:embed for name,embed in zip(names, embeds)}
    with open(embed_file, 'wb') as f:
        pickle.dump(result, f)
