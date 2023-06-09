#!/usr/bin/env python
import torch,toml
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW
from tqdm import trange,tqdm

config_dict = toml.load("config.toml")
train_param = config_dict['train']

def train(model, device, train_dataset, val_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size = train_param['BATCH_SIZE'], shuffle = True, )
    optimizer_parameters = model.parameters()
    optimizer = AdamW(optimizer_parameters, lr = train_param['LR'], eps = 1e-8, weight_decay = 1e-4)
    epochs = train_param['EPOCHS']
    counter, train_loss, loss = 1, 0.0,0.0
    model.zero_grad()
    iterator = trange(int(epochs), desc="Epoch")
    for _ in iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = { "input_ids": batch[0],
                      "attention_masks": batch[1],
                      "labels":batch[2],}
            loss, predicts = model(**inputs)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            #scheduler.step()  
            model.zero_grad()
            counter += 1
            epoch_iterator.set_description("Loss :%f" % (train_loss/counter))
            epoch_iterator.refresh()

def evaluate(model, device, dataset):
    pass

def write_predictions(model, device, dataset):
    pass
