#!/usr/bin/env python
import torch,toml,re
import pickle,os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau as Scheduler
from tqdm import trange,tqdm
from utils.metric import Metrics

config_dict = toml.load("config.toml")
train_param = config_dict['train']
device = torch.device("cuda")

def train(model, save_path, dataset,):
    train_dataset, val_dataset = dataset.get_train_val()
    epochs = train_param['EPOCHS']
    optimizer = AdamW(model.parameters(), lr = train_param['LR'], eps = 1e-8, weight_decay = 1e-5)
    scheduler = Scheduler(optimizer, factor = 0.1, patience = 1)
    train_dataloader = DataLoader(train_dataset, batch_size = train_param['MINI_BATCH_SIZE'], shuffle = True, )
    counter, train_loss, loss, = 1, 0.0,0.0
    model.zero_grad()
    for epoch_number in range(int(epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch[1:])
            loss, predicts = model(*batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            counter += 1
            epoch_iterator.set_description(f"Epoch [{epoch_number+1}/{epochs}] Loss : {train_loss/counter:.6f}")
            epoch_iterator.refresh()
            if (i+1)%1000 == 0:
                torch.save(model.esm.state_dict(), save_path)
                print("Checkpoint saved")
        val_score = validate(model, val_dataset)
        scheduler.step(val_score)
        print(f"Val Metric : {val_score:.6f}")

def validate(model, val_dataset):
    distances, similarity = evaluate(model, val_dataset)
    mean_diff = np.abs(np.mean(distances*similarity) - np.mean(distances*(1-similarity)))
    return mean_diff

def evaluate(model, val_dataset):
    eval_dataloader = DataLoader(val_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False, )
    predictions, labels = [],[]
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        input = {"query_features":batch[1].to(device), "key_features":batch[2].to(device),
                 "sims":batch[3].to(device)}
        with torch.no_grad():
            preds, _ = model(**input)
            predictions.extend(preds.detach().cpu().numpy())
            labels.extend(batch[3].detach().cpu().numpy())
    predictions, labels = np.array(predictions), np.array(labels)
    return predictions, labels

def create_dataset(model, dataset, save_path):
    eval_dataloader = DataLoader(dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False, )
    predictions, names = [],[]
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        input = {"input_ids":batch[1].to(device), "attention_masks":batch[2].to(device)}
        with torch.no_grad():
            features = model(**input)
            predictions.extend(features.detach().cpu().numpy())
            names.extend(batch[0])
    res = {name:pred for name, pred in zip(names, predictions)}
    with open(save_path, "wb") as f:
        pickle.dump(res, f)
