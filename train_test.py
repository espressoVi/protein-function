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
    train_dataset, val_dataset = dataset.get()
    epochs = train_param['EPOCHS']
    optimizer = AdamW(model.parameters(), lr = train_param['LR'], eps = 1e-8, weight_decay = 1e-4)
    scheduler = Scheduler(optimizer, factor = 0.5, patience = 1)
    train_dataloader = DataLoader(train_dataset, batch_size = train_param['MINI_BATCH_SIZE'], shuffle = True, )
    counter, train_loss, loss = 1, 0.0,0.0
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
        res, f1, threshold = validate(model, val_dataset)
        print(res)
        scheduler.step(f1)
    torch.save(model.state_dict(), save_path)

def validate(model, val_dataset):
    outputs, labels = evaluate(model, val_dataset)
    metrics = Metrics()
    score = metrics.metrics['Micro F1']
    outputs = val_dataset.fill(outputs)
    labels = val_dataset.fill(labels)
    #best, threshold = 0, 0
    #for i in tqdm(np.linspace(-1, 1,200), desc="Tuning threshold"):
    #    outputs = (sims>i).astype(bool)
    #    f1 = score(labels, outputs)
    #    if not np.isnan(f1) and f1 > best:
    #        best, threshold = f1, i
    print("".join([f"Pred {np.where(outputs[i]==1)[0]}, {np.where(labels[i]==1)[0]}\n" for i in range(10)]))
    res = metrics.eval_and_show(labels, outputs)
    f1 = score(labels, outputs)
    return res, f1, 0

def evaluate(model, val_dataset):
    eval_dataloader = DataLoader(val_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False, )
    sims, labels = [],[]
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        input = {"edges":batch[1].to(device), "embeddings":batch[2].to(device),}
        with torch.no_grad():
            preds = model(**input).detach().cpu().numpy()
            labs = batch[-1].detach().cpu().numpy()
        sims.extend(preds)
        labels.extend(labs)
    sims, labels = np.array(sims), np.array(labels).astype(bool)
    preds = np.zeros_like(labels)
    for i,sim in enumerate(sims):
        preds[i,sim] = 1
    #sims = val_dataset.fill(sims)
    return preds, labels

def write_predictions(trained_model, threshold, dataset):
    trained_model.eval()
    test_dataset, names = dataset.get()
    eval_dataloader = DataLoader(test_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = True,)
    preds = []
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = trained_model(*batch, labels = None )
        pred_batch = pred.detach().cpu().numpy()
        preds.extend(pred_batch)
    rv = []
    for name, pred in zip(names, preds):
        for idx in np.where(pred > threshold)[0]:
            go_id = f"GO:{dataset.dataset.idx2go[idx]:07d}"
            val = pred[idx]
            rv.append(f"{name}\t{go_id}\t{val:.4f}\n")
    with open(os.path.join(config_dict['files']['SUBMIT'],f"submit.tsv"),'a') as f:
        f.writelines("".join(rv))
