#!/usr/bin/env python
import torch,toml,re
import pickle,os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR as Scheduler
from tqdm import trange,tqdm

config_dict = toml.load("config.toml")
train_param = config_dict['train']
device = torch.device("cuda")

def train(model, save_path, train_dataset, validate):
    epochs = train_param['EPOCHS']
    optimizer = AdamW(model.parameters(), lr = train_param['LR'], eps = 1e-8, weight_decay = 1e-4)
    scheduler = Scheduler(optimizer, epochs)
    train_dataloader = DataLoader(train_dataset, batch_size = train_param['MINI_BATCH_SIZE'], shuffle = True, )
    counter, train_loss, loss = 1, 0.0,0.0
    model.zero_grad()
    for epoch_number in range(int(epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            loss, predicts = model(*batch)
            train_loss += loss.item()
            loss.backward()
            if (i+1)%train_param['ACCUMULATE'] == 0:
                optimizer.step()
                model.zero_grad()
            counter += 1
            epoch_iterator.set_description(f"Epoch [{epoch_number+1}/{epochs}] Loss : {train_loss/counter:.6f}")
            epoch_iterator.refresh()
        scheduler.step()
        if (epoch_number+1)%10 == 0:
            print(validate(model))
    torch.save(model.state_dict(), save_path)

def validator(val_dataset, infer_parents, metrics):
    def validate(model):
        labels, predictions = evaluate(model, val_dataset, infer_parents)
        best, threshold = 0, 0
        for i in tqdm(np.linspace(0.1, 1,150), desc="Tuning threshold"):
            outputs = (predictions>i).astype(bool)
            score = metrics.metrics['F1'](labels, outputs)
            if not np.isnan(score) and score > best:
                best = score 
                threshold = i
        outputs = (predictions>threshold).astype(bool)
        return metrics.eval_and_show(labels, outputs)
    return validate

def evaluate(model, val_dataset, infer_parents):
    eval_dataloader = DataLoader(val_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False, )
    preds, labels = [],[]
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = model(*batch)
        pred_batch = pred.detach().cpu().numpy()
        label_batch = batch[-1].detach().cpu().numpy()
        preds.extend(infer_parents(pred_batch))
        labels.extend(label_batch)
    predictions,labels = np.array(preds),np.array(labels)
    return labels, predictions

def write_predictions(trained_model, threshold, dataset):
    trained_model.eval()
    test_dataset, names = dataset.get_test_dataset()
    """ test_dataset contains (input_ids, attention_masks) names 
    contains name of proteins corresponding to input_ids. """
    eval_dataloader = DataLoader(test_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False,)
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
            rv.append(f"{name}\t{go_id}\t{val:.4f}")
    with open(os.path.join(config_dict['files']['SUBMIT'],f"{dataset.subgraph}.tsv"),'w') as f:
        f.writelines("\n".join(rv))

def write_top(trained_model, dataset):
    trained_model.eval()
    test_dataset, names = dataset.get_test_dataset()
    freqs = dataset.freqs
    """ test_dataset contains (input_ids, attention_masks) names 
    contains name of proteins corresponding to input_ids. """
    eval_dataloader = DataLoader(test_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False,)
    preds = []
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = trained_model(*batch, labels = None )
        pred_batch = pred.detach().cpu().numpy()
        preds.extend(pred_batch)
    rv = []
    for name, pred in zip(names, preds):
        for idx in np.where(pred>freqs)[0]:
            go_id = f"GO:{dataset.dataset.idx2go[idx]:07d}"
            val = pred[idx]
            rv.append(f"{name}\t{go_id}\t{val:.3f}")
    with open(os.path.join(config_dict['files']['SUBMIT'],f"{dataset.subgraph}.tsv"),'w') as f:
        f.writelines("\n".join(rv))
