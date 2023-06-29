#!/usr/bin/env python
import torch,toml,re
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR as Scheduler
from tqdm import trange,tqdm

config_dict = toml.load("config.toml")
train_param = config_dict['train']

def train(model, device, dataset, metrics):
    train_dataset, val_dataset = dataset.get_train_dataset()
    infer = dataset.fill
    """ train_dataset, val_dataset are torch TensorDatasets
    which contains (input_ids, attention_masks, labels). """
    train_dataloader = DataLoader(train_dataset, batch_size = train_param['MINI_BATCH_SIZE'], shuffle = True, )
    optimizer_parameters = model.parameters()
    optimizer = AdamW(optimizer_parameters, lr = train_param['LR'], eps = 1e-8, weight_decay = 1e-4)
    epochs = train_param['EPOCHS']
    scheduler = Scheduler(optimizer, epochs)
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
            evaluate(model, device, val_dataset, infer, metrics)

def evaluate(model, device, val_dataset, infer, metrics):
    eval_dataloader = DataLoader(val_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False, )
    metric = metrics.metrics['CAFAMetric']
    preds, labels = [],[]
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = model(*batch)
        pred_batch = pred.detach().cpu().numpy()
        label_batch = batch[-1].detach().cpu().numpy()
        preds.extend(pred_batch)
        labels.extend(label_batch)
    preds,labels = np.array(preds),np.array(labels)
    threshold = find_threshold(metric, labels, preds, infer)
    outputs = infer((preds>threshold).astype(int))
    print(metrics.eval_and_show(labels, outputs))

def find_threshold(metric, labels, preds, infer):
    best, threshold = 0, 0
    for i in tqdm(np.linspace(0.1, min(0.99,np.amax(preds)),100), desc="Tuning threshold"):
        outputs = (preds>i).astype(bool)
        score = metric(labels, infer(outputs))
        if not np.isnan(score) and score > best:
            best = score 
            threshold = i
    return threshold

def write_predictions(model, device, dataset):
    pass
