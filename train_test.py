#!/usr/bin/env python
import torch,toml,re
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR as Scheduler
from tqdm import trange,tqdm
import pickle

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
            threshold = evaluate(model, device, val_dataset, infer, metrics)
    threshold = evaluate(model, device, val_dataset, infer, metrics)
    return model, threshold

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
    return threshold

def find_threshold(metric, labels, preds, infer):
    best, threshold = 0, 0
    for i in tqdm(np.linspace(0.1, min(0.99,np.amax(preds)),100), desc="Tuning threshold"):
        outputs = (preds>i).astype(bool)
        score = metric(labels, infer(outputs))
        if not np.isnan(score) and score > best:
            best = score 
            threshold = i
    return threshold

def write_predictions(trained_model, threshold, device, dataset, use_embeds = True):
    trained_model.eval()
    if use_embeds:
        eval_dataloader = load_pretrained_test()
        preds, names = [],[]
        for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
            name = batch[0]
            with torch.no_grad():
                pred = trained_model(batch[1].to(device), labels = None )
            pred_batch = pred.detach().cpu().numpy()
            preds.extend(pred_batch)
            names.extend(name)
    else:
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
    with open(config_dict['files']['SUBMIT'],'a') as f:
        f.writelines("\n".join(rv))

def load_pretrained_test():
    with open(config_dict['files']['EMBEDS_TEST'],'rb') as f:
        embeds = pickle.load(f)
    name_dataloader = DataLoader(list(embeds.keys()), batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False)
    for name_batch in name_dataloader:
        yield name_batch, torch.tensor(np.array([embeds[name] for name in name_batch]))
