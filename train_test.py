#!/usr/bin/env python
import torch,toml,re
import pickle,os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau as Scheduler
from tqdm import trange,tqdm

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
    sims, labels = evaluate(model, val_dataset)
    best, threshold = 0, 0
    for i in tqdm(np.linspace(-1, 1,200), desc="Tuning threshold"):
        pr, rc, f1 = check(sims, labels, i)
        if not np.isnan(f1) and f1 > best:
            best, threshold = f1, i
    pr, rc, f1 = check(sims, labels, threshold)
    res = f"Threshold : {threshold:.4f}, precision: {pr:.4f}, recall: {rc:.4f}, f1:{f1:.4f}"
    return res, f1, threshold

def evaluate(model, val_dataset):
    eval_dataloader = DataLoader(val_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False, )
    sims, labels = [],[]
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        input = {"edges":batch[1].to(device), "embeddings":batch[2].to(device),
                     "probe_nodes":batch[3].to(device)}
        with torch.no_grad():
            sim = model(**input)
            sim = sim.detach().cpu().numpy()
            labs = batch[-1].detach().cpu().numpy()
        sims.extend(sim)
        labels.extend(labs)
    sims, labels = np.array(sims), np.array(labels)
    return sims, labels

def check(sims, labels, threshold):
    predicts = (sims > threshold).astype(bool)
    labels = (labels == 1).astype(bool)
    tp = np.logical_and(predicts, labels).sum()
    fp = np.logical_and(predicts, np.logical_not(labels)).sum()
    fn = np.logical_and(np.logical_not(predicts), labels).sum()
    tn = np.logical_and(np.logical_not(predicts), np.logical_not(labels)).sum()
    pr = tp/(tp+fp) if (tp+fp) else 0
    rc = tp/(tp+fn)
    f1 = 2*pr*rc/(pr+rc)
    return pr, rc, f1

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
