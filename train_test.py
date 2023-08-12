#!/usr/bin/env python
import toml,re, pickle, os
import numpy as np
from tqdm import tqdm
from utils.metric import Metrics
from collections import Counter
from itertools import repeat
import multiprocessing as mp

config_dict = toml.load("config.toml")

class KNN:
    def __init__(self, train_dataset, K = 5):
        self.train_dataset = train_dataset
        self.norm = np.linalg.norm(train_dataset.embeddings, axis = -1)
        self.K = K
        self.cpu = 16
        assert len(self.train_dataset) >= self.K
    def _predict(self, queries):
        res = {}
        for name, emb in tqdm(queries):
            dist = np.dot(self.train_dataset.embeddings, emb)/self.norm
            topk = np.argsort(dist)[:self.K]
            prediction = np.mean(self.train_dataset.labels[topk], axis = 0)
            res[name] = prediction
        return res
    def predict(self, queries):
        parts = queries.split(self.cpu)
        res = {}
        with mp.Pool(processes = self.cpu) as pool:
            for preds in pool.map(self._predict, parts):
                res.update(preds)
        return res

def evaluate(model, val_dataset):
    eval_dataloader = DataLoader(val_dataset, batch_size = train_param['TEST_BATCH_SIZE'], shuffle = False, )
    predictions, labels = [],[]
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        input = {"edges":batch[1].to(device), "embeddings":batch[2].to(device),
                 "parent_mask":batch[3].to(device)}
        with torch.no_grad():
            preds = model(**input).detach().cpu().numpy()
            child = batch[-2].detach().cpu().numpy()
            #labs = batch[-1].detach().cpu().numpy()
        predictions.extend(preds)
        labels.extend(child)
    predictions, labels = np.array(predictions), np.array(labels)
    return predictions, labels

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
