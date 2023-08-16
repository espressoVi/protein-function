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
    def __init__(self, train_dataset, K = 7):
        self.train_dataset = train_dataset
        self.norm = np.linalg.norm(train_dataset.embeddings, axis = -1)
        self.K = K
        self.cpu = 24
    def _predict(self, queries):
        res = {}
        for name, emb in tqdm(queries):
            dist = np.sum(np.square(self.train_dataset.embeddings - emb), axis = -1)
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
