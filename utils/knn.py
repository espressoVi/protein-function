import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import toml

config_dict = toml.load("config.toml")

class KNN:
    def __init__(self, train_dataset, K = 7):
        self.train_dataset = train_dataset
        self.K = K
        self.cpu = 24
        assert len(self.train_dataset) >= self.K
    def _predict(self, queries):
        res = {}
        for name, emb in tqdm(queries):
            dist = np.sqrt(np.sum(np.square(self.train_dataset.embeddings - emb), axis = -1))
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

class FeatureDataset:
    def __init__(self, names, embeddings, labels = None):
        self.names = names
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.names)
    def __iter__(self):
        self.idx = 0
        return self
    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        self.idx+=1
        return self.names[self.idx-1], self.embeddings[self.idx-1]
    def split(self, count):
        res = []
        for i in range(1,count+1):
            start,end = int((i-1)*(len(self)/count)), int(min(len(self),(i)*(len(self)/count)))
            res.append(FeatureDataset(self.names[start:end], self.embeddings[start:end]))
        return res
