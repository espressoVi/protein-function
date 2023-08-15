import os,re,toml,torch
import networkx as nx
import numpy as np
from GO import GeneOntology 
from tqdm import tqdm

config_dict = toml.load("config.toml")
files = config_dict['files']

class GetDataset:
    def __init__(self, subgraph, train=True):
        self.subgraph, self.train = subgraph, train
        self.GO = GeneOntology(subgraph)
        self.idx2go, self.go2idx = self._idx_to_and_from_go()
        self.nodes = set([self.go2idx[node] for node in self.GO.Graph])
        self.weights = self._init_IA()
        self.train_labels = self._get_labels()
    def get_train_dataset(self):
        res = {}
        diamond = self._load_diamond()
        for name, label in tqdm(self.train_labels.items()):
            if name in diamond:
                database = diamond[name].union(set(np.random.choice(list(self.train_labels.keys()), size = 10, replace = False)))
            else:
                database = set(np.random.choice(list(self.train_labels.keys()), size = 10, replace = False))
            matches = self.compute_similarity(name, database)
            for key, value in matches.items():
                final_key = f"{min(name, key)}\t{max(name, key)}\t"
                res[final_key] = value
        res = "\n".join([f"{key}{value}" for key, value in res.items()])
        with open(f"train_{self.subgraph}.tsv","w") as f:
            f.writelines(res)
    def compute_similarity(self, query_name, database):
        res = {}
        for name in database:
            if name == query_name or name not in self.train_labels:
                continue
            res[name] = self._compute_cafa(self.train_labels[name], self.train_labels[query_name])
        return res
    def _load_diamond(self):
        with open(config_dict['files']["DIAMOND"],"r") as f:
            pairs = [tuple(i.split("\t")[:2]) for i in f.readlines()]
        res = {}
        for a,b in pairs:
            if a in res:
                res[a].add(b)
            else:
                res[a] = {b}
        return res
    def _compute_cafa(self, target, label):
        intersection = np.sum(self.weights*np.logical_and(target, label))
        pred = np.sum(self.weights*label)
        gt = np.sum(self.weights*target)
        pr = intersection/pred if pred!=0 else 0
        rc = intersection/gt if gt!=0 else 0
        return (2*pr*rc)/(pr+rc) if (pr+rc) !=0 else 0
    def _get_labels(self):
        _labels = [(i,self.go2idx[j]) for i,j in self.GO.labels if j in self.GO.Graph]
        positive_labels = {}
        for name, node in _labels:
            if name in positive_labels:
                positive_labels[name].add(node)
            else:
                positive_labels[name] = {node}
        labels = {}
        for name, label in positive_labels.items():
            lab = np.zeros(len(self.nodes), dtype = bool)
            lab[list(label)] = True
            labels[name] = lab
        return labels
    def _idx_to_and_from_go(self):
        idx2go,go2idx = {},{}
        for idx,node in enumerate(nx.topological_sort(self.GO.Graph)):
            idx2go[idx] = node
            go2idx[node] = idx
        return idx2go,go2idx
    def _init_IA(self):
        ia = self.GO.weights
        weights = np.zeros(len(self.go2idx), dtype = float)
        for key, value in ia.items():
            weights[self.go2idx[key]] = value
        return weights

def main():
    GetDataset(subgraph="CC").get_train_dataset()
    GetDataset(subgraph="BP").get_train_dataset()
    GetDataset(subgraph="MF").get_train_dataset()

if __name__ == "__main__":
    main()
