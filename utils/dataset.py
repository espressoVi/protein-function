#!/usr/bin/env python3
import os,re,toml,torch, pickle
import networkx as nx
import numpy as np
from tqdm import tqdm
from utils.GO import GeneOntology 
from torch.utils.data import Dataset


config_dict = toml.load("config.toml")
files = config_dict['files']

class PairDataset(Dataset):
    def __init__(self, X, queries, labels = None, keys = None, similarity = None):
        self.X = X #Dict of all names:features
        self.queries = queries #List of names which are queried
        self.train=labels is not None and keys is not None and similarity is not None
        self.labels = labels #Dict of name:labels
        self.keys = keys #List of key names
        self.similarity = similarity #Similarity
        self.train = True if self.labels is not None else False
    def __len__(self):
        return len(self.queries)
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"{idx} is greater than dataset size {len(self)}")
        if self.train:
            query = self.queries[idx]
            key = self.keys[idx]
            query_feature = torch.tensor(self.X[query], dtype = torch.float)
            key_feature = torch.tensor(self.X[key], dtype = torch.float)
            sim = torch.tensor(self.similarity[idx], dtype = torch.float)
            query_label = torch.tensor(self.labels[query], dtype = bool)
            key_label = torch.tensor(self.labels[key], dtype = bool)
            return query, query_feature, key_feature, sim, query_label, key_label
        else:
            query = self.queries[idx]
            query_feature = torch.tensor(self.X[query], dtype = torch.float)
            return query, query_feature

class GetDataset:
    def __init__(self, subgraph, train=True):
        self.subgraph, self.train = subgraph, train
        self.GO = GeneOntology(subgraph)
        self.idx2go, self.go2idx = self._idx_to_and_from_go()
        self.nodes = set([self.go2idx[node] for node in self.GO.Graph])
        self.weights = self._init_IA()
    def get_train_val(self):
        self.train_labels = self._get_labels()
        self.X = self._get_embeddings(mode = "train")
        queries, keys, sim = self._get_train_pairs()
        idx = np.random.binomial(1, config_dict['train']['TRAIN_SIZE'], size = len(queries))
        train_queries, val_queries = self._train_val_split(queries, idx)
        train_keys, val_keys = self._train_val_split(keys, idx)
        train_sim, val_sim = self._train_val_split(sim, idx)
        train_dataset = PairDataset(self.X, train_queries, self.train_labels, train_keys, train_sim)
        val_dataset = PairDataset(self.X, val_queries, self.train_labels, val_keys, val_sim)
        return train_dataset, val_dataset
    def _get_train_pairs(self):
        low, high = config_dict['dataset']['CUTOFF_LOW'], config_dict['dataset']['CUTOFF_HIGH'] 
        queries, keys, sim = [],[], []
        with open(f"{files['TRAIN_PAIRS']}{self.subgraph}.tsv", "r") as f:
            for i in f.readlines():
                q,k,s = i.rstrip().split("\t")
                s = float(s)
                if low < s < high:
                    continue
                if s > high:
                    s = 1.0
                elif s < low:
                    s = 0.0
                queries.append(q)
                keys.append(k)
                sim.append(s)
        return queries, keys, sim
    def _compute_cafa(self, target, label):
        intersection = np.sum(self.weights*np.logical_and(target, label))
        pred = np.sum(self.weights*label)
        gt = np.sum(self.weights*target)
        pr = intersection/pred if pred!=0 else 0
        rc = intersection/gt if gt!=0 else 0
        return (2*pr*rc)/(pr+rc) if (pr+rc) !=0 else 0
    def _get_embeddings(self, mode = 'train'):
        file = files['EMBEDS'] if mode == 'train' else files['EMBEDS_TEST']
        with open(file, 'rb') as f:
            emb = pickle.load(f)
        return emb
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
    @staticmethod
    def _train_val_split(arr, idx):
        assert isinstance(arr, list)
        train_arr = [arr[i] for i,j in enumerate(idx) if j]
        val_arr = [arr[i] for i,j in enumerate(idx) if not j]
        return train_arr, val_arr
