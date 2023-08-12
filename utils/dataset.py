#!/usr/bin/env python3
import os,re,toml,torch
import networkx as nx
import numpy as np
import pickle
from utils.GO import GeneOntology 
from collections import Counter


config_dict = toml.load("config.toml")
files = config_dict['files']

class GetDataset:
    """ Prepares two datasets queries and training samples for kNN """
    def __init__(self, subgraph):
        self.subgraph = subgraph
        self.GO = GeneOntology(subgraph)
        self.idx2go, self.go2idx = self._idx_to_and_from_go()
        self.nodes = set([self.go2idx[node] for node in self.GO.Graph])
    def get(self):
        labels = self._get_labels()
        train_names = list(labels.keys())
        train_labels = np.array([labels[name] for name in train_names])
        train_embeddings = self._get_embeddings(mode = 'train')
        train_embeddings = np.array([train_embeddings[name] for name in train_names])

        test_embeddings = self._get_embeddings(mode = "test")
        test_names = list(set(test_embeddings.keys()) - set(train_names))
        test_embeddings = np.array([test_embeddings[name] for name in test_names])

        train_dataset = FeatureDataset(train_names, train_embeddings, train_labels)
        test_dataset = FeatureDataset(test_names, test_embeddings)
        return train_dataset, test_dataset
    def _get_embeddings(self, mode = 'train'):
        file = config_dict['files']['EMBEDS'] if mode == 'train' else config_dict['files']['EMBEDS_TEST']
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

class FeatureDataset:
    def __init__(self, names, embeddings, labels = None):
        self.names = names
        self.embeddings = embeddings
        self.labels = labels
        self.train = True if self.labels is not None else False
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
