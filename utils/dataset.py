#!/usr/bin/env python3
import os,re,toml,torch
import networkx as nx
import numpy as np
import pickle
from utils.GO import GeneOntology 
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix

config_dict = toml.load("config.toml")
files = config_dict['files']

class GetDataset:
    def __init__(self, subgraph, train=True):
        self.subgraph, self.train = subgraph, train
        self.GO = GeneOntology(subgraph)
        self.idx2go, self.go2idx = self._idx_to_and_from_go()
        self.nodes = set([self.go2idx[node] for node in self.GO.Graph])
        self.edges = self.get_edge_index()
    def get(self):
        if not self.train:
            names, embeddings = zip(*self._get_embeddings(mode = 'test').items())
            test_dataset = GraphDataset(self.edges, names, embeddings, None, None)
            return test_dataset
        embeddings = self._get_embeddings(mode = 'train')
        labels = self._get_labels()
        names = list(labels.keys())
        labels = [labels[name] for name in names]
        embeddings = [embeddings[name] for name in names]
        assert len(names) == len(embeddings) == len(labels)
        idx = np.random.binomial(1, config_dict['train']['TRAIN_SIZE'], size = len(names))
        train_names, val_names = self._train_val_split(names, idx)
        train_labels, val_labels = self._train_val_split(labels, idx)
        train_embeddings, val_embeddings = self._train_val_split(embeddings, idx)
        train_dataset = GraphDataset(self.edges, train_names, train_embeddings, train_labels)
        val_dataset = GraphDataset(self.edges, val_names, val_embeddings, val_labels)
        return train_dataset, val_dataset
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
        for name in positive_labels.keys():
            lab = np.zeros(len(self.nodes), dtype = bool)
            lab[list(positive_labels[name])] = 1
            labels[name] = lab
        return labels
    def _idx_to_and_from_go(self):
        idx2go,go2idx = {},{}
        for idx,node in enumerate(nx.topological_sort(self.GO.Graph)):
            idx2go[idx] = node
            go2idx[node] = idx
        return idx2go,go2idx
    @staticmethod
    def _train_val_split(arr, idx):
        assert isinstance(arr, list)
        train_arr = [arr[i] for i,j in enumerate(idx) if j]
        val_arr = [arr[i] for i,j in enumerate(idx) if not j]
        return train_arr, val_arr
    def get_edge_index(self):
        adj = 2*np.identity(len(self.nodes))
        for source, dest in self.GO.Graph.edges():
            adj[self.go2idx[source], self.go2idx[dest]] = 1
        adj = torch.tensor(adj, dtype = torch.float)
        return adj

class GraphDataset(Dataset):
    def __init__(self, edges, names, embeddings, labels = None):
        self.names = names
        self.edges = edges
        self.embeddings = embeddings
        self.labels = labels
        self.train = True if self.labels is not None else False
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"{idx} is greater than dataset size {len(self)}")
        name = self.names[idx]
        embeds = torch.tensor(self.embeddings[idx], dtype=torch.float)
        if not self.train:
            return name, self.edges, embeds, None
        label = self.labels[idx]
        probs = 9*label/np.sum(label) + (1-label)/np.sum(1-label)
        probs = probs/np.sum(probs)
        probe_node = np.random.choice(self.edges.shape[0], p = probs)
        probe_node_t = torch.tensor(probe_node, dtype = torch.long)
        label = torch.tensor(label[probe_node], dtype = torch.float)
        return name, self.edges, embeds, probe_node_t, label
    
def main():
    train_dataset, val_dataset = GetDataset(subgraph="CC", train = True).get()
    dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    for batch in dataloader:
        print(batch)

if __name__ == "__main__":
    main()
