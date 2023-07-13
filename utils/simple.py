#!/usr/bin/env python3
import os,re,toml,torch
import numpy as np
from tqdm import tqdm
import pickle
from torch.utils.data import TensorDataset
from utils.IterativeStratification import IterativeStratification

config_dict = toml.load("config.toml")
files = config_dict['files']

class Preprocess:
    def __init__(self):
        self.class_number = 500
        self.labels = self.load_labels()
        self.go_terms = self._get_terms()
        self.labels = [(name, label) for name, label in self.labels if label in self.go_terms]
        self.idx2go,self.go2idx = self._idx_to_and_from_go()
        self.train_labels = self._get_labels()
    def get_train_dataset(self):
        embeddings = self._get_embeddings(mode = 'train')
        train_protein_names = list(self.train_labels.keys())
        train_labels = np.array([self.train_labels[name] for name in train_protein_names])
        train_embeddings = torch.tensor(np.array([embeddings[name] for name in train_protein_names]))
        return train_protein_names, train_embeddings, train_labels
    def get_test_dataset(self):
        embeddings = self._get_embeddings(mode = 'test')
        test_protein_names = list(embeddings.keys())
        test_embeddings = torch.tensor(np.array([embeddings[name] for name in test_protein_names]))
        return test_protein_names, test_embeddings
    def _get_embeddings(self, mode = 'train'):
        file = config_dict['files']['EMBEDS'] if mode == 'train' else config_dict['files']['EMBEDS_TEST']
        with open(file, 'rb') as f:
            emb = pickle.load(f)
        return emb
    def _get_labels(self):
        train_labels = {}
        for name, label in self.labels:
            if name in train_labels:
                train_labels[name][self.go2idx[label]] = 1
            else:
                train_labels[name] = np.zeros(self.class_number, dtype=bool)
                train_labels[name][self.go2idx[label]] = 1
        return train_labels
    def _get_terms(self):
        _count = {}
        for protein, go in tqdm(self.labels, desc = "Counting proteins"):
            if go in _count:
                _count[go].add(protein)
            else:
                _count[go] = {protein}
        _count = {key:len(value) for key,value in _count.items()}
        _count = sorted(_count.items(), key=lambda x:x[1], reverse = True)
        go = set([i for i,_ in _count][:self.class_number])
        return go
    def _idx_to_and_from_go(self):
        idx2go,go2idx = {},{}
        for idx,node in enumerate(self.go_terms):
            idx2go[idx] = node
            go2idx[node] = idx
        return idx2go,go2idx
    @staticmethod
    def load_labels():
        """ Reads label file """
        with open(config_dict['files']['TRAIN_LAB'],'r') as f:
            labels = f.readlines()[1:]
        rv = []
        for lab in tqdm(labels, desc = "Reading labels"):
            row = lab.strip().split('\t')[:2]
            rv.append((row[0],int(row[1].split(':')[-1])))
        return rv

class Dataset:
    def __init__(self, subgraph = None, finetune = False):
        self.subgraph = subgraph
        self.dataset = Preprocess()
        self.class_number = self.dataset.class_number
    def get_train_dataset(self):
        protein_names, embeddings, labels = self.dataset.get_train_dataset()
        split = self.get_folds(protein_names, labels)
        train_embeddings, val_embeddings = self._train_val_split(embeddings, split)
        train_labels, val_labels = self._train_val_split(labels, split)
        #self.freqs = train_labels.sum(axis=0)/train_labels.shape[0]
        #train_labels = np.where(train_labels>self.freqs, train_labels, self.freqs)
        train_dataset = TensorDataset(train_embeddings, torch.tensor(train_labels))
        val_dataset = TensorDataset(val_embeddings, torch.tensor(val_labels))
        return train_dataset, val_dataset
    def get_test_dataset(self):
        protein_names, embeddings = self.dataset.get_test_dataset()
        test_dataset = TensorDataset(embeddings)
        return test_dataset, protein_names 
    def get_folds(self, protein_names, labels):
        fold_array = np.where(IterativeStratification(labels, n_splits=config_dict['dataset']['N_FOLDS']) == 0, True, False)
        return fold_array
    @staticmethod
    def _train_val_split(arr, idx):
        if isinstance(arr, list):
            val_arr = [arr[i] for i,j in enumerate(idx) if j]
            train_arr = [arr[i] for i,j in enumerate(idx) if not j]
        else:
            idx = torch.tensor(idx, dtype = torch.bool)
            val_arr = arr[idx]
            train_arr = arr[torch.logical_not(idx)]
        return train_arr, val_arr
    
if __name__ == "__main__":
    main()
