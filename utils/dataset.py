#!/usr/bin/env python3
import os,re,toml,torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch_geometric.data import Data
#from utils.IterativeStratification import IterativeStratification
#from utils.tokenizer import Tokenizer
#from utils.GO import GeneOntology 
from GO import GeneOntology 
from torch.utils.data import TensorDataset

config_dict = toml.load("config.toml")
files = config_dict['files']

class Dataset:
    def __init__(self, subgraph):
        self.subgraph = subgraph
        self.GO = GeneOntology(subgraph)
        self.graph = Data(edge_index = self.GO.edge_index)
        _train_labels = self.GO.labels
        self.train_labels = [(i,j) for i,j in _train_labels if j in self.GO.Graph]
        print(self.train_labels)

def main():
    Dataset('CC')
if __name__ == "__main__":
    main()

#class Preprocess:
#    def __init__(self, subgraph):
#        self.GO = GeneOntology(subgraph)
#        self._train_labels = self.GO.labels
#        #self._train_labels = self.GO.load_labels()
#        self.label_graph = self.GO.Graph
#        self.class_number = len(self.label_graph)
#        self.idx2go,self.go2idx = self._idx_to_and_from_go()
#        self.tokenizer = Tokenizer()
#        self.IA = self._parse_ia()
#    def get_train_dataset(self, finetune = False):
#        """ Returns lists of protein sequences, their names and their labels (if applicable) """
#        train_protein_names,train_labels = self._get_protein_labels(self._train_labels)
#        if finetune:
#            p_dict = self._load_proteins(mode)  
#            train_proteins = [p_dict[protein_name] for protein_name in train_protein_names]
#            train_input_ids, train_attention_masks = self.tokenizer.tokenize(train_proteins)
#            return train_protein_names, train_input_ids, train_attention_masks, train_labels
#        embeddings = self._get_embeddings(mode = 'train')
#        train_embeddings = torch.tensor(np.array([embeddings[name] for name in train_protein_names]))
#        return train_protein_names, train_embeddings, train_labels
#    def get_test_dataset(self, finetune = False):
#        """ Returns lists of protein sequences, their names and their labels (if applicable) """
#        if finetune:
#            p_dict = self._load_proteins(mode)
#            test_protein_names, test_proteins = [list(i) for i in list(zip(*p_dict.items()))]
#            test_input_ids, test_attention_masks = self.tokenizer.tokenize(test_proteins)
#            return test_protein_names, test_input_ids, test_attention_masks
#        embeddings = self._get_embeddings(mode = 'test')
#        test_protein_names = list(embeddings.keys())
#        test_embeddings = torch.tensor(np.array([embeddings[name] for name in test_protein_names]))
#        return test_protein_names, test_embeddings
#    def _get_embeddings(self, mode = 'train'):
#        file = config_dict['files']['EMBEDS'] if mode == 'train' else config_dict['files']['EMBEDS_TEST']
#        with open(file, 'rb') as f:
#            emb = pickle.load(f)
#        return emb
#    def _idx_to_and_from_go(self):
#        idx2go,go2idx = {},{}
#        for idx,node in enumerate(nx.topological_sort(self.label_graph)):
#            idx2go[idx] = node
#            go2idx[node] = idx
#        return idx2go,go2idx
#    def _get_protein_labels(self, labels):
#        """ For each protein in the graph create its multi-label label after pruning."""
#        protein_labels = {}
#        for protein,go in tqdm(labels,desc='Processing labels'):
#            if go not in self.label_graph:
#                #protein_labels[protein] = set()             #Include negative examples also
#                continue
#            if protein in protein_labels:
#                protein_labels[protein].add(go)
#            else:
#                protein_labels[protein] = {go}
#            protein_labels[protein].update(self.GO.ancestors[go])
#        protein_names,labels = [],[]
#        for protein,label in protein_labels.items():
#            label_array = np.zeros(self.class_number,dtype = bool)
#            for node in label:
#                label_array[self.go2idx[node]] = 1
#            protein_names.append(protein)
#            labels.append(label_array)
#        return protein_names, np.array(labels)
#    def _parse_ia(self):
#        with open(config_dict['files']['IA'], 'r') as f:
#            raw = f.readlines()
#        weights = [i.strip().split() for i in raw]
#        weights = {int(i[0].split(':')[-1]):float(i[1]) for i in weights}
#        ia = np.zeros(len(self.go2idx), dtype=float)
#        for go,idx in self.go2idx.items():
#            ia[idx] = weights[go]
#        return ia
#    @staticmethod
#    def _load_proteins(mode = 'train'):
#        """ Reads fasta file and loads sequences. """
#        file = files['TRAIN_SEQ'] if mode == 'train' else files['TEST_SEQ']
#        sequences = {}
#        with open(file,'r') as f:
#            raw = f.readlines()
#        idx = [i for i,s in enumerate(raw) if '>' in s]
#        idx.append(None)
#        for start,end in zip(idx[:-1],idx[1:]):
#            name = raw[start].split()[0][1:]
#            sequence = ''.join([i.strip() for i in raw[start+1:end]])
#            sequences[name] = sequence
#        return sequences
#
#class Dataset:
#    def __init__(self, subgraph = None, finetune = False):
#        self.subgraph = subgraph
#        self.dataset = Preprocess(subgraph = subgraph)
#        self.IA = self.dataset.IA
#        self.class_number = self.dataset.class_number
#        self.finetune = finetune
#    def get_train_dataset(self):
#        if self.finetune:
#            protein_names, input_ids, attention_masks, labels = self.dataset.get_train_dataset(True)
#            split = self.get_folds(protein_names, labels)
#            train_input_ids, val_input_ids = self._train_val_split(input_ids,split)
#            train_attention_masks, val_attention_masks = self._train_val_split(attention_masks,split)
#            train_labels, val_labels = self._train_val_split(labels, split)
#            train_dataset = TensorDataset(train_input_ids, train_attention_masks, torch.tensor(train_labels))
#            val_dataset = TensorDataset(val_input_ids, val_attention_masks, torch.tensor(val_labels))
#        else:
#            protein_names, embeddings, labels = self.dataset.get_train_dataset(False)
#            split = self.get_folds(protein_names, labels)
#            train_embeddings, val_embeddings = self._train_val_split(embeddings, split)
#            train_labels, val_labels = self._train_val_split(labels, split)
#            train_dataset = TensorDataset(train_embeddings, torch.tensor(train_labels))
#            val_dataset = TensorDataset(val_embeddings, torch.tensor(val_labels))
#        return train_dataset, val_dataset
#    def get_test_dataset(self):
#        if self.finetune:
#            protein_names, input_ids, attention_masks = self.dataset.get_test_dataset(True)
#            test_dataset = TensorDataset(input_ids, attention_masks)
#        else:
#            protein_names, embeddings = self.dataset.get_test_dataset(False)
#            test_dataset = TensorDataset(embeddings)
#        return test_dataset, protein_names 
#    def fill(self, predictions):
#        rv = []
#        for pred in predictions:
#            prediction = np.zeros_like(pred, dtype = bool)
#            for idx in np.where(pred == 1)[0]:
#                anc = np.array([self.dataset.go2idx[i] for i in self.dataset.GO.ancestors[self.dataset.idx2go[idx]]]).astype(int)
#                prediction[anc] = 1
#            rv.append(prediction)
#        return np.array(rv)
#    def propagate(self, predictions):
#        rv = []
#        for pred in predictions:
#            prediction = np.zeros_like(pred)
#            for idx in reversed(range(len(pred))):
#                go = self.dataset.idx2go[idx]
#                successors = list(self.dataset.label_graph.successors(go))
#                score = np.amax([pred[self.dataset.go2idx[i]] for i in successors]) if len(successors) > 0 else 0
#                prediction[idx] = np.amax([prediction[idx],pred[idx], score])
#            rv.append(prediction)
#        return np.array(rv)
#    def get_folds(self, protein_names, labels):
#        fold_file = f"{files['FOLDS']}{self.subgraph}.pkl"
#        if os.path.exists(fold_file):
#            with open(fold_file, 'rb') as f:
#                fold_dict = pickle.load(f)
#            fold_array = np.array([fold_dict[name] for name in protein_names], dtype = bool)
#        else:
#            fold_array = np.where(IterativeStratification(labels, n_splits=config_dict['dataset']['N_FOLDS']) == 0, True, False)
#            fold_dict = {name:isVal for name, isVal in zip(protein_names, fold_array)}
#            with open(fold_file, 'wb') as f:
#                pickle.dump(fold_dict, f)
#        return fold_array
#    @staticmethod
#    def _train_val_split(arr, idx):
#        if isinstance(arr, list):
#            val_arr = [arr[i] for i,j in enumerate(idx) if j]
#            train_arr = [arr[i] for i,j in enumerate(idx) if not j]
#        else:
#            idx = torch.tensor(idx, dtype = torch.bool)
#            val_arr = arr[idx]
#            train_arr = arr[torch.logical_not(idx)]
#        return train_arr, val_arr
#    
#if __name__ == "__main__":
#    main()
