#!/usr/bin/env python3
import os,re,toml,torch
from utils.GO import GeneOntology 
import networkx as nx
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from utils.IterativeStratification import IterativeStratification
from torch.utils.data import TensorDataset
from time import perf_counter

config_dict = toml.load("config.toml")
files = config_dict['files']

class Preprocess:
    def __init__(self, subgraph):
        self.GO = GeneOntology(subgraph)
        self._train_labels = self.GO.labels
        self.label_graph = self.GO.Graph
        self.class_number = len(self.label_graph)
        self.idx2go,self.go2idx = self._idx_to_and_from_go()
    def get_dataset(self, mode):
        """ Returns lists of protein sequences, their names and their labels (if applicable) """
        p_dict = self._load_proteins(mode)
        if mode == 'test':
            test_protein_names, test_proteins = [list(i) for i in list(zip(*p_dict.items()))]
            return test_protein_names, test_proteins
        train_protein_names,train_labels = self._get_protein_labels(self._train_labels)
        train_proteins = [p_dict[protein_name] for protein_name in train_protein_names]
        return train_protein_names, train_proteins, torch.tensor(train_labels)
    def _idx_to_and_from_go(self):
        idx2go,go2idx = {},{}
        for idx,node in enumerate(nx.topological_sort(self.label_graph)):
            idx2go[idx] = node
            go2idx[node] = idx
        return idx2go,go2idx
    def _get_protein_labels(self, labels):
        """ For each protein in the graph create its multi-label label after pruning."""
        protein_labels = {}
        for protein,go in tqdm(labels,desc='Processing labels'):
            if go not in self.label_graph:
                continue
            if protein in protein_labels:
                protein_labels[protein].add(go)
            else:
                protein_labels[protein] = {go}
            protein_labels[protein].update(self.GO.ancestors[go])
        protein_names,labels = [],[]
        for protein,label in protein_labels.items():
            label_array = np.zeros(self.class_number,dtype = bool)
            for node in label:
                label_array[self.go2idx[node]] = 1
            protein_names.append(protein)
            labels.append(label_array)
        return protein_names, np.array(labels)
    @staticmethod
    def _load_proteins(mode = 'train'):
        """ Reads fasta file and loads sequences. """
        file = files['TRAIN_SEQ'] if mode == 'train' else files['TEST_SEQ']
        sequences = {}
        with open(file,'r') as f:
            raw = f.readlines()
        idx = [i for i,s in enumerate(raw) if '>' in s]
        idx.append(None)
        for start,end in zip(idx[:-1],idx[1:]):
            name = raw[start].split()[0][1:]
            sequence = ''.join([i.strip() for i in raw[start+1:end]])
            sequences[name] = sequence
        return sequences

class Dataset:
    def __init__(self, subgraph = None):
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.dataset = Preprocess(subgraph = subgraph)
        self.class_number = self.dataset.class_number
        self.terms_of_interest = self.dataset.go2idx
    def get_train_dataset(self):
        _, proteins, labels = self.dataset.get_dataset('train')
        # DELETE FOR ACTUAL RUN
        #proteins = proteins[:1000]
        #labels = labels[:1000]
        # UPTO HERE
        proteins = self.tokenize(proteins)
        split = self.get_folds(labels)
        train_input_ids, val_input_ids = self._train_val_split(proteins['input_ids'],split)
        train_attention_masks, val_attention_masks = self._train_val_split(proteins['attention_mask'],split)
        train_labels, val_labels = self._train_val_split(labels, split)
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        return train_dataset, val_dataset
    def get_test_dataset(self):
        protein_names, proteins = self.dataset.get_dataset('test')
        proteins = self.tokenize(proteins)
        test_dataset = TensorDataset(proteins['input_ids'], proteins['attention_mask'])
        return test_dataset, protein_names 
    def tokenize(self, sequences):
        _time_start = perf_counter()
        print("Tokenizing...",end = '\r')
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        sequences = self.tokenizer(sequences, max_length = config_dict['dataset']['MAX_SEQ_LEN'],
                          padding = "max_length", truncation = True, return_tensors = 'pt',)
        print(f"Tokenization took {perf_counter() - _time_start:.2f}s")
        return sequences
    def fill(self, predictions):
        rv = []
        for pred in predictions:
            prediction = np.zeros_like(pred, dtype = bool)
            for idx in np.where(pred == 1)[0]:
                anc = np.array([self.dataset.go2idx[i] for i in self.dataset.GO.ancestors[self.dataset.idx2go[idx]]]).astype(int)
                prediction[anc] = 1
            rv.append(prediction)
        return np.array(rv)
    @staticmethod
    def get_folds(labels):
        return np.where(IterativeStratification(labels, n_splits=config_dict['dataset']['N_FOLDS']) == 0, True, False)
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
