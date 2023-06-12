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
    def __init__(self, subgraph = None):
        assert subgraph in config_dict['dataset']['SUB_GRAPHS'] or subgraph is None
        self._train_labels = self._load_labels()
        self.label_graph = self._graph_prune(subgraph,self._train_labels,)
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
    @staticmethod
    def _graph_prune(subgraph, labels):
        """ Removes those nodes from the graph which do not have min_proteins examples of it in train """
        label_graph = GeneOntology().full_graph if subgraph is None else GeneOntology().sub_graphs[subgraph]
        min_proteins = config_dict['dataset']['MIN_PROTEINS']
        _count = {}
        for protein, go in tqdm(labels, desc = "Pruning graph"):
            if go not in label_graph:
                continue
            nodes = {go}.union(nx.ancestors(label_graph,go))
            for node in nodes:
                if node in _count:
                    _count[node].add(protein)
                else:
                    _count[node] = {protein}
        for node in reversed(list(nx.topological_sort(label_graph))):
            if node not in _count:
                label_graph.remove_node(node)
                continue
            if len(_count[node]) >= min_proteins:
                continue
            assert len(nx.descendants(label_graph,node)) == 0
            label_graph.remove_node(node)
        assert len(list(nx.weakly_connected_components(label_graph))) == 1
        #in [1,len(config_dict['dataset']['SUB_GRAPHS'])]
        return label_graph
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
            protein_labels[protein].update(nx.ancestors(self.label_graph, go))
        protein_names,labels = [],[]
        for protein,label in protein_labels.items():
            label_array = np.zeros(self.class_number,dtype = np.bool)
            for node in label:
                label_array[self.go2idx[node]] = 1
            protein_names.append(protein)
            labels.append(label_array)
        return protein_names, np.array(labels)
    @staticmethod
    def _load_labels():
        """ Reads label file """
        with open(files['TRAIN_LAB'],'r') as f:
            labels = f.readlines()[1:]
        rv = []
        for lab in tqdm(labels, desc = "Reading labels"):
            row = lab.strip().split('\t')[:2]
            rv.append((row[0],int(row[1].split(':')[-1])))
        return rv
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
    def get_train_dataset(self):
        _, proteins, labels = self.dataset.get_dataset('train')
        # DELETE FOR ACTUAL RUN
        # proteins = proteins[:5000]
        # labels = labels[:5000]
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
