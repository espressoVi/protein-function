#!/usr/bin/env python3
import numpy as np
import toml
import os
from utils.GO import GeneOntology 
import networkx as nx
from tqdm import tqdm

config_dict = toml.load("config.toml")
files = config_dict['files']

class Dataset:
    def __init__(self,*, mode = 'train', device = None, subgraph = None):
        self.device = device
        assert subgraph in config_dict['dataset']['SUB_GRAPHS'] or subgraph is None
        assert mode in ['train','test']
        self.mode = mode
        train_labels = self._load_labels()
        self.label_graph = GeneOntology().full_graph if subgraph is None else GeneOntology().sub_graphs[subgraph]
        self.label_graph = self._graph_prune(self.label_graph,train_labels,)
        self.class_number = len(self.label_graph)
        self.idx2go,self.go2idx = self._idx_to_and_from_go()
        p_dict = self._load_proteins(mode)
        if mode == 'train':
            protein_names,labels = self._get_protein_labels(train_labels)
            proteins = [p_dict[protein_name] for protein_name in protein_names]
        elif mode == 'test':
            protein_names, proteins = [list(i) for i in list(zip(*p_dict.items()))]
    @staticmethod
    def _graph_prune(label_graph, labels):
        """ Removes those nodes from the graph which do not have min_proteins examples of it in train """
        min_proteins = config_dict['dataset']['MIN_PROTEINS']
        _count = {}
        for protein, go in labels:
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
        assert len(list(nx.weakly_connected_components(label_graph))) in [1,len(config_dict['dataset']['SUB_GRAPHS'])]
        return label_graph
    def _idx_to_and_from_go(self):
        idx2go,go2idx = {},{}
        for idx,node in enumerate(nx.topological_sort(self.label_graph)):
            idx2go[idx] = node
            go2idx[node] = idx
        return idx2go,go2idx
    def _get_protein_labels(self, labels):
        """ For each protein in the graph create its multi-label label """
        protein_labels = {}
        for protein,go in tqdm(labels,desc='Processing labels'):
            if go not in self.label_graph:
                continue
            if protein in protein_labels:
                protein_labels[protein].add(go)
            else:
                protein_labels[protein] = set([go])
            protein_labels[protein].update(nx.ancestors(self.label_graph, go))
        proteins,labels = [],[]
        for protein,label in protein_labels.items():
            label_array = np.zeros(self.class_number,dtype = np.bool)
            for node in label:
                label_array[self.go2idx[node]] = 1
            proteins.append(protein)
            labels.append(label_array)
        return proteins, np.array(labels)
    @staticmethod
    def _load_labels():
        """ Reads label file """
        with open(files['TRAIN_LAB'],'r') as f:
            labels = f.readlines()[1:]
        labels = [lab.strip() for lab in labels]
        labels = [lab.split('\t')[:2] for lab in labels]
        labels = [(lab[0],int(lab[1].split(':')[-1])) for lab in labels]
        return labels
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
