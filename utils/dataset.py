#!/usr/bin/env python3
import numpy as np
import toml
import os
from utils.GO import GeneOntology 
import networkx as nx
from tqdm import tqdm

files = toml.load("config.toml")['files']

class Dataset:
    def __init__(self, mode = 'train', device = None):
        self.device = device
        assert mode in ['train','test']
        self.mode = mode
        self.label_graph = self.__init__ds()
        self._load_labels()
        _distribution = np.array(sorted([len(attr['proteins']) for node,attr in self.label_graph.nodes(data=True)],reverse=True))
    def _load_labels(self,):
        with open(files['TRAIN_LAB'],'r') as f:
            labs = f.readlines()[1:]
        labels = [lab.strip() for lab in labs]
        labels = [lab.split('\t')[:2] for lab in labels]
        labels = [(lab[0],int(lab[1].split(':')[-1])) for lab in labels]
        for protein,go in tqdm(labels,desc='Processing labels'):
            self.label_graph.nodes[go]['proteins'].add(protein)
            for node in nx.ancestors(self.label_graph,go):
                self.label_graph.nodes[node]['proteins'].add(protein)

    def __init__ds(self):
        graph = GeneOntology().graph
        nx.set_node_attributes(graph, {n:set() for n in graph.nodes()},'proteins')
        return graph

