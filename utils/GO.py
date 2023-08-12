#!/usr/bin/env python3
import toml
import networkx as nx
from tqdm import tqdm
import torch
import numpy as np

config_dict = toml.load("config.toml")

class GeneOntology:
    Graph = nx.DiGraph()
    def __init__(self, subgraph): 
        assert subgraph in config_dict['gene-ontology']['NAMESPACES']
        self._subgraph = subgraph
        _name = config_dict['gene-ontology']['NAMESPACES'][subgraph]
        terms = [term for term in self._load_terms() if term.namespace == _name]
        self.Graph = self.make_graph(terms)
        self.labels = [(protein,go) for protein, go in self.load_labels() if go in self.Graph]
        self.ancestors, self.descendants = {}, {}
        self.ancestors_and_descendants(self.Graph)
    def _load_terms(self):
        with open(config_dict['gene-ontology']['GO_FILE']) as f:
            raw = f.readlines()
        rv,term = [],[]
        append_state = False
        for i in raw:
            if append_state:
                term.append(i)
            if i == "[Term]\n":
                append_state = True
                continue
            if i == "\n" and term:
                rv.append(term[:-1])
                term = []
                append_state = False
        rv = {GoTerm(term) for term in rv}
        rv = [term for term in rv if not term.is_obsolete]
        return rv
    def make_graph(self, terms):
        return nx.DiGraph([(source,term.go_id) for term in terms for source in term.is_a])
    def ancestors_and_descendants(self, graph):
        def find_ancestors(node):
            if node in self.ancestors:
                return self.ancestors[node]
            predecessors = list(graph.predecessors(node))
            if len(predecessors) == 0:
                self.ancestors[node] = {node}
                return {node}
            ancs = {node}
            for nd in predecessors:
                ancs = ancs.union(find_ancestors(nd))
            self.ancestors[node] = ancs
            return ancs
        def find_descendants(node):
            if node in self.descendants:
                return self.descendants[node]
            successors = list(graph.successors(node))
            if len(successors) == 0:
                self.descendants[node] = {node}
                return {node}
            ancs = {node}
            for nd in successors:
                ancs = ancs.union(find_descendants(nd))
            self.descendants[node] = ancs
            return ancs
        for node in graph:
            find_ancestors(node)
            find_descendants(node)
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

class GoTerm:
    def __init__(self, term_string):
        self.is_obsolete = False
        self.is_a = []
        self.term = term_string
        self.preprocess()
    def preprocess(self):
        assert "id:" in self.term[0]
        self.go_id = int(self.term[0].strip().split(":")[-1])
        for i in self.term:
            if "is_a: " in i:
                entry = int(i.strip().split(" ")[1].split(":")[1])
                self.is_a.append(entry)
            elif "relationship: part_of " in i:
                entry = int(i.strip().split(" ")[2].split(":")[1])
                self.is_a.append(entry)
            elif "is_obsolete: true" in i:
                self.is_obsolete = True
            elif "namespace: " in i:
                namespace = i.strip().split()[-1]
                self.namespace = namespace
            elif "name: " in i:
                self.name = i.strip().split(' ',1)[-1]
    def __hash__(self):
        return hash(self.go_id)
    def __eq__(self, other):
        return self.go_id == other.go_id
