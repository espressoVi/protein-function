#!/usr/bin/env python3
import os
import toml
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

config_dict = toml.load("config.toml")['gene-ontology']

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
            elif "is_obsolete: true" in i:
                self.is_obsolete = True
            elif "namespace: " in i:
                namespace = i.strip().split()[-1]
                assert namespace in config_dict['NAMESPACES'].keys()
                self.namespace = namespace
            elif "name: " in i:
                self.name = i.strip().split(' ',1)[-1]
    def __hash__(self):
        return hash(self.go_id)
    def __eq__(self, other):
        return self.go_id == other.go_id

class GeneOntology:
    Graph = nx.DiGraph()
    def __init__(self): 
        self.terms = self._load_terms()
        self.CC = self.make_graph([term for term in self.terms if term.namespace == "cellular_component"])
        self.MF = self.make_graph([term for term in self.terms if term.namespace == "molecular_function"])
        self.BP = self.make_graph([term for term in self.terms if term.namespace == "biological_process"])
    def _load_terms(self):
        with open(config_dict['GO_FILE']) as f:
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
