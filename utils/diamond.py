#!/usr/bin/env python
import numpy as np
import toml,pickle
import networkx as nx
#from utils.GO import GeneOntology
from GO import GeneOntology
from tqdm import tqdm

config_dict = toml.load("config.toml")

class Diamond:
    def __init__(self, subgraph):
        self.subgraph = subgraph
        self.GO = GeneOntology(subgraph, prune = False)
        self.idx2go, self.go2idx = self._idx_to_and_from_go()
        self.predictions = self.compute_labels()
        self.write(self.predictions)
    def compute_labels(self):
        bit_scores, test_set, train_set = self.get_bit_scores()
        train_labels = self._get_labels()
        preds = {}
        for protein in tqdm(test_set, desc = "Predicting "):
            if protein not in bit_scores:
                continue
            matches = bit_scores[protein]
            final_lab, score_sum = np.zeros(len(self.idx2go)), 0
            for p, score in matches.items():
                final_lab += train_labels[p]*score
                score_sum += score
            final_lab = final_lab/score_sum
            preds[protein] = final_lab
        return preds
    def write(self, predictions):
        terms = []
        for protein, label in predictions.items():
            for idx in np.where(label>0)[0]:
                term = f"GO:{self.idx2go[idx]:07d}"
                s = label[idx]
                terms.append(f"{protein}\t{term}\t{s}\n")
        with open("diamond_predicts.tsv","a") as f:
            f.writelines("".join(terms))
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
            lab = np.zeros(len(self.idx2go), dtype = bool)
            lab[list(label)] = True
            labels[name] = lab
        return labels
    def _idx_to_and_from_go(self):
        idx2go,go2idx = {},{}
        for idx,node in enumerate(nx.topological_sort(self.GO.Graph)):
            idx2go[idx] = node
            go2idx[node] = idx
        return idx2go,go2idx
    def get_bit_scores(self):
        scores = self._read_bit_scores()
        proteins_with_labels = self._get_train_names()
        proteins_to_predict = self._get_test_names() - proteins_with_labels
        bit_scores = {}
        for source, dest, score in scores:
            if source not in proteins_to_predict:
                continue
            if dest not in proteins_with_labels:
                continue
            if source not in bit_scores:
                bit_scores[source] = {dest:score}
            else:
                bit_scores[source].update({dest:score})
        return bit_scores, proteins_to_predict, proteins_with_labels
    def _get_train_names(self):
        return set([protein for protein, _ in self.GO.labels])
    def _get_test_names(self):
        with open(config_dict['files']['EMBEDS_TEST'], 'rb') as f:
            names = set(pickle.load(f).keys())
        return names
    @staticmethod
    def _read_bit_scores():
        file = config_dict['files']['DIAMOND']
        with open(file, 'r') as f:
            raw = [i.split('\t') for i in tqdm(f.readlines(), desc = "Reading DIAMOND ")]
        rv = [(i[0],i[1],float(i[-1].rstrip())) for i in raw]
        return rv
def main():
    Diamond('BP')
    Diamond('CC')
    Diamond('MF')
if __name__ == "__main__":
    main()
