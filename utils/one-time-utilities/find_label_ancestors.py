import numpy as np
from GO import GeneOntology

def read_labels(subgraph):
    with open("./test_terms_quickGO.tsv","r") as f:
        labels = [i.split('\t')[:2] for i in f.readlines() if subgraph in i]
    labels = [(i[0], int(i[1][3:])) for i in labels]
    return labels

def main():
    subgraph = "MF"
    go = GeneOntology(subgraph)
    labels = read_labels(subgraph)
    labels = [(i,j) for i,j in labels if j in go.Graph]
    all_labs = {}
    for name, node in labels:
        if name in all_labs:
            all_labs[name] = all_labs[name].union(go.ancestors[node])
        else:
            all_labs[name] = go.ancestors[node]
    rv = []
    for name in all_labs.keys():
        for node in all_labs[name]:
            st = f"{name}\tGO:{node:07d}\t{subgraph}O\n"
            rv.append(st)
    with open("test_terms_full.tsv","a") as f:
        f.writelines(rv)

if __name__ == "__main__":
    main()
