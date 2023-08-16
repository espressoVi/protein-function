#!/usr/bin/env python3
import numpy as np
from utils.dataset import GetDataset
from train_test import KNN
import toml

config_dict = toml.load("config.toml")

def main(subgraph):
    dataset = GetDataset(subgraph = subgraph)
    train_dataset, queries = dataset.get()
    knn = KNN(train_dataset)
    predictions = knn.predict(queries)
    idx2go = dataset.idx2go
    res = []
    for key, value in predictions.items():
        for idx in np.where(value>=3.0/7)[0]:
            go = idx2go[idx]
            score = value[idx]
            res.append(f"{key}\tGO:{go:07d}\t{score:.3f}\n")
    with open("all.tsv","a") as f:
        f.writelines("".join(res))

if __name__ == "__main__":
    main("BP")
