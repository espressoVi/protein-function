#!/usr/bin/env python3
import torch
import os, pickle
import numpy as np
from utils.dataset import GetDataset
from utils.knn import FeatureDataset, KNN
from models.model import Similarity
from models.parts import EMBNet
from train_test import train, create_dataset
import toml
from utils.metric import Metrics

config_dict = toml.load("config.toml")

def get_path(subgraph):
    return f"{config_dict['files']['MODEL_FILE']}{subgraph}.pth"

def create_database(dataset, subgraph, tag):
    model = EMBNet(1280)
    model.load_state_dict(torch.load(get_path(subgraph)))
    model.to(torch.device("cuda"))
    write_to = os.path.join(config_dict['files']['DATABASE'], f"{subgraph}_{tag}.pkl")
    create_dataset(model, dataset, write_to)  

def load_database(dataset, subgraph, tag):
    read_from = os.path.join(config_dict['files']['DATABASE'], f"{subgraph}_{tag}.pkl")
    with open(read_from, "rb") as f:
        embeddings = pickle.load(f)
    names = list(embeddings.keys())
    embeddings = np.array([embeddings[name] for name in names])
    labels = np.array([dataset.labels[name] for name in names]) if dataset.labels is not None else None
    return FeatureDataset(names, embeddings, labels)

def Create(subgraph):
    dataset = GetDataset(subgraph = subgraph)
    train_dataset, val_dataset = dataset.get_train_val()
    create_database(train_dataset, subgraph, "train")
    create_database(val_dataset, subgraph, "val")

def Validate(subgraph):
    dataset = GetDataset(subgraph = subgraph)
    train_dataset, val_dataset = dataset.get_train_val()
    train_database = load_database(train_dataset, subgraph, "train")
    val_database = load_database(val_dataset, subgraph, "val")
    knn = KNN(train_database)
    predictions = knn.predict(val_database)
    actual = val_dataset.labels
    names = list(predictions.keys())
    predicted = np.array([predictions[name] for name in names])
    labels = np.array([actual[name] for name in names])
    print(Metrics().eval_and_show((predicted>0.5).astype(bool), labels))

def Predict(subgraph):
    dataset = GetDataset(subgraph = subgraph)
    train_dataset, val_dataset = dataset.get_train_val()
    test_dataset = dataset.get_test()
    create_database(test_dataset, subgraph, "test")
    train_database = load_database(train_dataset, subgraph, "train")
    test_database = load_database(test_dataset, subgraph, "test")
    knn = KNN(train_database)
    predictions = knn.predict(test_database)
    idx2go = dataset.idx2go
    res = []
    for key, value in predictions.items():
        for idx in np.where(value>0.2)[0]:
            go = idx2go[idx]
            score = value[idx]
            res.append(f"{key}\tGO:{go:07d}\t{score:.3f}\n")
    with open(f"{subgraph}.tsv","w") as f:
        f.writelines("".join(res))

def Train(subgraph):
    dataset = GetDataset(subgraph = subgraph)
    node_number = len(dataset.nodes)
    model = Similarity(node_number = node_number)
    model.to(torch.device("cuda"))
    train(model, get_path(dataset.subgraph), dataset)

def main():
    #subgraph = "CC"
    #Train(subgraph)
    #Create(subgraph)
    #Validate(subgraph)
    #Predict(subgraph)
    subgraph = "BP"
    Train(subgraph)
    Create(subgraph)
    Validate(subgraph)
    Predict(subgraph)

if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
