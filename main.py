#!/usr/bin/env python3
import torch
import numpy as np
from utils.dataset import GetDataset
from models.model import Node
from train_test import train, write_predictions
from utils.metric import Metrics
import toml

config_dict = toml.load("config.toml")

def get_path(subgraph):
    return f"{config_dict['files']['MODEL_FILE']}{subgraph}.pth"

def Train(subgraph):
    dataset = GetDataset(subgraph = subgraph)
    node_number = len(dataset.nodes)
    model = Node(node_number = node_number)
    model.to(torch.device("cuda"))
    train(model, get_path(dataset.subgraph), dataset)

#    metrics = Metrics()
#    model.load_state_dict(torch.load(get_path(dataset.subgraph)))
#    model.to(torch.device("cuda"))
#    labels, predictions = evaluate(model, val_dataset, lambda x:x)
#    best, threshold = 0, 0
#    for i in tqdm(np.linspace(0.1, 1,150), desc="Tuning threshold"):
#        outputs = (predictions>i).astype(bool)
#        score = metrics.metrics['F1'](labels, outputs)
#        if not np.isnan(score) and score > best:
#            best = score
#            threshold = i
#    outputs = (predictions>threshold).astype(bool)
#    print(metrics.eval_and_show(labels, outputs))
#    write_predictions(model, threshold, dataset)
    #write_top(model, dataset)

#def EvaluateAll():
#    subgraphs = list(config_dict['gene-ontology']['NAMESPACES'].keys())
#    device = torch.device("cuda")
#    labels, predictions, all_metrics = {}, {}, {}
#    for subgraph in subgraphs:
#        dataset = GetDataset(subgraph = subgraph)
#        _, val_dataset = dataset.get_train_dataset()
#        metric = Metrics()
#        model = Node(dataset.class_number)
#        model.load_state_dict(torch.load(get_path(dataset.subgraph)))
#        model.to(device)
#        label, preds = evaluate(model, val_dataset, infer_parents)
#        labels[subgraph] = label
#        predictions[subgraph] = preds
#        all_metrics[subgraph] = metric
#    best_score, threshold = 0, 0
#    for i in tqdm(np.linspace(0.1, 0.99,100), desc="Tuning threshold"):
#        cafa = []
#        for subgraph in subgraphs:
#            outputs = (predictions[subgraph]>i).astype(bool)
#            _score = all_metrics[subgraph].metrics['CAFA Metric'](labels[subgraph],outputs)
#            cafa.append(_score)
#        score = np.mean(cafa)
#        if not np.isnan(score) and score > best_score:
#            best_score = score
#            threshold = i
#    for subgraph in subgraphs:
#        outputs = (predictions[subgraph]>threshold).astype(bool)
#        print(subgraph, all_metrics[subgraph].eval_and_show(labels[subgraph], outputs))
#    return threshold
#
#def Write(threshold):
#    subgraphs = list(config_dict['gene-ontology']['NAMESPACES'].keys())
#    for subgraph in subgraphs:
#        dataset = Dataset(subgraph = subgraph)
#        model = Top(dataset.class_number)
#        model.load_state_dict(torch.load(get_path(dataset.subgraph)))
#        model.to(torch.device("cuda"))
#        write_predictions(model, threshold/2, dataset)
#

def main():
    Train('CC')

if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
