#!/usr/bin/env python3
import torch
import numpy as np
#from utils.dataset import Dataset
from utils.simple import Dataset
from models.model import Top, Embeddings, Finetune
from utils.pretrained_embeddings import save_embeddings
from train_test import train, validator, evaluate, write_predictions, write_top
from utils.metric import Metrics
import toml
from tqdm import tqdm

config_dict = toml.load("config.toml")

def get_path(subgraph):
    return f"{config_dict['files']['MODEL_FILE']}{subgraph}.pth"

def generate_pretrained_embeddings(device, test = False):
    model = Embeddings()
    model.to(device)
    save_embeddings(model, device, test = test)

#def finetune(device, dataset):
#    train_dataset, val_dataset = dataset.get_train_dataset()
#    model = Finetune(dataset.class_number)
#    model.send(device)
#    train(model, get_path(dataset.subgraph), device, train_dataset)

def TrainTogether():
    subgraph = "ALL"
    dataset = Dataset(subgraph = subgraph)
    print(f"{'-'*20} {subgraph} | {dataset.class_number} {'-'*20}")
    train_dataset, val_dataset = dataset.get_train_dataset()
    metrics = Metrics(np.zeros(dataset.class_number))
    model = Top(dataset.class_number)
    model.to(torch.device("cuda"))
    #train(model, get_path(dataset.subgraph), train_dataset, None)
    model.load_state_dict(torch.load(get_path(dataset.subgraph)))
    model.to(torch.device("cuda"))
    labels, predictions = evaluate(model, val_dataset, lambda x:x)
    best, threshold = 0, 0
    for i in tqdm(np.linspace(0.3, 1,100), desc="Tuning threshold"):
        outputs = (predictions>i).astype(bool)
        score = metrics.metrics['F1'](labels, outputs)
        if not np.isnan(score) and score > best:
            best = score
            threshold = i
    outputs = (predictions>threshold).astype(bool)
    print(metrics.eval_and_show(labels, outputs))
    write_top(model, dataset)

def TrainTop():
    subgraphs = list(config_dict['gene-ontology']['NAMESPACES'].keys())
    for subgraph in subgraphs:
        dataset = Dataset(subgraph = subgraph)
        print(f"{'-'*20} {subgraph} | {dataset.class_number} {'-'*20}")
        train_dataset, val_dataset = dataset.get_train_dataset()
        """ train_dataset, val_dataset are torch TensorDatasets
        which contains (input_ids, attention_masks, labels). """
        metrics = Metrics(dataset.IA)
        infer_parents = dataset.propagate
        validate = validator(val_dataset, infer_parents, metrics)
        model = Top(dataset.class_number)
        model.to(torch.device("cuda"))
        train(model, get_path(dataset.subgraph), train_dataset, validate)

def EvaluateAll():
    subgraphs = list(config_dict['gene-ontology']['NAMESPACES'].keys())
    device = torch.device("cuda")
    labels, predictions, all_metrics = {}, {}, {}
    for subgraph in subgraphs:
        dataset = Dataset(subgraph = subgraph)
        _, val_dataset = dataset.get_train_dataset()
        infer_parents = dataset.propagate
        metric = Metrics(dataset.IA)
        model = Top(dataset.class_number)
        model.load_state_dict(torch.load(get_path(dataset.subgraph)))
        model.to(device)
        label, preds = evaluate(model, val_dataset, infer_parents)
        labels[subgraph] = label
        predictions[subgraph] = preds
        all_metrics[subgraph] = metric
    best_score, threshold = 0, 0
    for i in tqdm(np.linspace(0.1, 0.99,100), desc="Tuning threshold"):
        cafa = []
        for subgraph in subgraphs:
            outputs = (predictions[subgraph]>i).astype(bool)
            _score = all_metrics[subgraph].metrics['CAFA Metric'](labels[subgraph],outputs)
            cafa.append(_score)
        score = np.mean(cafa)
        if not np.isnan(score) and score > best_score:
            best_score = score
            threshold = i
    for subgraph in subgraphs:
        outputs = (predictions[subgraph]>threshold).astype(bool)
        print(subgraph, all_metrics[subgraph].eval_and_show(labels[subgraph], outputs))
    return threshold

def Write(threshold):
    subgraphs = list(config_dict['gene-ontology']['NAMESPACES'].keys())
    for subgraph in subgraphs:
        dataset = Dataset(subgraph = subgraph)
        model = Top(dataset.class_number)
        model.load_state_dict(torch.load(get_path(dataset.subgraph)))
        model.to(torch.device("cuda"))
        write_predictions(model, threshold/2, dataset)

def manager(**kwargs):
    if kwargs['generate_embeddings']:
        generate_pretrained_embeddings(device)
    if kwargs['train']:
        TrainTop()
        threshold = EvaluateAll()
        print(threshold)
        Write(threshold)

def main():
    TrainTogether()
    #manager(generate_embeddings = False, train = True)

if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
