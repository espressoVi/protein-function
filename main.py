#!/usr/bin/env python3
import torch
import numpy as np
from utils.dataset import Dataset
from models.model import Top, Embeddings, Finetune
from utils.pretrained_embeddings import save_embeddings
from train_test import train, write_predictions
from utils.metric import Metrics

def generate_pretrained_embeddings(device, test = False):
    model = Embeddings()
    model.to(device)
    save_embeddings(model, device, test = test)

def finetune(device, dataset):
    metric = Metrics(dataset.IA)
    model = Finetune(dataset.class_number)
    model.send(device)
    train(model, device, dataset, metric)

def traintop(device, dataset):
    metric = Metrics(dataset.IA)
    model = Top(dataset.class_number)
    model.to(device)
    trained_model, threshold = train(model, device, dataset, metric)
    return trained_model, threshold

def write(device, dataset, trained_model, threshold, **kwargs):
    write_predictions(trained_model, threshold, device, dataset)

def manager(device, dataset, **kwargs):
    if kwargs['generate_embeddings']:
        generate_pretrained_embeddings(device)
    if kwargs['train']:
        trained_model, threshold = traintop(device, dataset)
        print("Threshold : ", threshold)
        write(device, dataset, trained_model, threshold,)
    if kwargs['finetune']:
        finetune(device, dataset)

def main():
    subgraphs = ['CC', 'MF', 'BP']
    options = {'generate_embeddings':False, 'train':True, 'finetune':False}
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    for subgraph in subgraphs:
        print(f"{'-'*20} {subgraph} {'-'*20}")
        dataset = Dataset(subgraph = subgraph, finetune = options['finetune'])
        manager(device, dataset, **options)

if __name__ == "__main__":
    main()
