#!/usr/bin/env python3
import torch
import numpy as np
from utils.dataset import Dataset
from models.model import Top, Embeddings, Finetune
from utils.pretrained_embeddings import save_embeddings
from train_test import train, write_predictions
from utils.metric import Metrics

def generate_pretrained_embeddings(device):
    model = Embeddings()
    model.to(device)
    save_embeddings(model, device)

def finetune(device, dataset):
    metric = Metrics(dataset.IA)
    model = Finetune(dataset.class_number)
    model.send(device)
    train(model, device, dataset, metric)

def traintop(device, dataset):
    metric = Metrics(dataset.IA)
    model = Top(dataset.class_number)
    model.to(device)
    train(model, device, dataset, metric)

def eval(device, dataset):
    test_dataset, names = dataset.get_test_dataset()
    """ test_dataset contains (input_ids, attention_masks) names 
    contains name of proteins corresponding to input_ids. """
    write_predictions(model, device, test_dataset)

def manager(device, dataset, **kwargs):
    if kwargs['generate_embeddings']:
        generate_pretrained_embeddings(device)
    if kwargs['train']:
        traintop(device, dataset)
    if kwargs['finetune']:
        finetune(device, dataset)

def main():
    subgraph = "BP"
    options = {'generate_embeddings':False, 'train':True, 'finetune':False}
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dataset = Dataset(subgraph = subgraph)
    manager(device, dataset, **options)

if __name__ == "__main__":
    main()
