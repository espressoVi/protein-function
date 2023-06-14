#!/usr/bin/env python3
import torch
import numpy as np
from utils.dataset import Dataset
from models.model import ProtBERT
from train_test import train, write_predictions
from utils.metric import Metrics

def manager():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dataset = Dataset(subgraph = 'CC')
    metric = Metrics(dataset)
    model = ProtBERT(dataset.class_number)
    model.send(device)
    train(model, device, dataset, metric)
    """ test_dataset contains (input_ids, attention_masks) names 
    contains name of proteins corresponding to input_ids. """
    test_dataset, names = dataset.get_test_dataset()
    write_predictions(model, device, test_dataset)
    
if __name__ == "__main__":
    manager()
