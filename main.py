#!/usr/bin/env python3
import torch
import numpy as np
from utils.dataset import Dataset
from models.model import ProtBERT
from train_test import train, write_predictions

def manager():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dataset = Dataset(subgraph = 'CC')
    model = ProtBERT(dataset.class_number)
    model.send(device)

    """ train_dataset, val_dataset are torch TensorDatasets
    which contains (input_ids, attention_masks, labels). """
    train_dataset, val_dataset = dataset.get_train_dataset()
    train(model, device, train_dataset, val_dataset)
    """ test_dataset contains (input_ids, attention_masks) names 
    contains name of proteins corresponding to input_ids. """
    test_dataset, names = dataset.get_test_dataset()
    write_predictions(model, device, test_dataset)
    
if __name__ == "__main__":
    manager()
