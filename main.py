#!/usr/bin/env python3
import torch
import os
import numpy as np
from utils.dataset import GetDataset
from models.model import Similarity
from models.parts import EMBNet
from train_test import train, create_dataset
import toml

config_dict = toml.load("config.toml")

def get_path(subgraph):
    return f"{config_dict['files']['MODEL_FILE']}{subgraph}.pth"

def create_database(dataset, subgraph, tag):
    model = EMBNet(1280)
    model.load_state_dict(torch.load(get_path(subgraph)))
    model.to(torch.device("cuda"))
    write_to = os.path.join(config_dict['files']['DATABASE'], f"{subgraph}_{tag}.pkl")
    create_dataset(model, dataset, write_to)  

def Train(subgraph):
    dataset = GetDataset(subgraph = subgraph)
    train_dataset, val_dataset = dataset.get_train_val()
    create_database(train_dataset, subgraph, "train")
    create_database(val_dataset, subgraph, "val")
    #node_number = len(dataset.nodes)
    #model = Similarity(node_number = node_number)
    #model.to(torch.device("cuda"))
    #train(model, get_path(dataset.subgraph), dataset)

def main():
    Train('CC')

if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
