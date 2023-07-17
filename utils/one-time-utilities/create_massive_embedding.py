#!/usr/bin/env python
import numpy as np
import pickle

def get_files(embed):
    train = f"./data/embeddings/{embed}_embeds_train.pkl"
    test = f"./data/embeddings/{embed}_embeds_test.pkl"
    return train, test

def get_massive_labels():
    filename = "./data/train/train_massive.tsv"
    with open(filename, 'r') as f:
        names = [i.split('\t')[0] for i in f.readlines()[1:]]
    return set(names)

def main(embeds):
    names = get_massive_labels()
    train_file, test_file = get_files(embeds)
    with open(train_file, 'rb') as f:
        train = pickle.load(f)
    with open(test_file, 'rb') as f:
        test = pickle.load(f)
    for name in names:
        if name not in train:
            train[name] = test[name]
    with open(f"./data/embeddings/{embeds}_more_embeds_train.pkl", 'wb') as f:
        pickle.dump(train, f)
if __name__ == "__main__":
    main("PROTBERT")
    main("T5")
    main("ESM")
