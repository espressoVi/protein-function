# Protein function prediction (CAFA-5) - [NODE EMBEDDINGS].
---
This repository contains code to predict protein functions using deep learning.
We use a graph convolutional network to create embeddings for nodes and match the similarity with embeddings for proteins.

## Authors
- [Soumadeep Saha](https://www.github.com/espressovi), Indian Statistical Institute, Kolkata, India

## To-Do:
* Write inference routine.
* Deal with unseen labels in test.

## Done:
* Reading and parsing GO into directed graphs.
* Collected GO, train stats.
* Prune tree.
* Each protein has the whole path from the "ancestor" to its label node as the label.
* Generate X,Y pair.
* Iterative stratification.
* Tokenize.
* Write train loop.
* Basic metrics.
* Write evaluation loop.
* Train and get baseline results.
* Optimized graph pruning.
* Move pruning to GO, start with reduced graph.
* Optimized inference routine (ancestor fill).
* Load embeddings from memory.
* Optimize metric calculations.
* Implement scheduler, and saturate train.
* Function to write predictions.
* Write graph convolutional NN.
* Re-write dataset code for GCN training.

## Files:
- main.py               -> Runs everything.
- config.toml           -> Configuration file.
- train_test.py         -> Implements training/evaluation routines.
- utils
  - utils/GO.py         -> Reads the GO file and creates 3 graphs for CC, MF, BP
  - utils/dataset.py    -> Reads the train/test dataset.
  - utils/metrics.py    -> Implements several basic multi-label metrics.
- models
  - model.py            -> GCN + protein embedding model.
  - parts.py            -> Contains several smaller pieces for models.
- data
  - extras              -> Directory for extra files.
    - go_basic.obo      -> Gene Ontology file.
    - IA.txt            -> Weights for information accretion loss.
  - test                -> Test superset fasta files, etc
  - train               -> Train fasta file, labels, etc

## Usage:
After making sure all required dependencies are installed, try running with - 

``` python main.py```


## Problem observations
- Gene Ontology
  - There are 43,248 terms 27,942 BP, 11,263 MF, 4,043 CC.
  - In CC there are 3,167 'leaf' nodes.
  - In MF there are 9,222 'leaf' nodes.
- Dataset
  - There are 5,363,863 annotations with 31,520 distinct labels.
  - If we only consider labels with >= 10 annotations, the number of distinct labels drops to 14,901.
  - With rare classes eliminated we have 10,993 BP, 1,470 CC, and 2,438 MF labels.
