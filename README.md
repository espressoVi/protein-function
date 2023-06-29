# Protein function prediction (CAFA-5)
This repository contains code to predict protein functions using deep learning.

## Authors
- [Soumadeep Saha](https://www.github.com/espressovi), Indian Statistical Institute, Kolkata, India

## Problem observations
- Gene Ontology
  - There are 43,248 terms 27,942 BP, 11,263 MF, 4,043 CC.
  - In CC there are 3,167 'leaf' nodes.
  - In MF there are 9,222 'leaf' nodes.
- Dataset
  - There are 5,363,863 annotations with 31,520 distinct labels.
  - If we only consider labels with >= 10 annotations, the number of distinct labels drops to 14,901.
  - With rare classes eliminated we have 10,993 BP, 1,470 CC, and 2,438 MF labels.

## Benchmarks
  - CC : MicroF1 = 0.6964 | MacroF1 = 0.3712 | F1 = 0.6993 | Accuracy = 0.5804 | SubsetAccuracy = 0.1140 | Hamming Loss = 0.0155 | CAFAMetric = 0.6152 |
  - BP : MicroF1 = 0.4664 | MacroF1 = 0.2812 | F1 = 0.4416 | Accuracy = 0.3188 | SubsetAccuracy = 0.0091 | Hamming Loss = 0.0122 | CAFAMetric = 0.4393 |
  - MF : MicroF1 = 0.6988 | MacroF1 = 0.4751 | F1 = 0.7216 | Accuracy = 0.6221 | SubsetAccuracy = 0.2456 | Hamming Loss = 0.0083 | CAFAMetric = 0.6825 |

## To-Do:
* Function to write predictions.
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
* IA score implementation.
* Optimized graph pruning.
* Move pruning to GO, start with reduced graph.
* Optimized inference routine (ancestor fill).
* Load embeddings from memory.
* Optimize metric calculations.
* Implement scheduler, and saturate train.

## Files:
- main.py               -> Runs everything.
- config.toml           -> Configuration file.
- train_test.py         -> Implements training/evaluation routines.
- utils
  - utils/GO.py         -> Reads the GO file and creates 3 graphs for CC, MF, BP
  - utils/dataset.py    -> Reads the train/test dataset.
  - utils/metrics.py    -> Implements several basic multi-label metrics.
  - utils/IterativeStratification.py    -> Iterative Stratification algorithm for train/val multi-label split.
- models
  - model.py            -> ProteinBert model with two FC layers (as designed by Debojyoti)
- data
  - extras              -> Directory for extra files.
    - go_basic.obo      -> Gene Ontology file.
    - IA.txt            -> Weights for information accretion loss.
  - test                -> Test superset fasta files, etc
  - train               -> Train fasta file, labels, etc

## Usage:
After making sure all required dependencies are installed, try running with - 

``` python main.py```
