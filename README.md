# Protein function prediction (CAFA-5)
-----------------------------------------------------
This repository contains code to predict protein functions using deep learning.

-----------------------------------------------------
## Authors
- [Soumadeep Saha](https://www.github.com/espressovi)

-----------------------------------------------------
## To-Do:
* Check if entire path is included in training data.

-----------------------------------------------------
## Done:
* Reading and parsing GO into directed graphs.

-----------------------------------------------------
## Files:
- main.py               -> Runs everything.
- config.toml           -> Configuration file.
- utils
  - utils/GO.py         -> Reads the GO file and creates 3 graphs for CC, MF, BP
  - utils/dataset.py    -> Reads the train/test dataset.
- data
  - extras              -> Directory for extra files.
    - go_basic.obo      -> Gene Ontology file.
    - IA.txt            -> Weights for information accretion loss.
  - test                -> Test superset fasta files, etc
  - train               -> Train fasta file, labels, etc

-----------------------------------------------------
## Usage:
After making sure all required dependencies are installed, try running with.
``` python main.py```
